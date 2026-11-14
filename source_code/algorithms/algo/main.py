import csv
import os
import os.path as osp
from datetime import datetime
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from gym.spaces.box import Box
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor
import random
import multiprocessing as mp
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect, Trajectory, TrajectoryBuffer, ModelBuffer
from algorithms.rcpe import RCPE
import time

def write_output(reward, info, output_dir, tag='train'):
    logging_path = osp.join(output_dir, f'{tag}_output.txt')
    with open(logging_path, 'a') as f:
        f.write('[' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']\n')
        f.write(f""
                f"reward: {'%.3f' % reward} "
                # f"best_{tag}_reward: {'%.3f' % self.best_eval_reward if tag == 'eval' else '%.3f' % self.best_train_reward} "
                f"CSE: {'%.3f' % info['CSE']} "
                f"episodic_aoi: {'%.3f' % info['episodic_aoi']} "
                f"aoi_satis_ratio: {'%.3f' % info['aoi_satis_ratio']} "
                f"coll_satis_ratio: {'%.3f' % info['coll_satis_ratio']} "
                f"data_satis_ratio: {'%.3f' % info['data_satis_ratio']} "
                f"tx_satis_ratio: {'%.3f' % info['tx_satis_ratio']} "
                # f"tx_satis_ratio: {'%.3f' % info['tx_satis_ratio']} "
                # f"soft_tx_satis_ratio: {'%.3f' % info['soft_tx_satis_ratio']} "
                f"energy_consuming: {'%.3f' % info['energy_consuming']} "
                + '\n'
                )

class OnPolicyRunner:
    def __init__(self, logger, agent, envs_learn, envs_test, dummy_env,
                 run_args, alg_args, input_args, **kwargs):
        self.run_args = run_args
        self.input_args = input_args
        self.debug = self.run_args.debug
        self.logger = logger
        self.name = run_args.name
        # agent initialization
        self.agent = agent
        self.num_agent = agent.n_agent
        self.device = self.agent.device if hasattr(self.agent, "device") else "cpu"

        if run_args.checkpoint is not None:  # not train from scratch
            self.agent.load_nets(run_args.checkpoint, best=True)
            logger.log(interaction=run_args.start_step)
        self.start_step = run_args.start_step
        self.env_name = input_args.env
        self.algo_name = input_args.algo
        self.n_thread = input_args.n_thread

        # yyx add
        self.best_episode_reward = float('-inf')
        self.best_test_episode_reward = float('-inf')
        self.best_count = 0

        # algorithm arguments
        self.n_iter = alg_args.n_iter
        self.n_inner_iter = alg_args.n_inner_iter
        self.n_warmup = alg_args.n_warmup if not self.run_args.debug else 1
        self.n_model_update = alg_args.n_model_update
        self.n_model_update_warmup = alg_args.n_model_update_warmup if not self.run_args.debug else 1
        self.n_test = alg_args.n_test
        self.test_interval = alg_args.test_interval
        self.rollout_length = alg_args.rollout_length
        self.use_stack_frame = alg_args.use_stack_frame

        # environment initialization
        self.envs_learn = envs_learn
        self.envs_test = envs_test
        self.dummy_env = dummy_env

        # buffer initialization
        self.model_based = alg_args.model_based
        self.model_batch_size = alg_args.model_batch_size
        if self.model_based:
            self.n_traj = alg_args.n_traj
            self.model_traj_length = alg_args.model_traj_length
            self.model_error_thres = alg_args.model_error_thres
            self.model_buffer = ModelBuffer(alg_args.model_buffer_size)
            self.model_update_length = alg_args.model_update_length
            self.model_validate_interval = alg_args.model_validate_interval
            self.model_prob = alg_args.model_prob
        # 一定注意，PPO并不是在每次调用rollout时reset，一次rollout和是否reset没有直接对应关系
        _, self.episode_len = self.envs_learn.reset(), 0
        self.agent.train_saved_hard_att = []
        self.infer_times = []
        self.train_action = []
        # 每个环境分别记录episode_reward
        self.episode_reward = np.zeros((self.input_args.n_thread))
        self.rcpe = RCPE(input_args, self.device) if input_args.use_rcpe else None
        self.episode_rcpe_reward = np.zeros((self.input_args.n_thread))
        self.episode_crgc_micro_mse_sum = np.zeros(
            self.input_args.n_thread, dtype=np.float64
        )
        self.episode_crgc_micro_mse_count = np.zeros(
            self.input_args.n_thread, dtype=np.int64
        )
        self.episode_crgc_macro_mse_sum = np.zeros(
            self.input_args.n_thread, dtype=np.float64
        )
        self.episode_crgc_macro_mse_count = np.zeros(
            self.input_args.n_thread, dtype=np.int64
        )
        self.crgc_train_history = None
        self._ep_r_csv_path = osp.join(self.run_args.output_dir, 'episode_reward_threads.csv')
        self._ep_pred_mse_csv_path = osp.join(
            self.run_args.output_dir, 'episode_pred_mse.csv'
        )
        self._ep_cse_csv_path = osp.join(self.run_args.output_dir, 'episode_CSE_threads.csv')
        self._train_episode_count = 0  # 训练以来 done 次数（并行块同步结束计 1 次）

        # load pretrained env model when model-based
        self.load_pretrained_model = alg_args.load_pretrained_model
        if self.model_based and self.load_pretrained_model:
            self.agent.load_model(alg_args.pretrained_model)

    def _append_ep_reward_csv(self, ep_r, episode_count):
        """每个 done 追加一行奖励：interaction、episode、各线程本局累计回报 r_i。"""
        path = self._ep_r_csv_path
        interaction = self.logger.buffer.get('interaction')
        row_head = [interaction if interaction is not None else '', int(episode_count)]
        row = row_head + [float(x) for x in ep_r]
        new_file = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['interaction', 'episode'] + [f'r_{i}' for i in range(len(ep_r))])
            w.writerow(row)

    def _append_ep_pred_mse_csv(self, micro_mse, macro_mse, episode_count):
        """Append per-thread episode means for CRGC reconstruction MSE."""
        path = self._ep_pred_mse_csv_path
        interaction = self.logger.buffer.get('interaction')
        row = [interaction if interaction is not None else '', int(episode_count)]
        header = ['interaction', 'episode']
        for thread_id, (micro_value, macro_value) in enumerate(
            zip(micro_mse, macro_mse)
        ):
            header.extend([
                f'micro_mse_t{thread_id}', f'macro_mse_t{thread_id}'
            ])
            row.extend([float(micro_value), float(macro_value)])
        new_file = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)

    def _pack_crgc_history(self, current_obs, history):
        """Return [thread, UAV, (L+1)*frame_dim] and the advanced history."""
        if torch.is_tensor(current_obs):
            current = current_obs.detach().cpu().numpy()
        else:
            current = np.asarray(current_obs)
        current = current.astype(np.float32, copy=False)
        if current.ndim == 2:
            current = current[None, ...]
        if current.ndim != 3 or current.shape[1] != self.num_agent:
            raise ValueError(
                'CRGC expects [thread, UAV, feature], got '
                f'{current.shape}'
            )
        length = int(self.input_args.crgc_history_len)
        expected = (current.shape[0], length, current.shape[1], current.shape[2])
        if history is None or history.shape != expected:
            history = np.zeros(expected, dtype=np.float32)
        sequence = np.concatenate((history, current[:, None]), axis=1)
        packed = sequence.transpose(0, 2, 1, 3).reshape(
            current.shape[0], current.shape[1], -1
        )
        if length:
            next_history = np.concatenate(
                (history[:, 1:], current[:, None]), axis=1
            )
        else:
            next_history = history
        return torch.from_numpy(packed), next_history

    def _append_ep_cse_csv(self, env_info, episode_count):
        """每个 done 追加一行各线程的 CSE。"""
        path = self._ep_cse_csv_path
        cse = [float(d['CSE']) for d in env_info]
        interaction = self.logger.buffer.get('interaction')
        row_head = [interaction if interaction is not None else '', int(episode_count)]
        row = row_head + cse
        new_file = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['interaction', 'episode'] + [f'CSE_{i}' for i in range(len(cse))])
            w.writerow(row)

    def run(self):  # 被launcher.py调用的主循环

        if self.run_args.test:
            self.test()
            return

        self.routine_count = 0
        self.rr = 0
        # for iter in trange(self.n_iter, desc='rollout env'):
        for iter in range(self.n_iter):
            if iter % 50 == 0:
                self.test()
                print('迭代次数======', iter)
            if iter % 1000 == 0:
                self.agent.save_nets(dir_name=self.run_args.output_dir, iter=iter)  # routine
                if self.rcpe is not None:
                    self.rcpe.save(osp.join(
                        self.run_args.output_dir, 'Models', f'RCPE_{iter}.pt'
                    ))

            trajs = self.rollout_env(iter)
            if self.model_based:
                self.model_buffer.storeTrajs(trajs)
                if iter % 10 == 0:
                    self.updateModel()

            agentInfo = []
            # for inner in trange(self.n_inner_iter, desc='inner-iter updateAgent'):
            for inner in range(self.n_inner_iter):
                if self.model_based and np.random.uniform() < self.model_prob:  # Use the model with a certain probability
                    trajs = self.rollout_model(trajs)
                info = self.agent.updateAgent(trajs)
                agentInfo.append(info)
                if self.agent.checkConverged(agentInfo):
                    break

            if not self.input_args.algo == 'Random':
                self.logger.log(inner_iter=inner + 1, iter=iter)

    def test(self):
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        returns = []
        lengths = []

        # for i in trange(self.n_test, desc='test'):
        for i in range(self.n_test):
            done, ep_ret, ep_len = False, np.zeros((1,)), 0  # ep_ret改为分threads存
            envs = self.envs_test
            envs.reset()
            crgc_history = None
            self.agent.test_saved_hard_att = []
            infer_times = []
            while not done:  # 测试时限定一个episode最大为length步
                s = envs.get_obs_from_outside()
                step_count = envs.get_step_count_from_outside()
                if self.input_args.obs_add_action:
                    if step_count[0] == 0:
                        a0 = torch.zeros(1, self.num_agent, 2)
                        s = np.concatenate((s, a0), axis=2)
                    else:
                        s = np.concatenate((s, a), axis=2)
                    s = torch.Tensor(s)
                if self.input_args.use_crgc:
                    s, next_crgc_history = self._pack_crgc_history(
                        s, crgc_history
                    )
                start = time.time()
                a = self.agent.act(s, phase='test')  # shape = (-1, 3)
                end = time.time()
                infer_times.append(end-start)
                # if self.input_args.test: print('inference_time:', end-start)
                # 0221凌晨test改为取概率最大动作
                # action1 = a['branch1'].sample()
                # action2 = a['branch2'].sample()
                action1 = a['branch1'].probs.argmax(dim=-1)
                action2 = a['branch2'].probs.argmax(dim=-1)

                a = torch.stack([action1, action2], dim=-1)
                # if len(a.shape) == 2 and a.shape[0] == 1:  # for IA2C and IC3Net 注意：向量环境下这个需要改！
                #     a = a.squeeze(0)
                a = a.detach().cpu().numpy()  # # shape should be (UAV_NUM, )
                s1, r, done, envs_info = envs.step(a.tolist())
                if self.input_args.use_crgc:
                    crgc_history = next_crgc_history
                done = done.any()
                test_r = np.array(r, dtype=np.float64, copy=True)
                if (not self.input_args.r_th_vio) and self.input_args.reward_add_r_th_vio:
                    p2 = np.stack([
                        np.asarray(
                            d.get('penalty_r2_episode_only', np.zeros(self.num_agent)),
                            dtype=np.float64,
                        )
                        for d in envs_info
                    ])
                    test_r = test_r + p2
                ep_ret += test_r.sum(axis=-1)
                ep_len += 1
                self.logger.log(interaction=None)

            import matplotlib.pyplot as plt
            if self.input_args.test:
                print('average inference time:', np.mean(infer_times))
                # plt.hist(infer_times, bins=20, range=(0.003, 0.005))
                # plt.show()

            if ep_ret.max() > self.best_test_episode_reward:
                max_id = ep_ret.argmax()
                self.best_test_episode_reward = ep_ret.max()
                best_eval_trajs = self.envs_test.get_saved_trajs()
                poi_aoi_history = self.envs_test.get_poi_aoi_history()
                serves = self.envs_learn.get_serves()
                write_output(self.best_test_episode_reward, envs_info[max_id], self.run_args.output_dir, tag='test')
                adj = np.stack(self.agent.test_saved_hard_att, axis=1)[max_id] if self.input_args.algo == 'G2ANet' else None
                self.dummy_env.save_trajs_2(
                    best_eval_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='test', is_newbest=True, adj=adj)

            returns += [ep_ret.sum()]
            lengths += [ep_len]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns.mean(),
                        test_episode_len=lengths.mean(), test_round=None)
        average_ret = returns.mean()
        if self.input_args.debug:
            print(f"{self.n_test} episodes average accumulated reward: {average_ret}")

        return average_ret

    def rollout_env(self, iter):  # 与环境交互得到trajs
        """
        The environment should return sth like [n_agent, dim] or [batch_size, n_agent, dim] in either numpy or torch.
        """
        self.routine_count += 1

        trajBuffer = TrajectoryBuffer(device=self.device)
        envs = self.envs_learn
        for t in range(int(self.rollout_length / self.input_args.n_thread)):  # 加入向量环境后，控制总训练步数不变
            s = envs.get_obs_from_outside()
            step_count = envs.get_step_count_from_outside()
            if self.input_args.obs_add_action:
                if step_count[0] == 0:  ###加上action
                    a0 = torch.zeros(self.input_args.n_thread, self.num_agent, 2)
                    s = np.concatenate((s, a0), axis=2)
                else:
                    s = np.concatenate((s, self.train_action), axis=2)
                s = torch.Tensor(s)
            if self.input_args.use_crgc:
                s, next_crgc_history = self._pack_crgc_history(
                    s, self.crgc_train_history
                )
            start = time.time()
            dist = self.agent.act(s, phase='train')
            end = time.time()
            self.infer_times.append(end-start)
            if self.input_args.use_crgc:
                metrics = self.agent.last_crgc_prediction_mse
                micro_mse = np.asarray(
                    metrics['micro_mse'], dtype=np.float64
                ).reshape(-1)
                micro_count = np.asarray(
                    metrics['micro_count'], dtype=np.int64
                ).reshape(-1)
                macro_mse = np.asarray(
                    metrics['macro_mse'], dtype=np.float64
                ).reshape(-1)
                if (micro_mse.size != self.input_args.n_thread or
                        micro_count.size != self.input_args.n_thread or
                        macro_mse.size != self.input_args.n_thread):
                    raise ValueError(
                        'CRGC prediction metrics must contain one value per '
                        f'thread, got micro={micro_mse.shape}, '
                        f'count={micro_count.shape}, '
                        f'macro={macro_mse.shape}'
                    )
                self.episode_crgc_micro_mse_sum += micro_mse * micro_count
                self.episode_crgc_micro_mse_count += micro_count
                self.episode_crgc_macro_mse_sum += macro_mse
                self.episode_crgc_macro_mse_count += 1
            a = []
            logp = []
            for key in ['branch1', 'branch2']:
                a_tmp = dist[key].sample()
                logp_tmp = dist[key].log_prob(a_tmp)
                a.append(a_tmp)
                logp.append(logp_tmp)
            a = torch.stack(a, dim=-1)
            logp = torch.stack(logp, dim=-1)

            a = a.detach().cpu().numpy()
            s1, r, done, env_info = envs.step(a.tolist())
            if self.input_args.obs_add_action:
                s1 = np.concatenate((s1, a), axis=2) #s1加at
            if self.input_args.use_crgc:
                s1, _ = self._pack_crgc_history(s1, next_crgc_history)
                self.crgc_train_history = next_crgc_history
            self.train_action = a
            done_flags = np.asarray(done, dtype=bool).reshape(-1)
            policy_reward = np.asarray(r, dtype=np.float32).copy()
            if self.rcpe is not None:
                intrinsic_reward, rcpe_stats = self.rcpe.process_step(
                    env_info, a, r, done_flags
                )
                policy_reward += (
                    self.input_args.rcpe_lambda
                    * intrinsic_reward[:, None]
                    / self.num_agent
                )
                self.episode_rcpe_reward += intrinsic_reward
                log_values = {
                    'rcpe_intrinsic_reward': rcpe_stats['intrinsic_mean'],
                    'rcpe_intrinsic_reward_max': rcpe_stats['intrinsic_max'],
                    'rcpe_replay_size': rcpe_stats['replay_size'],
                    'rcpe_priority_entropy': rcpe_stats['priority_entropy'],
                    'rcpe_return_clip_rate': rcpe_stats['return_clip_rate'],
                }
                if 'q_loss' in rcpe_stats:
                    log_values['rcpe_q_loss'] = rcpe_stats['q_loss']
                if rcpe_stats['priority_updated']:
                    log_values['rcpe_priority_update'] = self.rcpe.priority_updates
                    log_values['rcpe_priority_max'] = float(self.rcpe.priority.max())
                self.logger.log(**log_values)
            done = done_flags.any()
            trajBuffer.store(s, a, policy_reward, s1,
                             np.repeat(done_flags[:, None], self.num_agent, axis=1),
                             logp)
            episode_r = np.array(r, dtype=np.float64, copy=True)
            if (not self.input_args.r_th_vio) and self.input_args.reward_add_r_th_vio:
                p2 = np.stack([
                    np.asarray(
                        d.get('penalty_r2_episode_only', np.zeros(self.num_agent)),
                        dtype=np.float64,
                    )
                    for d in env_info
                ])
                episode_r = episode_r + p2
            assert episode_r.ndim > 1
            episode_r = episode_r.sum(axis=-1)  # 对各agent奖励求和
            self.episode_reward += episode_r
            self.episode_len += 1
            self.logger.log(interaction=None)

            if done:
                self._train_episode_count += 1
                ep_r = self.episode_reward
                # ep_r = np.asarray(self.episode_reward, dtype=np.float64).copy()
                self._append_ep_reward_csv(ep_r, self._train_episode_count)
                self._append_ep_cse_csv(env_info, self._train_episode_count)
                if self.input_args.use_crgc:
                    micro_episode_mse = np.divide(
                        self.episode_crgc_micro_mse_sum,
                        self.episode_crgc_micro_mse_count,
                        out=np.zeros_like(self.episode_crgc_micro_mse_sum),
                        where=self.episode_crgc_micro_mse_count > 0,
                    )
                    macro_episode_mse = np.divide(
                        self.episode_crgc_macro_mse_sum,
                        self.episode_crgc_macro_mse_count,
                        out=np.zeros_like(self.episode_crgc_macro_mse_sum),
                        where=self.episode_crgc_macro_mse_count > 0,
                    )
                    self._append_ep_pred_mse_csv(
                        micro_episode_mse,
                        macro_episode_mse,
                        self._train_episode_count,
                    )
                    self.logger.log(
                        mean_episode_crgc_micro_mse=float(
                            micro_episode_mse.mean()
                        ),
                        mean_episode_crgc_macro_mse=float(
                            macro_episode_mse.mean()
                        ),
                        **{
                            f'episode_crgc_micro_mse/t{i}': float(value)
                            for i, value in enumerate(micro_episode_mse)
                        },
                        **{
                            f'episode_crgc_macro_mse/t{i}': float(value)
                            for i, value in enumerate(macro_episode_mse)
                        },
                    )
                self.logger.log(
                    **{f'episode_reward/t{i}': float(ep_r[i]) for i in range(len(ep_r))},
                    episode_reward_std=float(ep_r.std()),
                    episode_reward_min=float(ep_r.min()),
                )
                if self.rcpe is not None:
                    self.logger.log(
                        mean_episode_rcpe_reward=float(self.episode_rcpe_reward.mean()),
                        max_episode_rcpe_reward=float(self.episode_rcpe_reward.max()),
                    )
                if self.input_args.debug:
                    print('train episode reward:', ep_r)
                self.logger.log(mean_episode_reward=ep_r.mean(), episode_len=self.episode_len, episode=None)
                self.logger.log(max_episode_reward=ep_r.max(), episode_len=self.episode_len, episode=None)
                if ep_r.max() > self.best_episode_reward:
                    max_id = ep_r.argmax()
                    self.best_episode_reward = ep_r.max()
                    self.best_count += 1
                    self.agent.save_nets(dir_name=self.run_args.output_dir, is_newbest=True)
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                    serves = self.envs_learn.get_serves()
                    write_output(self.best_episode_reward, env_info[max_id], self.run_args.output_dir)
                    adj = np.stack(self.agent.train_saved_hard_att, axis=1)[max_id] if self.input_args.algo == 'G2ANet' else None
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id], phase='train', is_newbest=True, adj=adj, best_count=self.best_count)


                if self.routine_count // 500 > 0:  # routinely vis (OK)
                    max_id = ep_r.argmax()
                    best_train_trajs = self.envs_learn.get_saved_trajs()
                    poi_aoi_history = self.envs_learn.get_poi_aoi_history()
                    serves = self.envs_learn.get_serves()
                    adj = np.stack(self.agent.train_saved_hard_att, axis=1)[max_id] if self.input_args.algo == 'G2ANet' else None
                    self.dummy_env.save_trajs_2(
                        best_train_trajs[max_id], poi_aoi_history[max_id], serves[max_id],
                        iter=self.rr*500, phase='train', adj=adj)
                    self.rr += self.routine_count // 500
                    self.routine_count = self.routine_count % 500

                self.logger.log(CSE=sum(d['CSE'] for d in env_info) / len(env_info),
                                episodic_aoi=sum(d['episodic_aoi'] for d in env_info) / len(env_info),
                                aoi_satis_ratio=sum(d['aoi_satis_ratio'] for d in env_info) / len(env_info),
                                data_satis_ratio=sum(d['data_satis_ratio'] for d in env_info) / len(env_info),
                                coll_satis_ratio=sum(d['coll_satis_ratio'] for d in env_info) / len(env_info),
                                tx_satis_ratio=sum(d['tx_satis_ratio'] for d in env_info) / len(env_info),
                                energy_consuming=sum(d['energy_consuming'] for d in env_info) / len(env_info),
                                good_reward=sum(d['good_reward'] for d in env_info) / len(env_info),
                                aoi_penalty_reward=sum(d['aoi_penalty_reward'] for d in env_info) / len(env_info),
                                knn_reward=sum(d['knn_reward'] for d in env_info) / len(env_info),
                                energy_reward=sum(d['energy_reward'] for d in env_info) / len(env_info),
                                )
                '''执行env的reset'''
                try:
                    if self.debug:
                        print('average inference time:', np.mean(self.infer_times))
                    _, self.episode_len = self.envs_learn.reset(), 0
                    self.agent.train_saved_hard_att = []
                    self.infer_times = []
                    self.episode_reward = np.zeros((self.input_args.n_thread))
                    self.episode_rcpe_reward = np.zeros((self.input_args.n_thread))
                    self.episode_crgc_micro_mse_sum.fill(0.0)
                    self.episode_crgc_micro_mse_count.fill(0)
                    self.episode_crgc_macro_mse_sum.fill(0.0)
                    self.episode_crgc_macro_mse_count.fill(0)
                    self.crgc_train_history = None
                except Exception as e:
                    raise NotImplementedError

        return trajBuffer.retrieve()

