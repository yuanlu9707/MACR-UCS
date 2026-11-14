"""RCPE regional temporal exploration module.

RCPE learns a categorical model of the H-step *extrinsic team return* and
periodically measures how much that distribution changes when one region is
replaced by its own recent historical state.  The resulting regional priority
weights gate a semantic hash-count novelty bonus.

This module is deliberately independent from the PPO trajectory buffer.  In
particular, its return model never observes the intrinsic reward it produces.
"""

from collections import Counter, deque
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReturnReplay:
    def __init__(self, capacity, state_dim, n_agent):
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, n_agent, 2), dtype=np.int16)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, target_return):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.returns[self.ptr] = target_return
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.randint(0, self.size, size=int(batch_size))
        return self.states[indices], self.actions[indices], self.returns[indices]


class _ReturnDistributionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(self, x):
        return self.net(x)


class RCPE:
    """Training-only regional temporal exploration module."""

    REGION_FEATURES = 6

    def __init__(self, input_args, device):
        self.device = torch.device(device)
        self.n_thread = int(input_args.n_thread)
        self.n_agent = int(input_args.uav_num)
        self.grid_size = int(input_args.map_size)
        self.n_region = self.grid_size * self.grid_size
        self.poi_num = int(input_args.poi_num)
        self.max_episode_step = int(input_args.max_episode_step)
        self.service_action_dim = int(input_args.update_num)

        self.horizon = int(input_args.rcpe_horizon)
        self.gamma = float(input_args.rcpe_gamma)
        self.anchor_batch = int(input_args.rcpe_b)
        self.action_samples = int(input_args.rcpe_l)
        self.lags = tuple(int(x) for x in input_args.rcpe_lags.split(',') if x.strip())
        self.update_interval = int(input_args.rcpe_update_interval)
        self.warmup = int(input_args.rcpe_warmup)
        self.batch_size = int(input_args.rcpe_batch_size)
        self.train_every = int(input_args.rcpe_train_every)
        self.train_steps = int(input_args.rcpe_train_steps)
        self.predict_batch_size = int(input_args.rcpe_predict_batch_size)
        self.n_bins = int(input_args.rcpe_reward_bins)
        self.reward_min = float(input_args.rcpe_reward_min)
        self.reward_max = float(input_args.rcpe_reward_max)
        self.divergence = input_args.rcpe_divergence
        self.eps = float(input_args.rcpe_eps)
        self.priority_ema = float(input_args.rcpe_priority_ema)
        self.priority_floor = float(input_args.rcpe_priority_floor)

        if not self.lags or min(self.lags) < 1:
            raise ValueError('rcpe_lags must contain positive integers')
        if self.reward_min >= self.reward_max:
            raise ValueError('rcpe_reward_min must be smaller than rcpe_reward_max')
        if self.n_bins < 2:
            raise ValueError('rcpe_reward_bins must be at least 2')

        # Region features plus centralized UAV positions/energy and time index.
        self.context_dim = self.n_agent * 3 + 1
        self.state_dim = self.n_region * self.REGION_FEATURES + self.context_dim
        self.action_dim = self.n_agent * (9 + self.service_action_dim)
        model_input_dim = self.state_dim + self.action_dim

        self.model = _ReturnDistributionNet(
            model_input_dim, int(input_args.rcpe_hidden_dim), self.n_bins
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(input_args.rcpe_lr)
        )
        self.bin_edges = torch.linspace(
            self.reward_min, self.reward_max, self.n_bins + 1, device=self.device
        )

        seed = int(getattr(input_args, 'rcpe_seed', 1))
        self.rng = np.random.RandomState(seed)
        self.replay = _ReturnReplay(
            input_args.rcpe_replay_size, self.state_dim, self.n_agent
        )
        self.pending = [deque() for _ in range(self.n_thread)]
        history_size = max(int(input_args.rcpe_history_size), max(self.lags) + 1)
        self.history = [deque(maxlen=history_size) for _ in range(self.n_thread)]
        self.episode_ids = np.zeros(self.n_thread, dtype=np.int64)

        self.visit_counts = [Counter() for _ in range(self.n_region)]
        self.count_bins = self._parse_bins(input_args.rcpe_count_bins)
        self.aoi_bins = self._parse_bins(input_args.rcpe_aoi_bins)
        self.tau_bins = self._parse_bins(input_args.rcpe_tau_bins)

        self.priority = np.full(self.n_region, 1.0 / self.n_region, dtype=np.float32)
        self.env_steps = 0
        self.priority_updates = 0
        self.return_count = 0
        self.clipped_return_count = 0
        self.last_loss = None

    @staticmethod
    def _parse_bins(value):
        bins = [float(x) for x in value.split(',') if x.strip()]
        if bins != sorted(bins):
            raise ValueError('RCPE semantic bin boundaries must be sorted')
        return np.asarray(bins, dtype=np.float32)

    def _flat_state(self, regions, context):
        return np.concatenate((
            np.asarray(regions, dtype=np.float32).reshape(-1),
            np.asarray(context, dtype=np.float32).reshape(-1),
        ))

    def _encode_actions(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 2:
            actions = actions[None, ...]
        batch = actions.shape[0]
        encoded = np.zeros((batch, self.action_dim), dtype=np.float32)
        stride = 9 + self.service_action_dim
        for uav_id in range(self.n_agent):
            move = np.clip(actions[:, uav_id, 0], 0, 8)
            service = np.clip(
                actions[:, uav_id, 1], 0, self.service_action_dim - 1
            )
            rows = np.arange(batch)
            encoded[rows, uav_id * stride + move] = 1.0
            encoded[rows, uav_id * stride + 9 + service] = 1.0
        return encoded

    def _return_bin_targets(self, returns):
        values = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # Interior edges yield indices in [0, n_bins - 1]; tails go to edge bins.
        return torch.bucketize(values, self.bin_edges[1:-1])

    def _train_model(self):
        # Train throughout the warm-up period; only priority estimation waits
        # for ``warmup`` complete H-step targets.
        if self.replay.size < self.batch_size:
            return None
        losses = []
        self.model.train()
        for _ in range(self.train_steps):
            states, actions, returns = self.replay.sample(self.batch_size, self.rng)
            action_features = self._encode_actions(actions)
            inputs = np.concatenate((states, action_features), axis=-1)
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            targets = self._return_bin_targets(returns)
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def _predict(self, states, actions):
        action_features = self._encode_actions(actions)
        inputs = np.concatenate((states, action_features), axis=-1).astype(np.float32)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(inputs), self.predict_batch_size):
                batch = torch.as_tensor(
                    inputs[start:start + self.predict_batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(torch.softmax(self.model(batch), dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _finalize_oldest(self, thread_id, length):
        queue = self.pending[thread_id]
        rewards = [queue[i]['reward'] for i in range(min(length, len(queue)))]
        target_return = sum((self.gamma ** i) * reward for i, reward in enumerate(rewards))
        first = queue[0]
        self.replay.add(first['state'], first['action'], target_return)
        self.return_count += 1
        if target_return < self.reward_min or target_return > self.reward_max:
            self.clipped_return_count += 1
        queue.popleft()

    def _store_transition(self, thread_id, regions, context, action, reward, done):
        state = self._flat_state(regions, context)
        queue = self.pending[thread_id]
        queue.append({'state': state, 'action': action, 'reward': float(reward)})
        if len(queue) >= self.horizon:
            self._finalize_oldest(thread_id, self.horizon)
        if done:
            while queue:
                self._finalize_oldest(thread_id, len(queue))

        self.history[thread_id].append({
            'regions': np.asarray(regions, dtype=np.float32).copy(),
            'context': np.asarray(context, dtype=np.float32).copy(),
            'episode': int(self.episode_ids[thread_id]),
        })
        if done:
            self.episode_ids[thread_id] += 1

    def _semantic_key(self, region):
        region = np.asarray(region)
        n_user = int(round(float(region[0]) * self.poi_num))
        n_risk = int(round(float(region[3]) * self.poi_num))
        n_vio = int(round(float(region[4]) * self.poi_num))
        tau = float(region[5]) * self.max_episode_step
        return (
            int(np.digitize(n_user, self.count_bins)),
            int(np.digitize(region[1], self.aoi_bins)),
            int(np.digitize(region[2], self.aoi_bins)),
            int(np.digitize(n_risk, self.count_bins)),
            int(np.digitize(n_vio, self.count_bins)),
            int(np.digitize(tau, self.tau_bins)),
        )

    def _intrinsic_rewards(self, post_regions, visited_masks):
        rewards = np.zeros(self.n_thread, dtype=np.float32)
        pending_increments = []
        for thread_id in range(self.n_thread):
            visited = np.flatnonzero(np.asarray(visited_masks[thread_id]) > 0)
            for region_id in visited:
                key = self._semantic_key(post_regions[thread_id][region_id])
                count = self.visit_counts[region_id][key]
                novelty = 1.0 / np.sqrt(count + 1.0)
                rewards[thread_id] += self.priority[region_id] * novelty
                pending_increments.append((region_id, key))
        # All vector environments at the same time step see the pre-update counts.
        for region_id, key in pending_increments:
            self.visit_counts[region_id][key] += 1
        return rewards

    def _valid_anchors(self):
        candidates = []
        max_lag = max(self.lags)
        for thread_id, records_deque in enumerate(self.history):
            records = list(records_deque)
            for index in range(max_lag, len(records)):
                episode = records[index]['episode']
                if all(records[index - lag]['episode'] == episode for lag in self.lags):
                    candidates.append((thread_id, index, records))
        return candidates

    def _sample_action_bank(self):
        actions = np.zeros((self.action_samples, self.n_agent, 2), dtype=np.int64)
        actions[:, :, 0] = self.rng.randint(
            0, 9, size=(self.action_samples, self.n_agent)
        )
        actions[:, :, 1] = self.rng.randint(
            0, self.service_action_dim,
            size=(self.action_samples, self.n_agent),
        )
        return actions

    def _distribution_distance(self, counterfactual, normal):
        if self.divergence == 'js':
            middle = 0.5 * (counterfactual + normal)
            distance = 0.5 * np.sum(
                counterfactual * np.log((counterfactual + self.eps) / (middle + self.eps)),
                axis=-1,
            ) + 0.5 * np.sum(
                normal * np.log((normal + self.eps) / (middle + self.eps)),
                axis=-1,
            )
        else:
            distance = np.sum(
                counterfactual * np.log((counterfactual + self.eps) / (normal + self.eps)),
                axis=-1,
            )
        # KL/JS are non-negative analytically; remove tiny negative round-off.
        return np.maximum(distance, 0.0)

    def _update_priority(self):
        if self.replay.size < max(self.warmup, self.batch_size):
            return False
        candidates = self._valid_anchors()
        if len(candidates) < self.anchor_batch:
            return False
        selected_ids = self.rng.choice(
            len(candidates), size=self.anchor_batch, replace=False
        )
        selected = [candidates[i] for i in selected_ids]
        action_bank = self._sample_action_bank()

        normal_states = []
        for _, index, records in selected:
            anchor = records[index]
            state = self._flat_state(anchor['regions'], anchor['context'])
            normal_states.extend([state] * self.action_samples)
        normal_actions = np.tile(action_bank, (self.anchor_batch, 1, 1))
        normal_probs = self._predict(
            np.asarray(normal_states), normal_actions
        ).reshape(self.anchor_batch, self.action_samples, self.n_bins).mean(axis=1)

        counterfactual_states = []
        counterfactual_actions = []
        for _, index, records in selected:
            anchor = records[index]
            for region_id in range(self.n_region):
                for lag in self.lags:
                    changed_regions = anchor['regions'].copy()
                    changed_regions[region_id] = records[index - lag]['regions'][region_id]
                    state = self._flat_state(changed_regions, anchor['context'])
                    counterfactual_states.extend([state] * self.action_samples)
                    counterfactual_actions.extend(action_bank)

        counterfactual_probs = self._predict(
            np.asarray(counterfactual_states), np.asarray(counterfactual_actions)
        ).reshape(
            self.anchor_batch,
            self.n_region,
            len(self.lags),
            self.action_samples,
            self.n_bins,
        ).mean(axis=3)

        normal = normal_probs[:, None, None, :]
        distances = self._distribution_distance(counterfactual_probs, normal)
        raw_priority = np.maximum(distances.mean(axis=(0, 2)), 0.0)
        total = float(raw_priority.sum())
        if not np.isfinite(total) or total <= self.eps:
            normalized = np.full(self.n_region, 1.0 / self.n_region, dtype=np.float32)
        else:
            normalized = (raw_priority / total).astype(np.float32)

        if self.priority_floor > 0:
            normalized = (
                (1.0 - self.priority_floor) * normalized
                + self.priority_floor / self.n_region
            )
        if self.priority_ema > 0:
            normalized = (
                self.priority_ema * self.priority
                + (1.0 - self.priority_ema) * normalized
            )
            normalized /= normalized.sum()
        normalized = np.maximum(normalized, 0.0)
        normalized /= normalized.sum()
        self.priority = normalized.astype(np.float32)
        self.priority_updates += 1
        return True

    def process_step(self, env_infos, actions, extrinsic_rewards, dones):
        """Consume one vector-environment step and return one bonus per thread."""
        pre_regions = np.stack([info['rcpe_pre_regions'] for info in env_infos])
        pre_context = np.stack([info['rcpe_pre_context'] for info in env_infos])
        post_regions = np.stack([info['rcpe_post_regions'] for info in env_infos])
        visited_masks = np.stack([info['rcpe_visited'] for info in env_infos])
        team_rewards = np.asarray(extrinsic_rewards, dtype=np.float32).sum(axis=-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)

        intrinsic = self._intrinsic_rewards(post_regions, visited_masks)
        for thread_id in range(self.n_thread):
            self._store_transition(
                thread_id,
                pre_regions[thread_id],
                pre_context[thread_id],
                np.asarray(actions[thread_id]),
                team_rewards[thread_id],
                bool(dones[thread_id]),
            )

        self.env_steps += 1
        loss = None
        if self.env_steps % self.train_every == 0:
            loss = self._train_model()
        priority_updated = False
        if self.env_steps % self.update_interval == 0:
            priority_updated = self._update_priority()

        stats = {
            'intrinsic_mean': float(intrinsic.mean()),
            'intrinsic_max': float(intrinsic.max()),
            'replay_size': int(self.replay.size),
            'priority_updated': priority_updated,
            'priority_entropy': float(-np.sum(
                self.priority * np.log(self.priority + self.eps)
            )),
            'return_clip_rate': (
                float(self.clipped_return_count / self.return_count)
                if self.return_count else 0.0
            ),
        }
        if loss is not None:
            stats['q_loss'] = loss
        return intrinsic, stats

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'priority': self.priority,
            'env_steps': self.env_steps,
            'priority_updates': self.priority_updates,
        }, path)
