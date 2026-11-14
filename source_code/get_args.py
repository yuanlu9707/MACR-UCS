import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json

    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )
    parser.add_argument('--test-save-heatmap', action='store_true')
    parser.add_argument('--user', type=str, default='yyx')
    parser.add_argument('--env', type=str, default='Mobile')
    parser.add_argument('--algo', type=str, choices=['IPPO'], default='IPPO',
                        help='IPPO/MAPPO backbone with CRGC and optional RCPE')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument("--dataset", type=str, default='roma', choices=['NCSU', 'roma'])
    parser.add_argument("--poi_num", type=int, default=48)
    parser.add_argument("--tag", type=str, default='', help='每个单独实验的备注')
    # dirs
    parser.add_argument("--output_dir", type=str, default='runs/debug', help="which fold to save under 'runs/'")
    parser.add_argument('--group', type=str, default='debug', help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    # system stub
    parser.add_argument('--mute_wandb', default=True, action='store_false')  # 后期服务器联网不稳定 网断了进程就会sleep 不要用wandb！
    # tune agent
    parser.add_argument('--checkpoint', type=str)  # load pretrained model
    parser.add_argument('--n_thread', type=int, default=16)
    # parser.add_argument('--n_thread', type=int, default=16)
    parser.add_argument('--n_iter', type=int, default=5000)
    # tune algo
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_v', type=float)
    parser.add_argument('--lr_colla', type=float)
    parser.add_argument('--use-stack-frame', action='store_true')
    # parser.add_argument('--use_extended_value', action='store_false', help='反逻辑，仅用于DPPO')
    # parser.add_argument('--use-mlp-model', action='store_true', help='将model改为最简单的mlp，仅用于DMPO')
    # parser.add_argument('--multi-mlp', action='store_true', help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--g2a_hidden_dim', type=int, default=64, help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--map_size', type=int, default=6)  # hyper
    parser.add_argument('--g2a_hops', type=int, default=1)  # hyper
    parser.add_argument('--update_colla_by_v_0307', action='store_true')  # hyper

    # tune env
    ## setting
    parser.add_argument('--fixed-range', action='store_false')  # 重要，sensing range现在固定了
    parser.add_argument('--collect_range', type=float, default=500)
    parser.add_argument('--dyna_level', type=str, default='2', help='指明读取不同难度的poi_QoS.npy')
    parser.add_argument('--init_energy', type=float, default=1000000) #719280
    parser.add_argument('--w_noise', type=float, default=-107)  # 0222morning determined 用不到
    parser.add_argument('--user_data_amount', type=float, default=3) #单位-1Mbits
    parser.add_argument('--update_num', type=int, default=10)  # 两个数据集统一将default设为10

    parser.add_argument('--uav_num', type=int, default=5)
    parser.add_argument('--fixed-col-time', action='store_false')
    parser.add_argument('--aoith', default=30, type=int)  # 0222morning determined
    parser.add_argument('--txth', default=1, type=int) #传输要求，没啥意思，可改
    parser.add_argument('--r_th_vio', default=True) #r-th_vio 吞吐量满意率的奖励惩罚
    parser.add_argument('--reward_add_r_th_vio', default=False)
    parser.add_argument('--uav_height', default=100, type=int)
    parser.add_argument('--hao02191630', action='store_false')
    parser.add_argument('--always_fixed_antenna02230040', default=-1, type=int)

    ## MDP
    parser.add_argument('--max_episode_step', type=int, default=120)
    parser.add_argument('--MAPPO', default=False)
    parser.add_argument('--obs_add_action', default=False)
    parser.add_argument('--future_obs', type=int, default=0)
    parser.add_argument('--use_snrmap', action='store_true', default=False)  # shrotcut is always used
    parser.add_argument('--high_level_dont_use_snrmap', action='store_true')  #
    parser.add_argument('--high_level_knn_coefficient', type=float, default=-1)  #
    parser.add_argument('--aVPS', type=float, default=0.2)
    parser.add_argument('--tVPS', type=float, default=0.2)
    parser.add_argument('--agent_field', type=float, default=500)

    # CRGC: reconstruction-assisted multi-UAV communication.
    parser.add_argument('--use_crgc', '--crgc', dest='use_crgc',
                        action='store_true', default=True)
    parser.add_argument('--crgc_history_len', '--crgc_l', dest='crgc_history_len',
                        type=int, default=5,
                        help='number L of past observation frames; CRGC uses L+1 frames')
    parser.add_argument('--crgc_risk_margin', type=float, default=3.0,
                        help='risk threshold is aoith minus this margin')
    parser.add_argument('--crgc_distance_lambda', type=float, default=1.0,
                        help='coefficient of the negative normalized-distance attention bias')
    parser.add_argument('--crgc_feature_dim', type=int, default=64)
    parser.add_argument('--crgc_message_dim', type=int, default=64,
                        help='dimension of the communication message received by each UAV')
    parser.add_argument('--crgc_region_embed_dim', type=int, default=8)
    parser.add_argument('--crgc_continuous_dim', type=int, default=32,
                        help='dimension used to encode joint continuous entity attributes')
    parser.add_argument('--crgc_attention_heads', type=int, default=2)
    parser.add_argument('--crgc_transformer_layers', type=int, default=1)
    parser.add_argument('--crgc_transformer_ff_dim', type=int, default=64)
    parser.add_argument('--crgc_alignment_temperature', type=float, default=0.1)
    parser.add_argument('--crgc_physical_temperature', type=float, default=0.02)
    parser.add_argument('--crgc_observed_aoi_scale', type=float, default=120.0,
                        help='inverse of the AoI normalization used by get_obs_agent')
    parser.add_argument('--crgc_use_mse_loss', dest='crgc_use_mse_loss',
                        action='store_true',
                        help='use MSE for reconstruction/physical losses (default)')
    parser.add_argument('--crgc_use_smooth_l1_loss', dest='crgc_use_mse_loss',
                        action='store_false',
                        help='use SmoothL1 instead of MSE for reconstruction/physical losses')
    parser.set_defaults(crgc_use_mse_loss=True)
    parser.add_argument('--use_mse_weight', default=False, help='risk-weight the visible-user microscopic MSE (default)')
    parser.add_argument('--mse_risk_alpha', type=float, default=1.0,
                        help='extra reconstruction weight for users near the AoI threshold')
    parser.add_argument('--mse_violation_alpha', type=float, default=2.0,
                        help='extra reconstruction weight for users at or above the AoI threshold')
    parser.add_argument('--crgc_micro_loss_weight', type=float, default=1.0)
    parser.add_argument('--crgc_macro_loss_weight', type=float, default=1.0)
    parser.add_argument('--crgc_physical_loss_weight', type=float, default=1.0)
    parser.add_argument('--crgc_e2r_loss_weight', type=float, default=1.0)
    parser.add_argument('--crgc_r2e_loss_weight', type=float, default=1.0)

    # RCPE: regional temporal counterfactual-priority exploration.
    # Disabled by default, so existing experiments keep exactly the old reward.
    parser.add_argument('--use_rcpe', '--rcpe', dest='use_rcpe', action='store_true', default=True)
    parser.add_argument('--rcpe_lambda', type=float, default=0.1,
                        help='intrinsic team-reward coefficient; the bonus is divided by UAV_NUM')
    parser.add_argument('--rcpe_horizon', type=int, default=3,
                        help='number of steps in the extrinsic team-return target')
    parser.add_argument('--rcpe_gamma', type=float, default=0.99)
    parser.add_argument('--rcpe_b', type=int, default=1,
                        help='number of recent anchor states per priority update')
    parser.add_argument('--rcpe_l', type=int, default=8,
                        help='number of shared counterfactual joint actions')
    parser.add_argument('--rcpe_lags', type=str, default='1,3,5,7,9',
                        help='comma-separated historical lags; its length is M')
    parser.add_argument('--rcpe_update_interval', type=int, default=10,
                        help='vector-environment time slots between priority updates')
    parser.add_argument('--rcpe_reward_min', type=float, default=-3.0)
    parser.add_argument('--rcpe_reward_max', type=float, default=3.0)
    parser.add_argument('--rcpe_reward_bins', type=int, default=21)
    parser.add_argument('--rcpe_divergence', choices=['kl', 'js'], default='kl')
    parser.add_argument('--rcpe_warmup', type=int, default=10000,
                        help='minimum number of completed H-step targets before priorities are learned')
    parser.add_argument('--rcpe_replay_size', type=int, default=50000)
    parser.add_argument('--rcpe_history_size', type=int, default=240,
                        help='recent per-thread states retained for temporal interventions')
    parser.add_argument('--rcpe_batch_size', type=int, default=256)
    parser.add_argument('--rcpe_train_every', type=int, default=10,
                        help='vector-environment time slots between return-model updates')
    parser.add_argument('--rcpe_train_steps', type=int, default=1)
    parser.add_argument('--rcpe_predict_batch_size', type=int, default=4096)
    parser.add_argument('--rcpe_hidden_dim', type=int, default=256)
    parser.add_argument('--rcpe_lr', type=float, default=1e-3)
    parser.add_argument('--rcpe_eps', type=float, default=1e-8)
    parser.add_argument('--rcpe_priority_ema', type=float, default=0.0,
                        help='0 uses Eq.(19-20) directly; values such as 0.9 smooth priorities')
    parser.add_argument('--rcpe_priority_floor', type=float, default=0.0,
                        help='uniform mass mixed into learned regional priorities')
    parser.add_argument('--rcpe_count_bins', type=str, default='1,2,3,5,8')
    parser.add_argument('--rcpe_aoi_bins', type=str, default='0.2,0.4,0.6,0.8,1.0')
    parser.add_argument('--rcpe_aoi_scale', type=float, default=60.0,
                        help='mean/max regional AoI normalization divisor')
    parser.add_argument('--rcpe_tau_bins', type=str, default='1,2,4,8,16,32,64')
    parser.add_argument('--rcpe_seed', type=int, default=1)
    input_args = parser.parse_args()


    if input_args.high_level_dont_use_snrmap:
        input_args.use_snrmap = False
    if input_args.high_level_knn_coefficient != -1:
        input_args.knn_coefficient = input_args.high_level_knn_coefficient

    if input_args.use_rcpe:
        if input_args.high_level_knn_coefficient > -1:
            parser.error('RCPE and the legacy KNN intrinsic reward cannot be enabled together')
        if input_args.rcpe_horizon < 1 or input_args.rcpe_b < 1 or input_args.rcpe_l < 1:
            parser.error('rcpe_horizon, rcpe_b and rcpe_l must be positive')
        if input_args.rcpe_update_interval < 1 or input_args.rcpe_train_every < 1:
            parser.error('RCPE update/train intervals must be positive')
        if input_args.rcpe_aoi_scale <= 0:
            parser.error('rcpe_aoi_scale must be positive')
        if not 0.0 <= input_args.rcpe_priority_ema < 1.0:
            parser.error('rcpe_priority_ema must be in [0, 1)')
        if not 0.0 <= input_args.rcpe_priority_floor < 1.0:
            parser.error('rcpe_priority_floor must be in [0, 1)')

    if input_args.use_crgc:
        if input_args.algo not in ('IPPO', 'DPPO', 'CPPO'):
            parser.error('CRGC currently supports IPPO, DPPO and CPPO actors')
        if input_args.use_stack_frame:
            parser.error('CRGC supplies its own L+1 history; do not combine it with use_stack_frame')
        if input_args.crgc_history_len < 0 or input_args.map_size < 1:
            parser.error('crgc_history_len must be nonnegative and map_size must be positive')
        if input_args.crgc_feature_dim % input_args.crgc_attention_heads != 0:
            parser.error('crgc_feature_dim must be divisible by crgc_attention_heads')
        if input_args.crgc_message_dim < 1:
            parser.error('crgc_message_dim must be positive')
        if input_args.crgc_continuous_dim < 1:
            parser.error('crgc_continuous_dim must be positive')
        if input_args.crgc_risk_margin < 0 or input_args.crgc_distance_lambda < 0:
            parser.error('CRGC risk margin and distance lambda must be nonnegative')
        if input_args.crgc_alignment_temperature <= 0 or input_args.crgc_physical_temperature <= 0:
            parser.error('CRGC temperatures must be positive')
        if input_args.crgc_observed_aoi_scale <= 0:
            parser.error('crgc_observed_aoi_scale must be positive')
        if input_args.use_mse_weight and not (0 < input_args.mse_risk_alpha <= input_args.mse_violation_alpha):
            parser.error('weighted MSE requires 0 < mse_risk_alpha <= mse_violation_alpha')

    if input_args.debug:
        input_args.group = 'debug'
        input_args.n_thread = 2
    input_args.output_dir = f'runs/{input_args.group}'

    if input_args.test:
        input_args.group = 'test'
        input_args.n_thread = 1
        input_args.output_dir = f'{input_args.checkpoint}/test'


    if input_args.dataset == 'roma':
        metadata_path = Path(__file__).resolve().parent / 'envs' / 'roma' / 'metadata_user80_2350.json'
        dataset_users = json.loads(metadata_path.read_text(encoding='utf-8'))['user_count']
        if input_args.poi_num not in (48, dataset_users):
            parser.error(f'--dataset roma requires --poi_num {dataset_users}')
        input_args.poi_num = dataset_users

    env_args = {
        "max_episode_step": input_args.max_episode_step,
        "collect_range": input_args.collect_range,
        "initial_energy": input_args.init_energy,
        "user_data_amount": input_args.user_data_amount,
        "update_num": input_args.update_num,
        "uav_num": input_args.uav_num,
        "AoI_THRESHOLD": input_args.aoith,
        "RATE_THRESHOLD": input_args.txth,
        "uav_height": input_args.uav_height,
        "aoi_vio_penalty_scale": input_args.aVPS,
        "hao02191630": input_args.hao02191630,
        "w_noise": input_args.w_noise,
        "agent_field": input_args.agent_field,
    }
    if input_args.poi_num is not None:
        env_args["poi_num"] = input_args.poi_num

    return input_args, env_args
