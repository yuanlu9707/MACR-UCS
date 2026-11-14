# MACR-UCS

MACR-UCS for **"Cross-Scale Reconstruction-Guided Communication and Regional Causal Priority-Guided Exploration for UAV Crowdsourcing"**.

## 📄 Description

MACR-UCS is a multi-agent reinforcement learning framework for cooperative UAV data collection under partial observability. UAVs coordinate their movement and service decisions to improve information freshness and collection efficiency while accounting for energy consumption.

The framework combines two modules with an IPPO actor-critic backbone:

- **CRGC — Cross-Scale Reconstruction-Guided Communication:** fuses UAV observations into microscopic entity inputs and macroscopic regional inputs, processes observation histories with GRUs and spatial attention, and exchanges information through a cross-scale Transformer. Reconstruction, physical consistency, and bidirectional alignment objectives support representation learning. Reconstructed information generates personalized, distance-aware messages for each UAV.
- **RCPE — Regional Causal Priority-Guided Exploration:** learns a distribution over short-horizon extrinsic team returns, evaluates regional temporal counterfactual interventions, and combines regional priorities with count-based novelty to produce a training-time intrinsic reward. Reported episode rewards exclude this intrinsic bonus.

Two datasets are included: **NCSU** (48 users) and **roma** (80 trajectories from the Roma taxi dataset).

The reported **CSE** metric is

\[
\mathrm{CSE}=\frac{S_{\mathrm{AoI}}\,S_{\mathrm{collection}}\,S_{\mathrm{data}}}{E_{\mathrm{total}}/(U\times10^6)},
\]

where the numerator contains AoI satisfaction, collection ratio, and the code's AoI-derived data satisfaction ratio. The denominator is average energy consumption per UAV in MJ.

```text
MACR-UCS/
├── README.md
├── requirements.txt
├── source_code/
│   ├── main_DPPO.py              # Training and testing entry point
│   ├── get_args.py               # Environment, CRGC, and RCPE parameters
│   ├── algorithms/
│   │   ├── crgc.py               # Communication and reconstruction
│   │   ├── rcpe.py               # Regional exploration
│   │   ├── algo/                 # PPO agents, runner, and buffers
│   │   └── config/Mobile_IPPO.py  # PPO optimization settings
│   ├── envs/                     # Environment and NCSU/Roma data
│   ├── env_configs/              # Maps and parallel environment wrappers
│   ├── tools/post/vis.py         # Automatically invoked trajectory renderer
│   └── vis.py                    # Standalone trajectory visualization
└── runs/                         # Generated experiment outputs
```

## 🔧 Dependencies

The dependency file provides a **suggested compatibility baseline** for Python 3.9 and PyTorch 1.13.1, based on the current imports and APIs. It is not a lockfile exported from a validated experimental environment. A fresh installation and end-to-end training run still need to be verified on your machine.

```bash
git clone https://github.com/yuanlu9707/MACR-UCS.git
cd MACR-UCS
conda create -n macr-ucs python=3.8 -y
conda activate macr-ucs
python -m pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch build first. For example, the CUDA 11.7 build for Linux/Windows is:

```bash
python -m pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1131) for other builds. GPU execution requires a compatible NVIDIA driver. Use `--device cpu` for CPU execution. The geospatial dependencies are needed for trajectory rendering, which is also called when training saves trajectories. `wandb` is imported by the code even when online logging is muted.

## 💻 Training

Run commands from **`source_code`**; the code uses this working directory for imports and visualization.

```bash
cd source_code
python main_DPPO.py --dataset NCSU --device cuda:0 --group NCSU
```

Train on Roma:

```bash
python main_DPPO.py --dataset roma --device cuda:0 --group roma
```



CRGC and RCPE are both enabled by default in the current configuration. Important settings in [`source_code/get_args.py`](source_code/get_args.py) include:

| Parameter | Default | Meaning |
| --- |-------| --- |
| `dataset` | `roma` | `NCSU` or `roma` |
| `poi_num` | `80`  | Parser default; Roma automatically selects 80 users |
| `uav_num` | `5`   | Number of UAVs |
| `n_iter` | `5000` | Training iterations |
| `max_episode_step` | `120` | Episode length |
| `agent_field` | `500` | Observation distance |




Outputs are written to **`runs/<group>/<timestamp-and-settings>/`** at the repository root. Depending on training progress, an experiment directory contains:

- `params.json`: environment and command-line settings.
- `Models/best_actor.pt`: best training-reward actor, including its CRGC parameters.
- `Models/RCPE_*.pt`: periodically saved exploration-module states.
- `episode_reward_threads.csv`, `episode_CSE_threads.csv`: per-thread episode metrics.
- `train_output.txt`, `test_output.txt`, trajectory archives, HTML visualizations, and TensorBoard events.

## 🏁 Testing

Pass the **experiment directory containing `Models/`**, rather than a `.pt` file, to `--checkpoint`:

```bash
# From source_code; replace the path with your actual experiment directory.
python main_DPPO.py --test --dataset NCSU --device cuda:0 --checkpoint ../runs/NCSU/EXPERIMENT_DIRECTORY
```

Use the same dataset, UAV count, CRGC dimensions, history length, and other architecture settings as training. Consult the saved `params.json`; the testing command does not automatically restore all settings from it. Testing loads `Models/best_actor.pt` and selects each action branch by its highest probability. Results are written under `<checkpoint>/test/`.

RCPE is used only during training; actor evaluation does not require an RCPE checkpoint. The checkpoint entry point loads actor weights, so it should not be treated as an exact resumption of the optimizer, critic, replay buffer, and exploration state.

To render saved trajectories manually:

```bash
python vis.py --output_dir ../runs/NCSU/EXPERIMENT_DIRECTORY --tag train
python vis.py --output_dir ../runs/NCSU/EXPERIMENT_DIRECTORY/test --tag test
```

The default visualization selects saved best trajectories. Use `--iter <NUMBER>` for a periodically saved trajectory. Metrics and reconstruction accuracy correspond to the available observations; reconstruction MSE is not an evaluation against an independently supplied complete global-state target.

## 👏 Reference

Related implementations and documentation consulted for this repository:

- MetaCS: https://github.com/yuxiaooye/DRL_dyna_AoI.

## 📜 Acknowledgement

This work is supported by the National Key Research and Development Program of China (2021YFF0901205).

Corresponding author: Lei Yang.

## 📧 Contact

For questions, please contact yuanlu9707@163.com.
Thank you.

## Citation

"Cross-Scale Reconstruction-Guided Communication and Regional Causal Priority-Guided Exploration for UAV Crowdsourcing"
