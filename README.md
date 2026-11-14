# MACR-UCS

MACR-UCS for **"Cross-Scale Reconstruction-Guided Communication and Regional Causal Priority-Guided Exploration for UAV Crowdsourcing"**.

## 📄 Description

Unmanned aerial vehicles (UAV) crowdsourcing (UCS) has emerged as a promising paradigm for spatio-temporal data collection. However, in time-sensitive data collection involving mobile users, existing methods struggle to obtain comprehensive global spatio-temporal task information under partial observability and coordinate multi-UAV exploration, leading to insufficient collaboration, redundant coverage, and reduced data freshness. Therefore, we propose a novel multi-agent deep reinforcement learning framework, termed MACR-UCS, to improve composite sensing efficiency in UCS tasks. Specifically, we first propose a cross-scale reconstruction-guided communication (CRGC) mechanism that reconstructs global spatio-temporal task states at microscopic entity and macroscopic region scales by integrating incomplete local observations from multiple UAVs. Bidirectional micro-macro representation alignment and cross-scale state consistency constraints enhance cross-scale semantic relationships and reconstruction consistency. CRGC further generates personalized messages to improve multi-UAV collaborative decision-making. Second, we propose a regional causal priority-guided exploration (RCPE) mechanism that estimates regional state novelty via semantic quantization and Hash-Count and determines regional causal exploration priorities through counterfactual interventions and future return distribution modeling. RCPE combines both to construct an intrinsic reward, guiding task-value-aware collaborative exploration to reduce redundant coverage and regional service imbalance. Experimental results on the NCSU and Roma datasets demonstrate that MACR-UCS outperforms seven baseline methods.


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

The dependency file provides a suggested compatibility baseline for Python 3.8 and PyTorch 1.13.1, based on the current imports and APIs.

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
| `max_episode_step` | `120` | Episode length |
| `--aoith n` | `30` | AoI threshold, measured in time slots |
| `--txth n` | `1` | Transmission-rate threshold, measured in Mbps |
| `--update_num n` | `10` | Number of antennas; a UAV can serve at most `n` users in one time slot |




Outputs are written to **`runs/<group>/<timestamp-and-settings>/`** at the repository root. Depending on training progress, an experiment directory contains:

```text
EXPERIMENT_DIRECTORY/
├── Models/            # Best network model
├── params.json        # Parameters used for this experiment
├── train_saved_trajs/ # Best UAV trajectories from training episodes
├── train_output.txt   # Metrics associated with the best training model
├── test_output.txt    # Evaluation metrics
└── vis.html           # HTML visualization of the saved UAV trajectories
```

HTML filenames start with `vis_` in the current implementation. Standalone testing saves its outputs under the experiment's `test/` subdirectory.

## 🏁 Testing

From `source_code`, load the best actor using the experiment directory:

```bash
python main_DPPO.py --test --dataset NCSU --device cuda:0 --checkpoint ../runs/NCSU/EXPERIMENT_DIRECTORY
```

Replace `EXPERIMENT_DIRECTORY` with your saved run folder and use the same dataset and model settings as training. Results are saved to `<checkpoint>/test/`.

## 🗺️ Visualization

Generate an HTML visualization of the saved UAV trajectories from `source_code`:

```bash
python tools/post/vis.py --output_dir ../runs/NCSU/EXPERIMENT_DIRECTORY
```

## 👏 Reference

Thanks for:  MetaCS: https://github.com/yuxiaooye/DRL_dyna_AoI.

## 📜 Acknowledgement

This work is supported by the National Key Research and Development Program of China (2021YFF0901205).

Corresponding author: Lei Yang.

## 📧 Contact

For questions, please contact yuanlu9707@163.com.
Thank you.

## Citation

"Cross-Scale Reconstruction-Guided Communication and Regional Causal Priority-Guided Exploration for UAV Crowdsourcing"
