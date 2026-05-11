# TDCR Reinforcement Learning Formulation Benchmark

This repository contains the code used for the paper **"A Systematic Benchmark of Reinforcement Learning Formulations for Tendon-Driven Continuum Robot Control"**.

The benchmark compares RL formulations for tendon-driven continuum robot tip control. All policies use the same fixed **data-driven kinematic surrogate**, trained from motion-capture data, so the comparison focuses on the RL formulation: state definition, reward design, and action interface.

The workflow in the code follows the paper:

1. Phase I screening: train each algorithm over the state-reward grid and detect converged runs.
2. Checkpoint selection: choose the best checkpoint for each retained configuration using a kinematic validation set.
3. Phase II evaluation: test retained policies on held-out point reaching and waypoint tracking.
4. Deterministic evaluation and stochastic robustness: run Phase II without and with Gaussian policy perturbations.

## Environment

The RL environment is `ContinuumRobotEnv` in [`ContinuumRobot.py`](ContinuumRobot.py). It loads the pretrained surrogate from:

```text
SurrogateModel/trained_model_sl_1.pth
SurrogateModel/scaler_X_sl_1.pkl
SurrogateModel/scaler_y_sl_1.pkl
```

The surrogate maps encoded tendon commands `(l1, l2, l3)` to the Cartesian tip position `(x, y, z)`. Dataset coordinates are stored in millimeters; the environment uses meters internally.

Key settings from the code:

| Setting | Value |
| --- | --- |
| Tendon range | `u in [0, 100]^3` |
| Continuous action | incremental `delta_u in [-1, 1]^3` |
| Success tolerance | `1e-3 m` |
| Episode limit | `1000` steps |
| Training seed | `666` |

The environment is quasi-static. It does not model tendon history, friction, hysteresis, sensing latency, or hardware execution.

## Algorithms

The repository implements:

| Algorithm | Action interface | Training | Phase I | Checkpoint selection | Phase II |
| --- | --- | --- | --- | --- | --- |
| DQN | fixed 22-action discrete catalogue | yes | yes | no | no |
| DDPG | continuous tendon increments | yes | yes | yes | yes |
| TD3 | continuous tendon increments | yes | yes | yes | yes |
| SAC | continuous tendon increments | yes | yes | yes | yes |
| PPO | continuous tendon increments | yes | yes | yes | yes |

DQN uses [`DQN_learning/DiscreteAction.py`](DQN_learning/DiscreteAction.py) to wrap the same base environment with 22 discrete tendon-increment actions.

## State and Reward Grid

Each `start.py` trains 24 configurations:

```text
2 state families x 2 actuator-state options x 3 reward families x 2 reward-augmentation options
```

State families:

- `state_type=1`: error-based state, using normalized target-tip Cartesian error.
- `state_type=2`: pose-target state, using normalized current tip position and target position.
- `include_u_abs=True`: appends normalized absolute tendon state to either state family.

Reward families:

- `reward_type=1`: dense distance penalty.
- `reward_type=2`: progress reward based on whether the distance decreases.
- `reward_type=3`: shaped radial reward.

The optional composite reward adds a step penalty and terminal success bonus:

| Reward type | Step penalty | Success bonus |
| --- | ---: | ---: |
| 1 | 0.05 | 5.0 |
| 2 | 0.6 | 60.0 |
| 3 | 1.1 | 100.0 |

Log and checkpoint names encode the configuration:

```text
<algorithm>_continuum_robot_s<state_type>u<include_u_abs>r<reward_type>b<bonus_flag>_e<episode>
```

Example: `td3_continuum_robot_s1u0r1b0_e500`.

## Tasks

Phase I screening uses training logs. A run is marked converged after 100 consecutive rows with `avg_distance <= 0.001` m. For PPO, rows are policy updates; for DQN, DDPG, TD3, and SAC, rows are episodes.

Phase II has two task families:

- **Held-out point reaching:** zero-tendon start to targets in `dataset/phase2_region1.txt`, `dataset/phase2_region2.txt`, and `dataset/phase2_region3.txt`.
- **Sequential waypoint-following:** circular waypoint tracking and spiral waypoint tracking with state carried forward between waypoints.

The circular path uses 50 waypoints and closes the loop by revisiting the first point. The spiral path uses 150 waypoints along the empirical workspace envelope.

Phase II logs include mean error, maximum error, success rate, mean steps, waypoint error, path RMSE, maximum tracking error, and trajectory/reference fields for waypoint runs.

## Repository Layout

```text
.
|-- ContinuumRobot.py
|-- phase1_convergence_common.py
|-- phase1_checkpoint_common.py
|-- phase2_common.py
|-- dataset/
|   |-- Dataset-Actions-Positions.txt
|   |-- RL_training.txt
|   |-- phase1_region1.txt ... phase1_region3.txt
|   `-- phase2_region1.txt ... phase2_region3.txt
|-- SurrogateModel/
|   |-- trained_model_sl_1.pth
|   |-- scaler_X_sl_1.pkl
|   `-- scaler_y_sl_1.pkl
|-- SL_learning/
|-- DQN_learning/
|-- DDPG_learning/
|-- TD3_learning/
|-- SAC_learning/
|-- PPO_learning/
|-- DQNLearnedModel/
|-- DDPGLearnedModel/
|-- TD3LearnedModel/
|-- SACLearnedModel/
|-- PPOLearnedModel/
|-- requirements.txt
`-- LICENSE
```

The `*LearnedModel/` folders are checkpoint output folders. They are empty in a clean checkout and are populated by the training scripts.

## Installation

The included virtual environment was created with Python 3.10.12. For a fresh setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The RL scripts set `USE_WANDB = False`. The surrogate training script imports and initializes Weights & Biases.

## Data and Surrogate

The main data files are:

| File | Use |
| --- | --- |
| `dataset/Dataset-Actions-Positions.txt` | supervised surrogate training data |
| `dataset/RL_training.txt` | target and reset samples used by the RL environment |
| `dataset/phase1_region*.txt` | validation targets for checkpoint selection |
| `dataset/phase2_region*.txt` | held-out point-reaching targets |

Check that the surrogate artifacts are present before running RL training or evaluation:

```bash
ls SurrogateModel/trained_model_sl_1.pth SurrogateModel/scaler_X_sl_1.pkl SurrogateModel/scaler_y_sl_1.pkl dataset/RL_training.txt
```

To retrain the surrogate:

```bash
python3 SL_learning/SurrogateLearning.py
```

This writes the same filenames used by `ContinuumRobotEnv`.

## Training

Each training script runs all 24 state-reward configurations for one algorithm. DQN, DDPG, TD3, and SAC train for 1500 episodes. PPO trains for 1500 policy updates with 1024 rollout steps per update.

Run from the repository root:

```bash
cd DQN_learning && python3 start.py
cd ../DDPG_learning && python3 start.py
cd ../TD3_learning && python3 start.py
cd ../SAC_learning && python3 start.py
cd ../PPO_learning && python3 start.py
cd ..
```

Training writes per-configuration CSV logs in each algorithm folder and checkpoints in the corresponding `*LearnedModel/` folder. Checkpoints are saved every 100 episodes or updates and once more at the final horizon.

## Phase I Screening

Run the convergence detector after training:

```bash
cd DQN_learning && python3 Phase1ConvDQN.py
cd ../DDPG_learning && python3 Phase1ConvDDPG.py
cd ../TD3_learning && python3 Phase1ConvTD3.py
cd ../SAC_learning && python3 Phase1ConvSAC.py
cd ../PPO_learning && python3 Phase1ConvPPO.py
cd ..
```

Each script writes `<algorithm>_convergence.csv` in its algorithm folder. Each row contains the training-log filename, a yes/no convergence flag, and the convergence episode or update.

The shared detector also supports:

```bash
python3 TD3_learning/Phase1ConvTD3.py --root-dir TD3_learning --output-file td3_convergence.csv
```

## Checkpoint Selection

Checkpoint selection evaluates converged configurations on `dataset/phase1_region1.txt` through `dataset/phase1_region3.txt`. Rollouts start from zero tendon actuation, and checkpoints are ranked by validation MAE in millimeters.

Supported adapters:

```bash
cd DDPG_learning && python3 Phase1CheckpointDDPG.py
cd ../TD3_learning && python3 Phase1CheckpointTD3.py
cd ../SAC_learning && python3 Phase1CheckpointSAC.py
cd ../PPO_learning && python3 Phase1CheckpointPPO.py
cd ..
```

Each adapter writes `best_results.csv` in its algorithm folder.

Example with explicit options:

```bash
python3 TD3_learning/Phase1CheckpointTD3.py --model-dir TD3LearnedModel --dataset-dir dataset --step-cap 500
```

## Phase II Evaluation

The Phase II scripts read the convergence CSV and `best_results.csv`, load the selected checkpoints, and write a single CSV log per algorithm.

Deterministic evaluation:

```bash
cd DDPG_learning && python3 Phase2DDPG.py --mode all --eval-mode deterministic --csv-log-path phase2_ddpg_log.csv
cd ../TD3_learning && python3 Phase2TD3.py --mode all --eval-mode deterministic --csv-log-path phase2_td3_log.csv
cd ../SAC_learning && python3 Phase2SAC.py --mode all --eval-mode deterministic --csv-log-path phase2_sac_log.csv
cd ../PPO_learning && python3 Phase2PPO.py --mode all --eval-mode deterministic --csv-log-path phase2_ppo_log.csv
cd ..
```

Stochastic robustness evaluation:

```bash
cd DDPG_learning && python3 Phase2DDPG.py --mode all --eval-mode stochastic --noise-sigma 0.01 --csv-log-path phase2_ddpg_log.csv
cd ../TD3_learning && python3 Phase2TD3.py --mode all --eval-mode stochastic --noise-sigma 0.01 --csv-log-path phase2_td3_log.csv
cd ../SAC_learning && python3 Phase2SAC.py --mode all --eval-mode stochastic --noise-sigma 0.01 --csv-log-path phase2_sac_log.csv
cd ../PPO_learning && python3 Phase2PPO.py --mode all --eval-mode stochastic --noise-sigma 0.01 --csv-log-path phase2_ppo_log.csv
cd ..
```

Use `--mode batch` for held-out point reaching only, `--mode trajectory` for circular and spiral waypoint tracking only, or `--mode all` for both. In stochastic mode, the default noise standard deviation is `0.01`; batch evaluation uses the first 5 seeds, and trajectory evaluation uses the first 10 seeds from the default seed list in [`phase2_common.py`](phase2_common.py).

## Outputs

| Output | Location |
| --- | --- |
| Training logs | `<algorithm>_learning/*.csv` |
| RL checkpoints | `<Algorithm>LearnedModel/` |
| Phase I convergence tables | `<algorithm>_learning/<algorithm>_convergence.csv` |
| Selected checkpoint tables | `<algorithm>_learning/best_results.csv` |
| Phase II result tables | `<algorithm>_learning/phase2_<algorithm>_log.csv` |

## Extending the Code

When adding a new algorithm, follow the existing pattern: an algorithm folder with `start.py`, model code, a Phase I convergence wrapper, a checkpoint-selection adapter, and a Phase II adapter. Keep the checkpoint naming convention if you want to use the shared selection and evaluation code.

For formulation studies, keep the surrogate fixed unless the surrogate itself is the experimental variable.

## Citation

```bibtex
@ARTICLE{11510251,
  author={Rusinak, Darius and Kelemen, Michal and Virgala, Ivan},
  journal={IEEE Access}, 
  title={A Systematic Benchmark of Reinforcement Learning Formulations for Tendon-Driven Continuum Robot Control}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Filtering;Contacts;Circuits and systems;Feedback;Circuits;Filters;Central Processing Unit;Oscillators;Protocols;Communication systems;Continuum robots;reinforcement learning;tendon-driven robots;benchmarking;state-reward design;kinematic control;soft robotics},
  doi={10.1109/ACCESS.2026.3690990}}
```

## License

MIT License. See [`LICENSE`](LICENSE).

## Contact

Corresponding author: Darius Rusinak, `darius.rusinak@tuke.sk`.
