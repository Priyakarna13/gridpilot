# GridPilot

**Stochastic grid navigation with Deep Q-Networks.**

GridPilot trains a DQN agent to cross a classic 3×4 grid world — reaching the goal, avoiding traps, and learning a robust policy under noisy transitions. Built with PyTorch.

## Features

- Custom stochastic GridWorld environment with a simple `reset` / `step` API
- DQN with experience replay, target network, and epsilon-greedy exploration
- Live training visualization with Q-values and policy arrows
- Hyperparameter sweeps for learning rate, gamma, epsilon decay, and target sync

## Environment

The agent starts bottom-left and must reach the goal without falling into the trap. Actions are stochastic: 80% intended direction, 10% slip left, 10% slip right.

```
(0,0) (0,1) (0,2) (0,3)  GOAL (+1)
(1,0) (1,1)  BLOCK  (1,3)  TRAP (-1)
(2,0) START (2,1) (2,2) (2,3)
```

| Setting | Value |
|---------|-------|
| Living cost | -0.04 per step |
| Actions | N, E, S, W (0–3) |

## Project layout

| File | Description |
|------|-------------|
| `grid_world.py` | Stochastic grid environment |
| `dqn_agent.py` | Q-network, replay buffer, and DQN agent |
| `train.py` | Training loop with live visualization |
| `run_experiments.py` | Hyperparameter comparison runs |
| `plots/` | Sample training and experiment outputs |

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/gridpilot.git
cd gridpilot

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python train.py
```

CUDA is used automatically when available; otherwise training runs on CPU.

## Training

`train.py` runs 6,000 episodes and produces:

- A live grid view of Q-values and greedy policy arrows (updates every 50 episodes)
- `dqn_learning_curve.png` — episode reward and epsilon decay
- A final policy printed to the terminal

```bash
python train.py
```

> Requires a display for matplotlib's interactive windows. Use a non-interactive backend for headless runs.

## Experiments

`run_experiments.py` benchmarks hyperparameter choices across full training runs:

```bash
python run_experiments.py
```

Generates comparison plots for learning rate, discount factor, epsilon decay, and target network update frequency.

## Default hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 6,000 |
| Learning rate | 0.001 |
| Gamma | 0.95 |
| Batch size | 64 |
| Target update | Every 10 episodes |
| Epsilon | 0.9 → 0.05 (decay over 30k steps) |

## License

See [LICENSE](LICENSE).
