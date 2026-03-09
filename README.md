# Frogger Lite PPO Starter

This project is a minimal Python starter for training a Frogger-lite agent with PPO.

## Stack

- Python 3.10+
- `gymnasium`
- `stable-baselines3`
- `torch`

## Project Layout

- `frogger_lite/env.py`: custom `FroggerLiteEnv`
- `train.py`: PPO training entrypoint
- `eval.py`: deterministic evaluation script
- `requirements.txt`: dependencies

## Environment Design

### Actions (`Discrete(5)`)

- `0`: stay
- `1`: up
- `2`: down
- `3`: left
- `4`: right

### Observation (`Box(shape=(16,), low=-1, high=1)`)

Order:

1. `row_norm`
2. `col_norm`
3. `dist_goal_norm`
4. `step_frac`
5. `danger_current_row`
6. `lane_dir_current_row`
7. `lane_speed_current_row`
8. `occupied_current_row`
9. `danger_row_minus_1`
10. `lane_dir_row_minus_1`
11. `lane_speed_row_minus_1`
12. `occupied_row_minus_1`
13. `danger_row_minus_2`
14. `lane_dir_row_minus_2`
15. `lane_speed_row_minus_2`
16. `occupied_row_minus_2`

### Reward Shaping

- `-0.01` per step (time pressure)
- `+0.10` per row of upward progress
- `+0.20` when reaching a new best row in the episode
- `-1.00` on collision (terminal)
- `+2.00` on reaching goal row (terminal)
- `-0.20` on time truncation (`max_steps`)

This is dense enough to learn quickly while still rewarding actual goal completion.

## PPO Baseline Hyperparameters

Defined in `train.py`:

- `learning_rate=3e-4`
- `n_steps=1024`
- `batch_size=256`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.01`
- `vf_coef=0.5`
- `max_grad_norm=0.5`

## Setup

```powershell
cd "d:\VSCode Projects\mypython-project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py --total-timesteps 500000 --n-envs 8 --seed 42
```

Training is headless by default (fastest).

To get a simple live terminal visualization during training:

```powershell
python train.py --total-timesteps 500000 --n-envs 8 --preview --preview-mode ansi --preview-freq 50000 --preview-sleep 0.08
```

To watch previews in a basic window (pygame UI) during training:

```powershell
python train.py --total-timesteps 500000 --n-envs 8 --preview --preview-mode human --preview-freq 50000 --preview-render-fps 6
```

Artifacts:

- Final model: `models/ppo_frogger_lite_final.zip`
- Best checkpoint: `models/best/best_model.zip`
- Eval logs: `runs/eval/`

## Evaluate

```powershell
python eval.py --model models/best/best_model.zip --episodes 20
```

Render in terminal:

```powershell
python eval.py --model models/best/best_model.zip --episodes 5 --render-mode ansi
```

Render in a basic window (pygame):

```powershell
python eval.py --model models/best/best_model.zip --episodes 5 --render-mode human --render-fps 6
```

If you want it even slower, add extra delay per frame:

```powershell
python eval.py --model models/best/best_model.zip --episodes 5 --render-mode human --render-fps 6 --sleep 0.05
```

In human window mode, press `ESC` (or close window) to hide preview without stopping training.

## Next Tuning Steps

1. Increase `--total-timesteps` to `1_000_000+`.
2. If exploration is weak, raise `ent_coef` to `0.02`.
3. If policy is unstable, lower `learning_rate` to `1e-4`.
4. If success plateaus, add one more look-ahead lane to observation.
