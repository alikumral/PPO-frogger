# PPO Frogger

Frogger-style reinforcement learning project with:
- `v1_classic`: original beginner-friendly profile
- `v2_arcade`: arcade-chaos profile with lane archetypes, event director, dash action, combo scoring, and multi-goal runs

## Stack
- Python 3.10+
- gymnasium
- stable-baselines3 (PPO)
- pygame (human render)

## Project Layout
- `frogger_lite/env.py`: environment with profile-based mechanics
- `train.py`: PPO training entrypoint
- `eval.py`: deterministic evaluation + metrics export
- `tests/`: unit, deterministic replay, and smoke integration tests

## V2 Arcade Mechanics
- Lane archetypes: `normal`, `burst`, `stop_go`, `conveyor`, `hazard`
- Event director:
  - `EMERGENCY_SWEEP`
  - `BLACKOUT`
  - `ROADBLOCK_SHIFT`
  - `TRAFFIC_SURGE`
  - no same event twice in a row
- Action space: `Discrete(6)` with `DASH_UP` cooldown 10 steps
- Scoring:
  - `+1.0` per forward row
  - `-0.02` for idle (`stay`)
  - `+0.4` near miss
  - `+3.0` goal
  - `-1.2` collision
  - combo multiplier from `1.0` to `2.0`
- Run structure:
  - up to 3 goals per episode
  - on non-final goal: reset to start, lane speed +8%, event interval tightened

## Observation Space
- `v1_classic`: 16-dim features
- `v2_arcade`: 26-dim features
  - keeps v1 core features
  - adds event one-hot/time, dash cooldown, lane type embeddings, blocked-cell indicator

## Setup
```powershell
cd "D:\VSCode Projects\PPO-frogger"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

### Classic profile
```powershell
py -3 train.py --env-profile v1_classic --total-timesteps 500000 --n-envs 8 --seed 42
```

### Arcade profile (recommended retrain target)
```powershell
py -3 train.py --env-profile v2_arcade --total-timesteps 500000 --n-envs 8 --seed 42
```

V2 uses default `ent_coef=0.015` (v1 default stays `0.01`).  
Override anytime with `--ent-coef`.

### Live preview during training
```powershell
py -3 train.py --env-profile v2_arcade --total-timesteps 500000 --n-envs 8 --preview --preview-mode human --preview-freq 50000 --preview-render-fps 6 --preview-sleep 0.02
```

V2 human preview now includes:
- card-based HUD (`Run State`, `Action & Progress`, `Events`)
- lane legend with color mapping
- event banner + phase visibility
- pulse animation on agent marker
- live reward trace chart (current episode)

## Evaluate

### Basic
```powershell
py -3 eval.py --env-profile v2_arcade --model models/best/best_model.zip --episodes 20
```

### Human render
```powershell
py -3 eval.py --env-profile v2_arcade --model models/best/best_model.zip --episodes 5 --render-mode human --render-fps 6 --sleep 0.03
```

### Export metrics JSON
```powershell
py -3 eval.py --env-profile v2_arcade --model models/best/best_model.zip --episodes 20 --metrics-out runs/eval/v2_metrics.json
```

Exported metrics include:
- `near_miss_rate`
- `combo_avg`
- `goals_per_episode`
- `event_survival_rate`
- `dash_usage_rate`

## Tests
```powershell
py -3 -m pytest -q
```

Includes:
- lane archetype behavior checks
- event non-repetition and roadblock sanity checks
- dash cooldown and combo rules
- fixed-seed deterministic replay check
- PPO smoke-train/save integration check
