---
title: Radiotherapy Dose Planning
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - radiotherapy
  - gymnasium
---
# RadiotherapyPlanningEnv — OpenEnv RL Environment

> **An OpenEnv-compatible RL environment for cancer radiotherapy treatment planning.**

An RL environment where an AI agent learns to plan cancer radiotherapy treatment. The agent places radiation beams to maximize tumor dose while protecting surrounding organs-at-risk (OARs) — a real clinical problem that takes human experts **2-4 hours per patient**.

Built for the **Meta x Scaler PyTorch OpenEnv Hackathon**.

- **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/VaishnaviAgrawal/Radiotherapy-Treatment-Planning-environment)
- **Repository**: [GitHub](https://github.com/VaishnaviAgrawal03/Radiotherapy-Treatment-Planning-environment)
  
The demo lets you:
- **Watch the agent** plan a treatment automatically
- **Play yourself** — step through actions manually
- See the live dose heatmap + DVH curves
- Compare your score to the baseline agent
  
---

## Clinical Motivation

~14 million cancer patients per year require radiotherapy. A radiation oncologist must decide:
- How many beams to use
- At what angles
- With what dose intensity

...while ensuring the tumor receives enough radiation and nearby healthy organs stay below safe limits. This environment simulates that decision-making process for RL agents.

---

## Environment Details

### Three Difficulty Levels

| Task | Env ID | Difficulty | OARs | Max Steps | Pass Score | Clinical Context |
|------|--------|------------|------|-----------|------------|-----------------|
| Prostate | `RadiotherapyEnv-prostate-v1` | Easy | 2 | 50 | ≥ 0.60 | Clear geometry, well-separated organs |
| Head & Neck | `RadiotherapyEnv-headneck-v1` | Medium | 7 | 60 | ≥ 0.55 | Complex anatomy, many competing constraints |
| Pediatric Brain | `RadiotherapyEnv-pediatricbrain-v1` | Hard | 5 | 70 | ≥ 0.50 | Near-zero margin for error, catastrophic brainstem penalty |

### Action Space: `Discrete(8)`

| Action | Effect |
|--------|--------|
| `0` | Add beam at next default angle |
| `1` | Rotate last beam +10° |
| `2` | Rotate last beam -10° |
| `3` | Increase last beam dose weight |
| `4` | Decrease last beam dose weight |
| `5` | Remove last beam |
| `6` | Fine-tune all beams (small perturbation) |
| `7` | Lock plan — terminates episode |

### Observation Space: `Dict`

| Key | Shape | Description |
|-----|-------|-------------|
| `dvh_tumor` | `Box(50,)` | Cumulative Dose-Volume Histogram for tumor |
| `dvh_oar` | `Box(3, 50)` | DVH for top 3 organs-at-risk |
| `beams` | `Box(7, 3)` | Per-beam: `[angle/180, dose_weight, is_active]` |
| `constraints` | `Box(4,)` | Normalized constraint violations `[tumor, oar1, oar2, oar3]` |
| `step_frac` | `Box(1,)` | Episode progress fraction `[0, 1]` |

All observations normalized to `[0, 1]` for neural network training stability.

### Reward Function

```
reward = tumor_coverage × 0.50
       − oar_penalty    × 0.40
       + plan_efficiency × 0.10
```

- **Dense signal** — reward at every step, not just at episode end
- **Partial credit** — each improvement earns positive feedback
- **Range**: `[0.0, 1.0]`

---

## Quick Start

```bash
pip install radiotherapy-env

import gymnasium as gym
import radiotherapy_env

# Easy task
env = gym.make("RadiotherapyEnv-prostate-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final score: {info['score']:.3f}")
env.close()
```

---

### state() API (OpenEnv spec)

```python
state = env.state()
# Returns:
# {
#   "task":        "prostate",
#   "patient":     {"case_id": ..., "tumor_voxels": ..., "oars": [...]},
#   "beams":       [{"angle": 45.0, "dose_weight": 0.8}, ...],
#   "dose_grid":   [[...], ...],   # 64×64 grid
#   "step_count":  12,
#   "last_reward": 0.623,
#   "score":       0.681,
# }
```

---

## Training a Baseline Agent

```bash
# Install with training extras
pip install "radiotherapy-env[training]"

# Train PPO on prostate (easy) — ~15 minutes on CPU
python baseline/train_ppo.py --task prostate --timesteps 200000

# Train all three tasks
python baseline/train_ppo.py --all

# Evaluate
python baseline/evaluate.py
```

### Baseline Results (PPO, stable-baselines3)

| Task | Mean Score | Std | Pass Rate | Training Steps |
|------|-----------|-----|-----------|---------------|
| Prostate | 0.697 | 0.054 | 100% | 200K |
| Head & Neck | 0.750 | 0.059 | 96.7% | 350K |
| Pediatric Brain | 0.717 | 0.090 | 95.0% | 1M |
| **Aggregate** | **0.721** | — | — | — |

## Run Tests

```bash
pip install "radiotherapy-env[dev]"
pytest tests/ -v
```

All tests include:
- ✅ `gymnasium.utils.env_checker` compliance
- ✅ Seed reproducibility
- ✅ Physics correctness (dose calculation)
- ✅ Reward range `[0, 1]`
- ✅ Task difficulty verification

---

## Docker

```bash
# Build
docker build -t radiotherapy-env:latest .

# Run Gradio demo
docker run -p 7860:7860 radiotherapy-env:latest

# Run tests
docker run radiotherapy-env:latest pytest tests/ -v

# Train
docker run radiotherapy-env:latest \
    python baseline/train_ppo.py --task prostate
```

## Project Structure

```
radiotherapy-env/
├── radiotherapy_env/
│   ├── env.py                  ← Main RadiotherapyEnv class
│   ├── physics/
│   │   ├── phantom.py          ← Patient models (3 task generators)
│   │   ├── dose_calculator.py  ← Pencil-beam dose model
│   │   └── dvh.py              ← DVH computation
│   ├── tasks/
│   │   ├── prostate.py         ← Task 1: Easy
│   │   ├── head_neck.py        ← Task 2: Medium
│   │   └── pediatric_brain.py  ← Task 3: Hard
│   ├── reward/
│   │   ├── reward_fn.py        ← Dense reward + final score
│   │   └── grader.py           ← Auto-grader (0.0 → 1.0)
│   └── rendering/
│       └── dose_heatmap.py     ← Dose heatmap + DVH renderer
├── baseline/
│   ├── train_ppo.py            ← PPO training script
│   └── evaluate.py             ← Reproducible evaluation
├── tests/
│   └── test_env.py             ← Full test suite
├── app/
│   └── app.py                  ← Gradio UI components
├── server/
│   └── app.py                  ← Server logic
├── server.py                   ← Docker entry point (Gradio demo)
├── app.py                      ← Standalone demo
├── inference.py                ← Inference script
├── openenv.yaml                ← OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Clinical Significance

| Metric | Current (Manual) | With RL Agent |
|--------|-----------------|---------------|
| Planning time | 2–4 hours | < 5 minutes |
| Patients/year | 14 million | 14 million |
| Time saved/year | — | ~28 million hours |
| Accessibility | Specialist required | Any hospital |

---

## Author

**Vaishnavi Agrawal** — vagrawal_be22@thapar.edu
