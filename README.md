---
title: Wellness Agent
emoji: 🏋️
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Wellness Outcome Simulator — OpenEnv Environment

An **outcome-based** wellness simulator where an RL/LLM agent guides a simulated user toward better health outcomes — measured by **8 biomarker changes** (resting HR, HRV, VO2 max, body fat %, lean mass, sleep efficiency, cortisol proxy, energy level). Built for the [Scaler/Meta PyTorch Hackathon](https://openenv.org) using the OpenEnv specification.

## Overview

The environment is fundamentally **outcome-driven**: the agent takes actions (sleep/exercise/nutrition recommendations), and the reward is computed from how those actions change measurable biomarkers. Each persona has a **hidden physiological response model** — the same action produces different outcomes for different people, forcing the agent to learn optimal strategies through experience.

Key features:
- **8 biomarker outcomes** — resting HR, HRV, VO2 max, body fat %, lean mass, sleep efficiency, cortisol proxy, energy level
- **Goal-weighted rewards** — each health goal (weight loss, muscle gain, athletic performance, etc.) weights biomarkers differently
- **Hidden response models** — persona-specific physiology the agent must discover through observation
- **Persona-based compliance** — users don't always follow recommendations (25–70% compliance)
- **Cross-action interactions** — sleep debt hurts exercise gains, protein boosts recovery, overtraining spikes cortisol
- **Stochasticity** — life events, action noise, and biological variability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      inference.py                           │
│                                                             │
│  ┌──────────┐   prompt   ┌──────────┐                       │
│  │ LLM Agent│◄──────────►│ Rule-    │                       │
│  │ (GPT-4o) │            │ Based    │                       │
│  └──────────┘            └──────────┘                       │
│       │                       │                             │
│       │  Action               │  Action                     │
│       ▼                       ▼                             │
│  ┌────────────────────────────────┐                         │
│  │         WellnessEnv            │  ◄── env.py             │
│  │  reset() · step() · grade()   │                          │
│  └────────────────────────────────┘                         │
│       │                       │                             │
│       │  Observation,         │  Observation,               │
│       │  Reward, Done         │  Reward, Done               │
│       ▼                       ▼                             │
│  ┌──────────┐            ┌──────────┐                       │
│  │ LLM Agent│            │ Rule-    │                       │
│  │ (next    │            │ Based    │                       │
│  │  step)   │            │ (next)   │                       │
│  └──────────┘            └──────────┘                       │
│                                                             │
│  ── Internal components of WellnessEnv ──                   │
│                                                             │
│  ┌──────┐  ┌──────────┐  ┌──────────┐                       │
│  │Payoff│  │Simulator  │  │ Personas │  ◄── personas.py      │
│  │(goal-│  │(hidden    │  │ +Compli- │                       │
│  │weight│  │ response  │  │  ance    │                       │
│  │delta)│  │ models)   │  │          │                       │
│  └──────┘  └──────────┘  └──────────┘                       │
│  payoff.py  simulator.py                                    │
│                   │                                         │
│                   ▼                                         │
│           ┌──────────────┐                                  │
│           │   Graders    │  ◄── graders.py                  │
│           │ (0.0 → 1.0)  │                                  │
│           └──────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

**Reward flow:** Action → Simulator (hidden response model) → Biomarker deltas → Payoff (goal-weighted) → Reward (0-100)

**Round 1 (current):** Outcome-based biomarker payoff + LLM/fallback agent + persona compliance  
**Round 2 (planned):** Trained NN policies, richer state space, mobile app integration

## Tasks

| Task | Difficulty | Persona | Goal | Days | Compliance | Focus |
|------|-----------|---------|------|------|------------|-------|
| `single_goal` | Easy | Athletic Performance | Athletic Performance | 14 | 70% | Optimize primary biomarker (VO2 max) |
| `multi_outcome` | Medium | Stress Management | Stress Management | 30 | 50% | Balance all 8 biomarkers |
| `resistant_adaptation` | Hard | Weight Loss | Weight Loss | 30 | 25% | Improve outcomes despite low compliance |

## Quick Start

```bash
# Clone
git clone https://github.com/raviasha/Wellness-Outcome.git
cd Wellness-Outcome

# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run inference (rule-based fallback, no API key needed)
python inference.py

# Run with LLM agent
export OPENAI_API_KEY=your-key
export MODEL_NAME=gpt-4o-mini
python inference.py
```

## Docker

```bash
docker build -t wellness-simulator .
docker run -e OPENAI_API_KEY=your-key wellness-simulator
```

## Project Structure

```
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                 # HF Spaces deployment
├── requirements.txt           # Python dependencies
├── inference.py               # Baseline LLM agent + rule-based fallback
├── wellness_env/
│   ├── __init__.py            # Package exports
│   ├── models.py              # Pydantic models (Action, Biomarkers, Observation, etc.)
│   ├── env.py                 # WellnessEnv (reset/step/state/grade)
│   ├── simulator.py           # Hidden biomarker transition dynamics (response models)
│   ├── payoff.py              # Blended reward: 70% goal-weighted deltas + 30% state quality
│   ├── personas.py            # Persona configs + compliance + hidden ResponseModel
│   └── graders.py             # Multi-criteria task graders (0.0–1.0)
└── tests/
    ├── test_env.py            # OpenEnv interface + end-to-end tests
    ├── test_payoff.py         # Reward computation tests
    ├── test_simulator.py      # Transition dynamics tests
    └── test_graders.py        # Grader tests (112 tests total)
```

## OpenEnv Interface

```python
from wellness_env import WellnessEnv, Action, SleepDuration, ExerciseType, NutritionType

env = WellnessEnv(seed=42)
obs = env.reset("single_goal")

action = Action(
    sleep=SleepDuration.OPTIMAL_LOW,
    exercise=ExerciseType.MODERATE_CARDIO,
    nutrition=NutritionType.BALANCED,
)
result = env.step(action)

print(result.reward.total)            # Per-step reward (0-100, baseline 50)
print(result.observation.biomarkers)  # Current biomarker values
print(result.observation.deltas)      # Biomarker changes this step
print(result.observation.goal)        # User's health goal
print(result.info["complied"])        # Did the user follow the recommendation?

# After episode ends:
score = env.grade()                   # Grader score 0.0–1.0
```

## Action Space

150 discrete combinations (5 × 6 × 5):

| Category | Options |
|--------|---------|
| Sleep | `less_than_6h`, `6_to_7h`, `7_to_8h`, `8_to_9h`, `more_than_9h` |
| Exercise | `none`, `light_cardio`, `moderate_cardio`, `hiit`, `strength`, `yoga` |
| Nutrition | `skipped`, `processed`, `high_carb`, `balanced`, `high_protein` |

## Scoring

Rewards are **outcome-based**, computed per-step (0–100) using a blended formula:

$$R = 0.7 \times R_{\text{delta}} + 0.3 \times R_{\text{state}}$$

- **Delta component (70%)**: Rewards biomarker improvements. Each delta is normalized by a scale representing "excellent daily change," weighted by the user's health goal, and centered at 50
- **State quality component (30%)**: Rewards maintaining good absolute biomarker values. Prevents the agent from being penalized for stability once biomarkers are already healthy
- **Goal weighting** — different goals prioritize different biomarkers (e.g., weight loss weights body_fat_pct at 0.35, athletic performance weights VO2 max at 0.30)
- **Hidden response models** — each persona responds differently to the same action, so the agent must learn through observation

### Graders

Each task has a multi-criteria grader producing a 0.0–1.0 score:

**single_goal**: 60% avg reward + 20% primary biomarker improvement + 20% reward trend

**multi_outcome**: 35% avg reward + 25% biomarker breadth (fraction of 8 markers that improved) + 20% consistency (inverse stddev) + 20% positive trend

**resistant_adaptation**: 30% avg reward + 20% last-7-vs-first-7 improvement + 20% consistency + 15% compliance effectiveness + 15% breadth

### Baseline Scores

| Task | Score | Avg Reward | Stddev | Trend |
|------|-------|-----------|--------|-------|
| single_goal | 0.8271 | 83.58 | 9.91 | -0.30 |
| multi_outcome | 0.4154 | 53.50 | 35.15 | +0.06 |
| resistant_adaptation | 0.5965 | 43.13 | 28.86 | +0.66 |

## Stdout Format

```
[START] task=single_goal env=wellness-outcome model=gpt-4o-mini persona=athletic_performance goal=athletic_performance compliance=0.7 days=14
[STEP]  step=1 action={...} actual={...} complied=true biomarkers={"resting_hr":61.5,...} deltas={"resting_hr":-0.5,...} reward=57.50 done=false error=null
...
[END]   task=single_goal success=true steps=14 score=0.72 avg_reward=58.40 reward_trend=+1.05 reward_stddev=4.30 compliance_rate_actual=0.64 rewards=57.50,...
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No* | OpenAI API key (falls back to rule-based agent) |
| `API_BASE_URL` | No | Custom API endpoint (default: `https://api.openai.com/v1`) |
| `MODEL_NAME` | No | Model to use (default: `gpt-4o-mini`) |
| `HF_TOKEN` | No | Hugging Face token (used as API key if `OPENAI_API_KEY` not set) |
| `SEED` | No | Random seed for reproducibility (default: `42`) |

## Expected Scores

| Task | Random Agent | Rule-based Fallback | LLM Agent |
|------|-------------|-------------------|----------|
| single_goal | 0.20–0.30 | 0.82 | 0.82 |
| multi_outcome | 0.15–0.25 | 0.42 | 0.42 |
| resistant_adaptation | 0.05–0.15 | 0.60 | 0.60 |

## Roadmap — Round 2 Vision

| Feature | Description | Impact |
|---------|-------------|--------|
| **Trained NN Policies** | Replace prompt-based agent with a neural network trained via PPO/DQN on the simulator | Orders-of-magnitude faster inference, better long-horizon planning |
| **Richer State Space** | Add mood, stress, hydration, medication interactions | More realistic user modeling |
| **Multi-Agent** | Separate sleep coach, exercise coach, nutrition coach agents that coordinate | Specialized expertise per action category |
| **Mobile App** | React Native frontend showing daily recommendations + progress | Real user feedback loop |
| **Adaptive Personas** | Personas that evolve compliance over time based on agent effectiveness | Tests long-term engagement strategies |

## License

MIT
