#!/usr/bin/env python3
"""Baseline LLM agent for the Outcome-Based Wellness Simulator.

Runs all 3 tasks sequentially, printing structured stdout.
Uses OpenAI-compatible API via env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure wellness_env is importable without pip install -e .
sys.path.append(str(Path(__file__).parent))

from wellness_env import WellnessEnv, Action, Observation
from wellness_env.models import SleepDuration, ExerciseType, NutritionType, Goal
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# LLM client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

SYSTEM_PROMPT = """\
You are an AI wellness coach controlling a day-by-day health simulation.
Your goal is to maximize the reward signal by choosing daily actions that improve
a simulated person's biomarkers over the course of the episode.

## Environment
- Each episode lasts a fixed number of days. Each step = 1 day.
- The person has a primary **goal** (e.g. weight_loss, athletic_performance, stress_management, muscle_gain, longevity, overall_wellness). Reward is weighted toward biomarkers relevant to that goal.
- The person has a **compliance_rate** — they may not always follow your recommendation. You cannot control this, but you can adapt.
- Random life events (bad sleep, missed exercise, social dinners) can override your action ~5% of the time.

## Action Space (pick exactly one from each category)
Sleep: "less_than_6h", "6_to_7h", "7_to_8h", "8_to_9h", "more_than_9h"
Exercise: "none", "light_cardio", "moderate_cardio", "hiit", "strength", "yoga"
Nutrition: "high_protein", "balanced", "high_carb", "processed", "skipped"

## Biomarkers you observe
- resting_hr (lower=better), hrv (higher=better), vo2_max (higher=better)
- body_fat_pct (lower=better), lean_mass_kg (higher=better)
- sleep_efficiency (higher=better), cortisol_proxy (lower=better, 0-100), energy_level (higher=better, 0-100)

## Reward
- Reward is computed from biomarker *changes* (deltas) weighted by the goal, blended with absolute state quality.
- Baseline reward is ~50. Improvements push above 50, regressions push below.
- Watch for overtraining: too many consecutive intense exercise days can backfire.
- Sleep debt accumulates and reduces exercise effectiveness.

## Strategy hints
- Use the action history and reward feedback to identify what works for this persona.
- If a biomarker is worsening, try changing the relevant action.
- Balance short-term intensity with recovery.

## OUTPUT: Respond with ONLY a JSON object, no markdown fences:
{"sleep": "...", "exercise": "...", "nutrition": "..."}
"""


def build_user_message(obs: Observation, step_num: int, history_actions: list[dict]) -> str:
    """Build user message with current state and action history."""
    b = obs.biomarkers
    d = obs.deltas

    # Current state
    lines = [
        f"Day {step_num}/{obs.total_days} | Goal: {obs.goal.value} | Compliance rate: {obs.compliance_rate:.0%}",
        "",
        "Current biomarkers:",
        f"  resting_hr={b.resting_hr:.1f}  hrv={b.hrv:.1f}  vo2_max={b.vo2_max:.2f}",
        f"  body_fat={b.body_fat_pct:.2f}%  lean_mass={b.lean_mass_kg:.2f}kg",
        f"  sleep_eff={b.sleep_efficiency:.1f}%  cortisol={b.cortisol_proxy:.1f}  energy={b.energy_level:.1f}",
        "",
        "Deltas (change from yesterday):",
        f"  resting_hr={d.resting_hr:+.3f}  hrv={d.hrv:+.3f}  vo2_max={d.vo2_max:+.4f}",
        f"  body_fat={d.body_fat_pct:+.4f}  lean_mass={d.lean_mass_kg:+.4f}",
        f"  sleep_eff={d.sleep_efficiency:+.3f}  cortisol={d.cortisol_proxy:+.3f}  energy={d.energy_level:+.3f}",
    ]

    # Trends if available
    if obs.trends:
        t = obs.trends
        lines.extend([
            "",
            "7-day trends:",
            f"  rhr_trend={t.resting_hr_trend:+.4f}  hrv_trend={t.hrv_trend:+.4f}  vo2_trend={t.vo2_max_trend:+.4f}",
            f"  bf_trend={t.body_fat_trend:+.4f}  lm_trend={t.lean_mass_trend:+.4f}  se_trend={t.sleep_efficiency_trend:+.4f}",
            f"  reward_trend={t.reward_trend:+.4f}  reward_consistency={t.reward_consistency:.4f}",
        ])

    # Action history (last 7 steps to stay within context limits)
    if history_actions:
        lines.extend(["", "Recent action history:"])
        for h in history_actions[-7:]:
            a = h["action"]
            actual = h.get("actual", a)
            complied = h.get("complied", True)
            compliance_note = "" if complied else " [NOT FOLLOWED]"
            lines.append(
                f"  Day {h['step']}: sleep={a['sleep']} exercise={a['exercise']} "
                f"nutrition={a['nutrition']} → reward={h['reward']:.2f}{compliance_note}"
            )
            if not complied:
                lines.append(
                    f"    (actual: sleep={actual['sleep']} exercise={actual['exercise']} "
                    f"nutrition={actual['nutrition']})"
                )
    else:
        lines.extend(["", "No action history yet (first day)."])

    lines.append("")
    lines.append("Choose your action for today.")
    return "\n".join(lines)


def call_llm(obs: Observation, step_num: int, history_actions: list[dict]) -> Action:
    """Call the LLM and parse the response into an Action."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(obs, step_num, history_actions)},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_tokens=80,
        )

        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        data = json.loads(content)
        return Action(
            sleep=SleepDuration(data["sleep"]),
            exercise=ExerciseType(data["exercise"]),
            nutrition=NutritionType(data["nutrition"]),
        )
    except Exception:
        return _fallback_action(obs)


def _fallback_action(obs: Observation) -> Action:
    """Rule-based fallback when LLM is unavailable.

    Uses biomarker values and day number to make decisions.
    """
    b = obs.biomarkers
    d = obs.deltas
    goal = obs.goal
    day = obs.day

    # Recovery priority: if cortisol is high or energy is low
    if b.cortisol_proxy > 65 or b.energy_level < 30:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.YOGA,
            nutrition=NutritionType.BALANCED,
        )

    # Low compliance: moderate, achievable changes
    if obs.compliance_rate <= 0.3:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.LIGHT_CARDIO,
            nutrition=NutritionType.BALANCED,
        )

    # Goal-specific strategies
    if goal == Goal.WEIGHT_LOSS:
        exercise = ExerciseType.MODERATE_CARDIO if b.energy_level > 50 else ExerciseType.LIGHT_CARDIO
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=exercise,
            nutrition=NutritionType.BALANCED,
        )
    elif goal == Goal.MUSCLE_GAIN:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.STRENGTH,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
    elif goal == Goal.ATHLETIC_PERFORMANCE:
        if d.hrv < -2:  # HRV dropped → need recovery
            return Action(
                sleep=SleepDuration.OPTIMAL_HIGH,
                exercise=ExerciseType.YOGA,
                nutrition=NutritionType.HIGH_PROTEIN,
            )
        # Alternate intense/recovery to avoid overtraining (threshold=2-3)
        if day % 3 == 0:
            exercise = ExerciseType.YOGA  # recovery day
        elif day % 2 == 1:
            exercise = ExerciseType.HIIT
        else:
            exercise = ExerciseType.STRENGTH
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=exercise,
            nutrition=NutritionType.HIGH_PROTEIN,
        )
    elif goal == Goal.STRESS_MANAGEMENT:
        return Action(
            sleep=SleepDuration.OPTIMAL_HIGH,
            exercise=ExerciseType.YOGA,
            nutrition=NutritionType.BALANCED,
        )
    elif goal == Goal.LONGEVITY:
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.MODERATE_CARDIO,
            nutrition=NutritionType.BALANCED,
        )
    else:  # overall_wellness
        return Action(
            sleep=SleepDuration.OPTIMAL_LOW,
            exercise=ExerciseType.MODERATE_CARDIO,
            nutrition=NutritionType.BALANCED,
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

TASKS = ["single_goal", "multi_outcome", "resistant_adaptation"]


def run_task(env: WellnessEnv, task_name: str, use_llm: bool = True) -> None:
    """Run a single task and print structured stdout."""
    obs = env.reset(task_name)
    config = env._config
    persona = env._persona

    print(
        f"[START] task={task_name} env=wellness-outcome model={MODEL_NAME}"
    )

    rewards: list[float] = []
    history_actions: list[dict] = []

    for step_num in range(1, config["total_days"] + 1):
        try:
            if use_llm:
                action = call_llm(obs, step_num, history_actions)
            else:
                action = _fallback_action(obs)

            result = env.step(action)

            action_dict = action.model_dump()
            actual_dict = result.info["actual_action"]
            complied = result.info["complied"]
            reward_val = result.reward.total
            rewards.append(reward_val)

            history_actions.append({
                "step": step_num,
                "action": action_dict,
                "actual": actual_dict,
                "complied": complied,
                "reward": reward_val,
            })

            action_str = json.dumps(action_dict)
            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward_val:.2f} "
                f"done={str(result.done).lower()} "
                f"error=null"
            )

            obs = result.observation
        except Exception as e:
            error_msg = str(e).replace("\n", " ")[:200]
            print(
                f"[STEP] step={step_num} "
                f"action=null "
                f"reward=0.00 "
                f"done=false "
                f"error=\"{error_msg}\""
            )

    # End summary
    score = env.grade()
    success = score >= 0.1
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={score:.2f} rewards={rewards_str}"
    )


def main():
    use_llm = bool(OPENAI_API_KEY)
    if not use_llm:
        print(
            "# WARNING: No API key found. Using rule-based fallback agent.",
            file=sys.stderr,
        )

    seed = int(os.environ.get("SEED", "42"))
    env = WellnessEnv(seed=seed)

    for task_name in TASKS:
        try:
            run_task(env, task_name, use_llm=use_llm)
        except Exception:
            print(
                f"[END] success=false steps=0 score=0.00 rewards="
            )
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
