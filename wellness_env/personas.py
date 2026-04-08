"""Persona definitions with hidden physiological response models.

Each persona has a unique response model — the mapping from actions to
biomarker changes is different per person.  The agent never sees these
parameters; it only sees the resulting biomarker deltas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .models import (
    Action,
    Biomarkers,
    ExerciseType,
    Goal,
    NutritionType,
    SleepDuration,
)


@dataclass
class ResponseModel:
    """Hidden parameters controlling how a persona responds to actions.

    These are NOT visible to the agent.  They define this simulated
    person's unique physiology.
    """

    # ----- Sleep response -----
    # How much HRV improves per hour of sleep above 7h (ms/hour)
    hrv_sleep_sensitivity: float = 3.0
    # How much resting HR drops per optimal-sleep night (bpm)
    rhr_sleep_benefit: float = -0.3
    # Sleep efficiency baseline (%) — some people sleep more efficiently
    sleep_efficiency_base: float = 85.0
    # How much cortisol drops per optimal-sleep night
    cortisol_sleep_recovery: float = -3.0

    # ----- Exercise response -----
    # VO2 max improvement per cardio session (ml/kg/min)
    vo2_cardio_gain: float = 0.15
    # Lean mass gain per strength session (kg)
    lean_mass_strength_gain: float = 0.05
    # Body fat loss per intense exercise session (%)
    body_fat_exercise_loss: float = -0.03
    # Resting HR improvement per exercise session (bpm)
    rhr_exercise_benefit: float = -0.1
    # Cortisol rise from intense exercise (0-100 scale)
    cortisol_exercise_stress: float = 5.0
    # Overtraining threshold — consecutive intense days before harm
    overtraining_threshold: int = 3

    # ----- Nutrition response -----
    # Body fat change per day from nutrition quality (-0.05 to +0.05)
    body_fat_nutrition_sensitivity: float = 0.02
    # Lean mass response to protein (kg per high-protein day)
    lean_mass_protein_gain: float = 0.02
    # Energy response to nutrition quality (0-100 scale per day)
    energy_nutrition_sensitivity: float = 8.0
    # Cortisol response to poor nutrition
    cortisol_nutrition_stress: float = 3.0

    # ----- Cross-action sensitivities -----
    # How much sleep debt hurts exercise gains (multiplier reduction per debt hour)
    sleep_debt_exercise_penalty: float = 0.1
    # How much protein intake boosts post-exercise recovery (multiplier)
    protein_recovery_multiplier: float = 1.3
    # Overtraining cortisol spike
    overtraining_cortisol_spike: float = 15.0


@dataclass
class PersonaConfig:
    """Full persona definition: identity + defaults + hidden response model."""

    name: str
    compliance_rate: float
    goal: Goal
    sleep_default: SleepDuration
    exercise_default: ExerciseType
    nutrition_default: NutritionType
    starting_biomarkers: Biomarkers
    response_model: ResponseModel
    random_defaults: bool = False


# ---------------------------------------------------------------------------
# Persona library
# ---------------------------------------------------------------------------

PERSONAS: dict[str, PersonaConfig] = {
    "athletic_performance": PersonaConfig(
        name="athletic_performance",
        compliance_rate=0.7,
        goal=Goal.ATHLETIC_PERFORMANCE,
        sleep_default=SleepDuration.SHORT,
        exercise_default=ExerciseType.HIIT,
        nutrition_default=NutritionType.HIGH_PROTEIN,
        starting_biomarkers=Biomarkers(
            resting_hr=62.0,
            hrv=55.0,
            vo2_max=42.0,
            body_fat_pct=18.0,
            lean_mass_kg=65.0,
            sleep_efficiency=78.0,
            cortisol_proxy=35.0,
            energy_level=70.0,
        ),
        response_model=ResponseModel(
            # Very responsive to exercise
            vo2_cardio_gain=0.25,
            lean_mass_strength_gain=0.08,
            body_fat_exercise_loss=-0.05,
            rhr_exercise_benefit=-0.15,
            # But sensitive to overtraining
            overtraining_threshold=3,
            overtraining_cortisol_spike=20.0,
            # Moderate sleep sensitivity
            hrv_sleep_sensitivity=3.0,
            cortisol_sleep_recovery=-3.0,
            # Good protein response
            protein_recovery_multiplier=1.4,
            lean_mass_protein_gain=0.03,
        ),
    ),
    "stress_management": PersonaConfig(
        name="stress_management",
        compliance_rate=0.65,
        goal=Goal.STRESS_MANAGEMENT,
        sleep_default=SleepDuration.VERY_SHORT,
        exercise_default=ExerciseType.NONE,
        nutrition_default=NutritionType.PROCESSED,
        starting_biomarkers=Biomarkers(
            resting_hr=78.0,
            hrv=30.0,
            vo2_max=30.0,
            body_fat_pct=28.0,
            lean_mass_kg=55.0,
            sleep_efficiency=65.0,
            cortisol_proxy=75.0,
            energy_level=35.0,
        ),
        response_model=ResponseModel(
            # Very responsive to sleep
            hrv_sleep_sensitivity=5.0,
            rhr_sleep_benefit=-0.5,
            cortisol_sleep_recovery=-5.0,
            sleep_efficiency_base=70.0,
            # Moderate exercise response
            vo2_cardio_gain=0.12,
            lean_mass_strength_gain=0.04,
            body_fat_exercise_loss=-0.03,
            # High cortisol sensitivity to stress
            cortisol_exercise_stress=8.0,
            overtraining_cortisol_spike=25.0,
            # Very sensitive to poor nutrition
            cortisol_nutrition_stress=6.0,
            energy_nutrition_sensitivity=12.0,
        ),
    ),
    "weight_loss": PersonaConfig(
        name="weight_loss",
        compliance_rate=0.25,
        goal=Goal.WEIGHT_LOSS,
        sleep_default=SleepDuration.SHORT,
        exercise_default=ExerciseType.NONE,
        nutrition_default=NutritionType.PROCESSED,
        starting_biomarkers=Biomarkers(
            resting_hr=82.0,
            hrv=25.0,
            vo2_max=25.0,
            body_fat_pct=35.0,
            lean_mass_kg=58.0,
            sleep_efficiency=70.0,
            cortisol_proxy=55.0,
            energy_level=40.0,
        ),
        response_model=ResponseModel(
            # Slow but steady body fat response
            body_fat_exercise_loss=-0.04,
            body_fat_nutrition_sensitivity=0.03,
            # Less responsive to exercise initially (low fitness)
            vo2_cardio_gain=0.08,
            lean_mass_strength_gain=0.03,
            rhr_exercise_benefit=-0.08,
            # Higher overtraining risk (deconditioned)
            overtraining_threshold=2,
            cortisol_exercise_stress=10.0,
            overtraining_cortisol_spike=20.0,
            # Strong sleep-debt-to-exercise penalty
            sleep_debt_exercise_penalty=0.15,
            # Moderate sleep gains
            hrv_sleep_sensitivity=3.5,
            cortisol_sleep_recovery=-4.0,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Compliance model (same as original)
# ---------------------------------------------------------------------------

def apply_compliance(
    recommended: Action,
    persona: PersonaConfig,
    rng: random.Random,
) -> tuple[Action, bool]:
    """Apply persona compliance model.  Returns (actual_action, complied)."""
    if rng.random() < persona.compliance_rate:
        return recommended, True

    # Non-compliant: 60% revert to defaults, 40% random
    if rng.random() < 0.6:
        if persona.random_defaults:
            actual = Action(
                sleep=rng.choice(list(SleepDuration)),
                exercise=rng.choice(list(ExerciseType)),
                nutrition=rng.choice(list(NutritionType)),
            )
        else:
            actual = Action(
                sleep=persona.sleep_default,
                exercise=persona.exercise_default,
                nutrition=persona.nutrition_default,
            )
    else:
        actual = Action(
            sleep=rng.choice(list(SleepDuration)),
            exercise=rng.choice(list(ExerciseType)),
            nutrition=rng.choice(list(NutritionType)),
        )
    return actual, False
