"""
Reward Function
===============
Computes per-step partial reward and final plan score.

Reward = tumor_coverage * 0.50
       - oar_penalty    * 0.40
       + plan_efficiency* 0.10

Range: [0.0, 1.0] (clipped)

Design principles:
  1. Partial reward every step → agent learns faster (dense reward signal)
  2. Tumor coverage dominates (50%) — primary clinical goal
  3. OAR penalty is weighted by organ priority (critical OARs penalize more)
  4. Plan efficiency rewards compact plans (fewer beams when possible)
  5. Final score uses stricter clinical criteria than training reward
"""

import numpy as np
from typing import List
from ..physics.phantom import PatientPhantom, Beam


# OAR priority weights for penalty computation
PRIORITY_WEIGHTS = {1: 1.5, 2: 1.0, 3: 0.5}  # critical, important, moderate


def compute_reward(
    dose: np.ndarray,
    patient: PatientPhantom,
    beams: List[Beam],
) -> float:
    """
    Compute per-step training reward in [0.0, 1.0].

    Args:
        dose:    Current dose distribution grid
        patient: Patient phantom with tumor and OARs
        beams:   Current list of active beams

    Returns:
        reward: float in [0.0, 1.0]
    """
    if not beams:
        return 0.0

    # ── 1. Tumor Coverage (weight: 0.50) ─────────────────────────────────────
    tumor_dose = dose[patient.tumor_mask]
    if len(tumor_dose) == 0:
        tumor_coverage = 0.0
    else:
        # Primary metric: fraction of tumor getting >= 95% of prescription
        tumor_coverage = float(
            np.mean(tumor_dose >= 0.95 * patient.prescription_dose)
        )
        # Smooth bonus for mean tumor dose approaching prescription
        mean_dose_ratio = np.mean(tumor_dose) / (patient.prescription_dose + 1e-8)
        tumor_coverage = 0.8 * tumor_coverage + 0.2 * min(1.0, mean_dose_ratio)

    # ── 2. OAR Penalty (weight: 0.40) ─────────────────────────────────────────
    oar_penalty = 0.0
    total_weight = 0.0

    for oar in patient.oars:
        oar_dose = dose[oar.mask]
        if len(oar_dose) == 0:
            continue

        w = PRIORITY_WEIGHTS.get(oar.priority, 1.0)
        mean_dose = float(np.mean(oar_dose))
        max_dose  = float(np.max(oar_dose))

        # Mean dose violation (normalized)
        mean_violation = max(0.0, mean_dose - oar.limit) / (oar.limit + 1e-8)

        # Max dose violation (for critical serial organs like spinal cord)
        if oar.priority == 1:
            max_violation = max(0.0, max_dose - oar.limit * 1.1) / (oar.limit + 1e-8)
            violation = 0.6 * mean_violation + 0.4 * max_violation
        else:
            violation = mean_violation

        oar_penalty += w * min(violation, 2.0)  # cap at 2x limit
        total_weight += w

    if total_weight > 0:
        oar_penalty /= total_weight  # normalize to [0, ~1]
    oar_penalty = min(oar_penalty, 1.0)

    # ── 3. Plan Efficiency (weight: 0.10) ────────────────────────────────────
    # Reward for clean plans: fewer beams with higher dose weight is better
    n_beams = len(beams)
    if n_beams == 0:
        plan_efficiency = 0.0
    else:
        # Optimal: 5-7 beams
        beam_efficiency = 1.0 - abs(n_beams - 6) / 7.0
        beam_efficiency = max(0.0, beam_efficiency)

        # Reward beams with meaningful dose weights (not too low)
        mean_weight = np.mean([b.dose_weight for b in beams])
        weight_efficiency = min(1.0, mean_weight / 0.6)

        plan_efficiency = 0.7 * beam_efficiency + 0.3 * weight_efficiency

    # ── Final reward ─────────────────────────────────────────────────────────
    reward = (
        tumor_coverage * 0.50
        - oar_penalty  * 0.40
        + plan_efficiency * 0.10
    )

    return float(np.clip(reward, 0.0, 1.0))


def compute_score(
    dose: np.ndarray,
    patient: PatientPhantom,
    beams: List[Beam],
) -> float:
    """
    Compute FINAL plan quality score [0.0, 1.0] for the auto-grader.

    Uses stricter clinical criteria:
      - Tumor D95 >= 95% prescription (hard requirement)
      - All OAR mean doses within limits
      - Critical OAR max doses within limits

    This is what judges see — not the training reward.
    """
    if not beams or dose is None:
        return 0.0

    score_components = []

    # ── Tumor coverage score ─────────────────────────────────────────────────
    tumor_dose = dose[patient.tumor_mask]
    if len(tumor_dose) > 0:
        d95 = float(np.percentile(tumor_dose, 5))  # D95
        coverage_95 = float(np.mean(tumor_dose >= 0.95 * patient.prescription_dose))
        tumor_score = 0.5 * min(1.0, d95 / patient.prescription_dose) + 0.5 * coverage_95
    else:
        tumor_score = 0.0
    score_components.append(("tumor", tumor_score, 0.55))

    # ── OAR compliance score ─────────────────────────────────────────────────
    oar_scores = []
    for oar in patient.oars:
        oar_dose = dose[oar.mask]
        if len(oar_dose) == 0:
            oar_scores.append(1.0)
            continue

        mean_dose = float(np.mean(oar_dose))
        max_dose  = float(np.max(oar_dose))

        if oar.priority == 1:
            # Critical: both mean and max must be within limits
            mean_ok = mean_dose <= oar.limit
            max_ok  = max_dose  <= oar.limit * 1.05
            oar_score = (0.5 * float(mean_ok) + 0.5 * float(max_ok))
        else:
            # Non-critical: linear gradient
            oar_score = max(0.0, 1.0 - max(0, mean_dose - oar.limit) / (oar.limit * 0.5))

        oar_scores.append(oar_score)

    oar_score = float(np.mean(oar_scores)) if oar_scores else 1.0
    score_components.append(("oar", oar_score, 0.40))

    # ── Plan efficiency score ─────────────────────────────────────────────────
    n = len(beams)
    eff = max(0.0, 1.0 - abs(n - 6) / 7.0)
    score_components.append(("efficiency", eff, 0.05))

    # Weighted final score
    final = sum(s * w for _, s, w in score_components)
    return float(np.clip(final, 0.0, 1.0))
