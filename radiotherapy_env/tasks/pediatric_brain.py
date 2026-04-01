"""Task 3: Pediatric Brain Tumor (Hard)"""
import numpy as np
from .base_task import BaseTask
from ..physics.phantom import PediatricBrainPatientGenerator
from ..reward.reward_fn import compute_reward


class PediatricBrainTask(BaseTask):
    """Hard task — pediatric brain tumor adjacent to brainstem."""
    def __init__(self):
        self._gen = PediatricBrainPatientGenerator()

    def sample_patient(self, rng):
        return self._gen.generate(rng)

    def reward(self, dose, patient, beams):
        base = compute_reward(dose, patient, beams)
        # Hard mode: steep penalty if brainstem is overdosed, but not zeroed
        # This keeps reward signal meaningful for RL learning
        brainstem = next((o for o in patient.oars if o.name == "Brainstem"), None)
        if brainstem is not None:
            bs_dose = dose[brainstem.mask]
            if len(bs_dose) > 0:
                bs_max = float(np.max(bs_dose))
                bs_mean = float(np.mean(bs_dose))
                # Graduated penalty: scales with how much limit is exceeded
                if bs_mean > brainstem.limit * 1.5:
                    base *= 0.3   # severe violation
                elif bs_mean > brainstem.limit * 1.2:
                    base *= 0.55  # moderate violation
                elif bs_mean > brainstem.limit:
                    base *= 0.75  # mild violation
        return float(base)
