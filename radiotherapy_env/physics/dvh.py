"""
DVH (Dose-Volume Histogram) Calculator
=======================================
Computes the cumulative DVH for a structure.

DVH[i] = fraction of structure volume receiving >= dose_bins[i]

The output is normalized so that:
  - x-axis: dose normalized by the structure's reference dose (limit or prescription)
  - y-axis: volume fraction [0, 1]

This representation is compact (n_bins values) and directly usable
as RL observations.
"""

import numpy as np


class DVHCalculator:
    """Compute normalized cumulative Dose-Volume Histograms."""

    def __init__(self, n_bins: int = 50, max_dose_factor: float = 2.0):
        """
        Args:
            n_bins: Number of histogram bins
            max_dose_factor: Maximum dose as multiple of reference dose
        """
        self.n_bins = n_bins
        self.max_dose_factor = max_dose_factor
        # Dose bins: 0.0 to max_dose_factor, normalized
        self.dose_bins = np.linspace(0.0, max_dose_factor, n_bins, dtype=np.float32)

    def compute(
        self,
        dose: np.ndarray,
        mask: np.ndarray,
        reference_dose: float,
    ) -> np.ndarray:
        """
        Compute cumulative DVH for a structure.

        Args:
            dose:           Full dose grid (H, W)
            mask:           Boolean mask for the structure
            reference_dose: Prescription dose (tumor) or limit dose (OAR)
                            Used to normalize the dose axis.

        Returns:
            dvh: (n_bins,) float32 array
                 dvh[i] = fraction of structure with dose >= dose_bins[i] * reference_dose
        """
        structure_dose = dose[mask]

        if len(structure_dose) == 0 or reference_dose <= 0:
            return np.zeros(self.n_bins, dtype=np.float32)

        # Normalize dose by reference
        norm_dose = structure_dose / (reference_dose + 1e-8)

        # Compute cumulative DVH: for each bin, fraction of volume >= that dose
        dvh = np.array(
            [np.mean(norm_dose >= d) for d in self.dose_bins],
            dtype=np.float32,
        )

        return dvh
