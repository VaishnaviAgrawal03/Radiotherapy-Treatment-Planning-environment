"""Base Task — abstract class for all tasks."""
from abc import ABC, abstractmethod
import numpy as np
from ..physics.phantom import PatientPhantom


class BaseTask(ABC):
    """Abstract base for all radiotherapy tasks."""

    @abstractmethod
    def sample_patient(self, rng: np.random.Generator) -> PatientPhantom:
        """Sample a new patient case."""
        ...

    @abstractmethod
    def reward(self, dose: np.ndarray, patient: PatientPhantom, beams: list) -> float:
        """Compute task-specific reward."""
        ...
