"""
Task Registry
=============
Defines the 3 graded tasks and maps them to patient generators.
"""

from .base_task import BaseTask
from .prostate import ProstateTask
from .head_neck import HeadNeckTask
from .pediatric_brain import PediatricBrainTask

TASK_REGISTRY = {
    "prostate":       ProstateTask,
    "head_neck":      HeadNeckTask,
    "pediatric_brain": PediatricBrainTask,
}

__all__ = ["TASK_REGISTRY", "BaseTask"]
