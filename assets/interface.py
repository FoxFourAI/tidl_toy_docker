from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    bboxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

