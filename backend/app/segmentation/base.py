from abc import ABC, abstractmethod

import numpy as np


class SegmentationModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray, points: list[list[int]]) -> np.ndarray:
        """
        Segment by prompt points and return a binary mask.

        image:  HxWx3 uint8 RGB array
        points: pixel coordinates [[x, y], ...]
            1 pt  → point prompt
            2 pts → bbox (top-left, bottom-right)
            3+ pts → foreground polygon points
        Returns: HxW bool mask (True = object)
        """
