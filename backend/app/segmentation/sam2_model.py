from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from app.config import config
from app.logger import get_logger
from app.segmentation.base import SegmentationModel

logger = get_logger(__name__)


class SAM2SegmentationModel(SegmentationModel):
    def __init__(self) -> None:
        logger.info(
            "Loading SAM 2 (%s) on %s...",
            config.segmentation.model,
            config.segmentation.device,
        )
        device = torch.device(config.segmentation.device)
        # build_sam2 uses Hydra with search path pkg://sam2 (the package root).
        # Configs live directly there, so the name is just the filename.
        # "models/sam2_hiera_s.yaml" -> "sam2_hiera_s.yaml"
        cfg_name = Path(config.segmentation.config_file).name
        sam2 = build_sam2(cfg_name, config.segmentation.checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(sam2)
        self._params = config.segmentation.params
        logger.info("Model ready.")

    def predict(self, image: np.ndarray, points: list[list[int]]) -> np.ndarray:
        self._predictor.set_image(image)
        n = len(points)

        if n == 1:
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
        elif n == 2:
            x0, y0 = points[0]
            x1, y1 = points[1]
            masks, scores, _ = self._predictor.predict(
                box=np.array([x0, y0, x1, y1], dtype=np.float32),
                multimask_output=False,
            )
        else:
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.ones(n, dtype=np.int32),
                multimask_output=True,
            )

        thresh = self._params.pred_iou_thresh
        valid = scores >= thresh
        idx = int(np.argmax(scores if not valid.any() else scores * valid))
        return masks[idx].astype(bool)
