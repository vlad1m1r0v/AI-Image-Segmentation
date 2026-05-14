from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.logger import get_logger

from config import config

logger = get_logger(__name__)


class SAM2SegmentationModel:
    def __init__(self) -> None:
        logger.info(
            "Loading SAM 2 (%s) on %s...",
            config.segmentation.model,
            config.segmentation.device,
        )
        device = torch.device(config.segmentation.device)

        cfg_name = Path(config.segmentation.config_file).name
        sam2 = build_sam2(cfg_name, config.segmentation.checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(sam2)
        self._params = config.segmentation.params
        logger.info("SAM 2 model is ready.")

    def __call__(self, image: np.ndarray, points: list[list[int]]) -> np.ndarray:
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
        mask_bool = masks[idx]
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255

        return mask_uint8
