import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import config
from app.logger import get_logger
from app.segmentation.sam2_model import SAM2SegmentationModel

logger = get_logger(__name__)


def main() -> None:
    image_path = Path("image.png")
    output_path = Path(f"output.{config.output.object_format}")

    logger.info("Reading %s", image_path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        logger.error("Cannot read %s", image_path)
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    center = [[w // 2, h // 2]]
    logger.info("Image %dx%d, center point: %s", w, h, center)

    model = SAM2SegmentationModel()
    mask = model.predict(rgb, center)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), rgba)
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
