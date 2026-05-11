import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import config
from app.inpainting import SimpleLama, extract_mask_from_rgba
from app.logger import get_logger
from app.segmentation import SAM2SegmentationModel

logger = get_logger(__name__)


def main() -> None:
    image_path = Path("image.png")
    object_path = Path(f"object.{config.output.object_format}")
    mask_path = Path(f"mask.{config.output.object_format}")
    inpainted_path = Path(f"image_without_object.{config.output.object_format}")

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
    cv2.imwrite(str(object_path), rgba)
    logger.info("Saved object: %s", object_path)

    object_img = cv2.imread(str(object_path), cv2.IMREAD_UNCHANGED)
    binary_mask = extract_mask_from_rgba(object_img)

    cv2.imwrite(str(mask_path), binary_mask)
    logger.info("Saved mask from object: %s", mask_path)

    logger.info("Starting inpainting with SimpleLama...")

    lama = SimpleLama()
    pil_img = Image.fromarray(rgb)
    pil_mask = Image.fromarray(binary_mask).convert("L")

    result_pil = lama(pil_img, pil_mask)
    result_pil.save(str(inpainted_path))
    logger.info("Final inpainted image saved: %s", inpainted_path)


if __name__ == "__main__":
    main()
