import cv2
import numpy as np

from ai.models.inpainting import LaMaInpaintingModel
from ai.models.segmentation import SAM2SegmentationModel
from utils.logger import get_logger
from utils.mask import dilate_mask, remove_mask_noise, smooth_mask

logger = get_logger(__name__)


def load_image(path: str) -> np.ndarray:
    """Loads an image and converts it to RGB format."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not find the file at path: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str, image_rgb: np.ndarray):
    """Saves an RGB image to the disk."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def save_rgba_object(path: str, image_rgb: np.ndarray, mask_uint8: np.ndarray):
    """Creates and saves an object with a transparent background (RGBA)."""
    # Create the 4th channel (alpha) based on the mask
    rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask_uint8

    # Save as BGRA (as expected by OpenCV)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, bgra)


# --- Main Script ---

if __name__ == "__main__":
    # 1. Models initialization
    logger.info("Initializing models... please wait.")
    sam2 = SAM2SegmentationModel()
    lama = LaMaInpaintingModel()

    # 2. Loading the image
    image_path = "image.png"
    img = load_image(image_path)
    h, w, _ = img.shape
    logger.info(f"Image loaded: {w}x{h}")

    # 3. Segmentation (using the center point)
    center_point = [[w // 2, h // 2]]
    logger.info(f"Analyzing object at point: {center_point}...")
    # sam2 returns uint8 (0/255) as per our previous updates
    raw_mask = sam2(img, center_point)

    # 4. Mask processing (Cleaning & Smoothing)
    logger.info("Processing mask (noise removal and smoothing)...")
    clean = remove_mask_noise(raw_mask)
    smooth = smooth_mask(clean)

    # 5. Extracting the object (extracted_object.png)
    logger.info("Saving the extracted object...")
    save_rgba_object("extracted_object.png", img, smooth)

    # 6. Mask preparation for LaMa (Dilation)
    logger.info("Expanding the mask for inpainting (dilation)...")
    # Create a copy of the mask with dilation
    final_mask_for_lama = dilate_mask(smooth, iterations=12)

    # 7. Background inpainting (LaMa)
    logger.info("Inpainting background via LaMa... this may take a few seconds.")
    result_bg = lama(img, final_mask_for_lama)

    # 8. Saving the final result
    save_image("result_background.png", result_bg)

    logger.info("Execution finished successfully:")
    logger.info(" - Object saved to: extracted_object.png")
    logger.info(" - Background without object saved to: result_background.png")
