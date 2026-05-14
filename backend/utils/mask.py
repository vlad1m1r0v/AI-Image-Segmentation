import cv2
import numpy as np

from config import config


def remove_mask_noise(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((config.mask.noise_kernel_size, config.mask.noise_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (config.mask.smooth_kernel_size, config.mask.smooth_kernel_size), 0)
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return smoothed


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=config.mask.dilate_iterations)
    return dilated
