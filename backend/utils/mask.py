import cv2
import numpy as np


def remove_mask_noise(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return smoothed


def dilate_mask(mask: np.ndarray, iterations: int = 10) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated
