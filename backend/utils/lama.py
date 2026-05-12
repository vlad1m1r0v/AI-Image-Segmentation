# Source: https://github.com/enesmsahin/simple-lama-inpainting

import os

import cv2
import numpy as np
import torch
from torch.hub import download_url_to_file

from utils.logger import get_logger

logger = get_logger(__name__)


def _ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _scale_image(img: np.ndarray, factor: float, interpolation: int = cv2.INTER_AREA):
    img = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    img = img[None, ...] if img.ndim == 2 else np.transpose(img, (2, 0, 1))

    return img


def _pad_img_to_modulo(img, mod):
    _, height, width = img.shape
    out_height = _ceil_modulo(height, mod)
    out_width = _ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def _get_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = np.transpose(image, (2, 0, 1))
    elif image.ndim == 2:
        image = image[np.newaxis, ...]

    result = image.astype(np.float32) / 255.0
    return np.asarray(result)


def prepare_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    pad_out_to_modulo=8,
    scale_factor=None,
):
    image = _get_image(image)
    mask = _get_image(mask)

    if scale_factor is not None:
        image = _scale_image(image, scale_factor)
        mask = _scale_image(image, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        image = _pad_img_to_modulo(image, pad_out_to_modulo)
        mask = _pad_img_to_modulo(mask, pad_out_to_modulo)

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

    mask_tensor = (mask_tensor > 0).float()

    return image_tensor, mask_tensor


def download_model(url: str, output_path: str):
    if not os.path.exists(output_path):
        download_url_to_file(url, output_path, hash_prefix=None, progress=True)
    return output_path
