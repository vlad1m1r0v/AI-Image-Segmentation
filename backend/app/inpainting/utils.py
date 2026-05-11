# Source: https://github.com/enesmsahin/simple-lama-inpainting

import os
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir

from app.logger import get_logger

logger = get_logger(__name__)


def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array.")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    img = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    img = img[None, ...] if img.ndim == 2 else np.transpose(img, (2, 0, 1))

    return img


def pad_img_to_modulo(img, mod):
    _, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def extract_mask_from_rgba(
    rgba_image: np.ndarray, dilation_iterations: int = 10
) -> np.ndarray:
    if rgba_image.shape[2] != 4:
        raise ValueError("Image must be in RGBA format.")

    alpha = rgba_image[:, :, 3]
    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)

    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    return dilated_mask


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url: str, output_path: str):
    if not os.path.exists(output_path):
        logger.info(f'Downloading: "{url}" to {output_path}\n')
        download_url_to_file(url, output_path, hash_prefix=None, progress=True)
    return output_path
