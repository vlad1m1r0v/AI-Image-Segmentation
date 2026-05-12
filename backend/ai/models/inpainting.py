# Source: https://github.com/enesmsahin/simple-lama-inpainting

import numpy as np
import torch

from utils.lama import download_model, prepare_image_and_mask
from utils.logger import get_logger

from .config import config

logger = get_logger(__name__)


class LaMaInpaintingModel:
    def __init__(
        self,
        device: torch.device = torch.device(config.inpainting.device),
    ) -> None:
        logger.info(
            f"Downloading LaMa model from {config.inpainting.model_url} to {config.inpainting.checkpoint}\n..."
        )
        download_model(config.inpainting.model_url, config.inpainting.checkpoint)
        logger.info("LaMa model is ready.")

        self.model = torch.jit.load(
            str(config.inpainting.checkpoint), map_location=device
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_tensor, mask_tensor = prepare_image_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image_tensor, mask_tensor)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            h, w = image.shape[:2]
            if cur_res.shape[0] != h or cur_res.shape[1] != w:
                cur_res = cur_res[:h, :w, :]

            return cur_res
