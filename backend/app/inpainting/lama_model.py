# Source: https://github.com/enesmsahin/simple-lama-inpainting

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.inpainting.utils import download_model, prepare_img_and_mask
from app.logger import get_logger

logger = get_logger(__name__)

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


class SimpleLama:
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        project_root = Path(__file__).parents[2]
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "big-lama.pt"

        if not model_path.exists():
            logger.info(f"Downloading LaMa model to {model_path}...")
            download_model(LAMA_MODEL_URL, str(model_path))

        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            cur_res = Image.fromarray(cur_res)
            return cur_res
