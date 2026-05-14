from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class SegmentationParamsConfig(BaseModel):
    pred_iou_thresh: float
    stability_score_thresh: float


class SegmentationConfig(BaseModel):
    model: str
    device: str
    checkpoint: str
    config_file: str
    params: SegmentationParamsConfig


class InpaintingConfig(BaseModel):
    model_url: str
    device: str
    checkpoint: str

class MaskConfig(BaseModel):
    noise_kernel_size: int
    smooth_kernel_size: int
    dilate_iterations: int


class AppConfig(BaseSettings):
    segmentation: SegmentationConfig
    inpainting: InpaintingConfig
    mask: MaskConfig

    model_config = SettingsConfigDict(toml_file="config.toml")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


config = AppConfig()
