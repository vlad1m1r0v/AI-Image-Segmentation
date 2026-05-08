from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class SegmentationParamsConfig(BaseModel):
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95


class SegmentationConfig(BaseModel):
    model: str
    device: str
    checkpoint: str
    config_file: str
    params: SegmentationParamsConfig


class OutputConfig(BaseModel):
    object_format: str


class AppConfig(BaseSettings):
    segmentation: SegmentationConfig
    output: OutputConfig

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
