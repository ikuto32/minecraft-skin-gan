from __future__ import annotations

from dataclasses import dataclass

from src.models.dcgan import (
    DCGANDiscriminator,
    DCGANGenerator,
    DCGANImprovedDiscriminator,
    DCGANImprovedGenerator,
    ResNetDiscriminator64,
    ResConvGenerator,
)
from src.models.r3gan import R3GANDiscriminator, R3GANGenerator


@dataclass(frozen=True)
class ModelSpec:
    generator_cls: type
    discriminator_cls: type
    generator_hparams: dict[str, object]
    discriminator_hparams: dict[str, object]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "r3gan": ModelSpec(
        generator_cls=R3GANGenerator,
        discriminator_cls=R3GANDiscriminator,
        generator_hparams={
            "channels": 4,
            "WidthPerStage": [384, 256, 192, 128, 96],
            "CardinalityPerStage": [48, 32, 24, 16, 12],
            "BlocksPerStage": [2, 2, 2, 2, 2],
            "ExpansionFactor": 2,
            "KernelSize": 3,
            "ResamplingFilter": [1, 2, 1],
        },
        discriminator_hparams={
            "channels": 4,
            "WidthPerStage": [96, 128, 192, 256, 384],
            "CardinalityPerStage": [12, 16, 24, 32, 48],
            "BlocksPerStage": [2, 2, 2, 2, 2],
            "ExpansionFactor": 2,
            "KernelSize": 3,
            "ResamplingFilter": [1, 2, 1],
        },
    ),
    "dcgan_baseline": ModelSpec(
        generator_cls=DCGANGenerator,
        discriminator_cls=DCGANDiscriminator,
        generator_hparams={"features": 64, "channels": 4},
        discriminator_hparams={"features": 64, "channels": 4},
    ),
    "dcgan_improved": ModelSpec(
        generator_cls=DCGANImprovedGenerator,
        discriminator_cls=DCGANImprovedDiscriminator,
        generator_hparams={"features": 64, "channels": 4},
        discriminator_hparams={"features": 48, "channels": 4},
    ),
    "dcgan_resd": ModelSpec(
        generator_cls=DCGANGenerator,
        discriminator_cls=ResNetDiscriminator64,
        generator_hparams={"features": 64, "channels": 4},
        discriminator_hparams={"features": 64, "channels": 4},
    ),
    "gan_resconv_v1": ModelSpec(
        generator_cls=ResConvGenerator,
        discriminator_cls=DCGANImprovedDiscriminator,
        generator_hparams={
            "features": 64,
            "channels": 4,
            "upsample_mode": "nearest",
            "norm_type": "group",
        },
        discriminator_hparams={"features": 48, "channels": 4},
    ),
}


def list_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def resolve_model(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        known = ", ".join(list_model_names())
        raise ValueError(f"Unknown model_name={model_name!r}. Available: {known}")
    return MODEL_REGISTRY[model_name]
