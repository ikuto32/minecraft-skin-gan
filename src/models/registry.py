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


@dataclass(frozen=True)
class ModelSpec:
    generator_cls: type
    discriminator_cls: type
    generator_hparams: dict[str, object]
    discriminator_hparams: dict[str, object]


MODEL_REGISTRY: dict[str, ModelSpec] = {
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


def resolve_model(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_name={model_name!r}. Available: {known}")
    return MODEL_REGISTRY[model_name]
