from __future__ import annotations

from dataclasses import dataclass

from src.models.dcgan import DCGANDiscriminator, DCGANGenerator, DCGANImprovedDiscriminator, DCGANImprovedGenerator


@dataclass(frozen=True)
class ModelSpec:
    generator_cls: type
    discriminator_cls: type
    generator_hparams: dict[str, int]
    discriminator_hparams: dict[str, int]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dcgan_baseline": ModelSpec(
        generator_cls=DCGANGenerator,
        discriminator_cls=DCGANDiscriminator,
        generator_hparams={"features": 64},
        discriminator_hparams={"features": 64},
    ),
    "dcgan_improved": ModelSpec(
        generator_cls=DCGANImprovedGenerator,
        discriminator_cls=DCGANImprovedDiscriminator,
        generator_hparams={"features": 64},
        discriminator_hparams={"features": 48},
    ),
}


def resolve_model(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_name={model_name!r}. Available: {known}")
    return MODEL_REGISTRY[model_name]
