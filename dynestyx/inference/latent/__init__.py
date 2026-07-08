"""Latent-path inference entry points."""

from dynestyx.inference.latent.builder import LatentPathBuilder
from dynestyx.inference.latent.parameterization import prepare_latent_path_layout

__all__ = ["LatentPathBuilder", "prepare_latent_path_layout"]
