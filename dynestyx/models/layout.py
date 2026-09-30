"""Utilities for converting fixed array pytrees to flat vector events."""

from dataclasses import dataclass
from math import prod
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_static
@dataclass(frozen=True)
class Layout:
    """A fixed pytree structure and its corresponding flat vector shape.

    Leaf order follows JAX's pytree order. All leaves must be numeric arrays
    with the same dtype. Leading batch axes are preserved by both conversions;
    each scalar leaf contributes one coordinate to the vector.
    """

    treedef: Any
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    dim: int

    @classmethod
    def from_example(cls, example: Any) -> "Layout":
        """Record the tree structure and event shape of each example leaf."""
        leaves, treedef = jax.tree_util.tree_flatten(example)
        cls._validate_leaves(leaves)
        shapes = tuple(tuple(leaf.shape) for leaf in leaves)
        sizes = tuple(prod(shape) for shape in shapes)
        if any(size == 0 for size in sizes):
            raise ValueError("Layout does not support zero-sized leaves.")

        offsets = []
        total = 0
        for size in sizes:
            offsets.append(total)
            total += size
        return cls(treedef, shapes, sizes, tuple(offsets), total)

    @staticmethod
    def _validate_leaves(leaves: list[Any]) -> None:
        if not leaves:
            raise ValueError("Layout requires a nonempty pytree of arrays.")
        for leaf in leaves:
            if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
                raise TypeError("Layout leaves must be numeric arrays.")
            if not jnp.issubdtype(leaf.dtype, jnp.number):
                raise TypeError("Layout leaves must have numeric dtypes.")
        if any(leaf.dtype != leaves[0].dtype for leaf in leaves):
            raise TypeError("Layout leaves must have the same numeric dtype.")

    def flatten(self, value: Any):
        """Flatten event axes, preserving common leading batch axes."""
        leaves, treedef = jax.tree_util.tree_flatten(value)
        if treedef != self.treedef:
            raise ValueError("Pytree structure does not match Layout.")
        self._validate_leaves(leaves)

        batch_shape = None
        flat_leaves = []
        for index, (leaf, shape, size) in enumerate(
            zip(leaves, self.shapes, self.sizes, strict=True)
        ):
            split = leaf.ndim - len(shape)
            if split < 0 or tuple(leaf.shape[split:]) != shape:
                raise ValueError(
                    f"Layout leaf {index} must have trailing shape {shape}; "
                    f"got {leaf.shape}."
                )
            leading = tuple(leaf.shape[:split])
            if batch_shape is not None and leading != batch_shape:
                raise ValueError(
                    "Layout leaves must have identical leading batch axes."
                )
            batch_shape = leading
            flat_leaves.append(jnp.reshape(leaf, (*leading, size)))
        return jnp.concatenate(flat_leaves, axis=-1)

    def unflatten(self, value):
        """Restore a vector's trailing coordinate axis to the recorded pytree."""
        value = jnp.asarray(value)
        if value.ndim == 0 or value.shape[-1] != self.dim:
            raise ValueError(
                f"Expected trailing flat axis {self.dim}; got {value.shape}."
            )
        self._validate_leaves([value])
        leaves = [
            value[..., offset : offset + size].reshape((*value.shape[:-1], *shape))
            for shape, size, offset in zip(
                self.shapes, self.sizes, self.offsets, strict=True
            )
        ]
        return jax.tree_util.tree_unflatten(self.treedef, leaves)


@jax.tree_util.register_static
@dataclass(frozen=True)
class Layouts:
    """Optional layouts for state, control, and observation values."""

    state: Layout | None = None
    control: Layout | None = None
    observation: Layout | None = None

    def __post_init__(self) -> None:
        for name in ("state", "control", "observation"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Layout):
                raise TypeError(f"Layouts.{name} must be a Layout or None.")
