"""Static layouts connecting fixed array pytrees to flat vector events."""

from dataclasses import dataclass
from math import prod
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_static
@dataclass(frozen=True)
class Layout:
    """Bijection between a fixed array pytree and one trailing vector axis.

    Construct with :meth:`from_example`. Leaves must have a common numeric
    dtype; flattening never silently promotes mixed leaf dtypes. The layout
    retains shapes, not values or dtype, and preserves common leading batch
    axes. Leaf order is JAX's pytree order (including sorted dictionary keys).
    A scalar array leaf contributes one coordinate to the flat vector.
    """

    treedef: Any
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]
    state_dim: int

    @classmethod
    def from_example(cls, example: Any) -> "Layout":
        leaves, treedef = jax.tree_util.tree_flatten(example)
        cls._validate_leaves(leaves)
        shapes = tuple(tuple(x.shape) for x in leaves)
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
    def _validate_leaves(leaves):
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
        """Flatten leaf event shapes, preserving identical leading batch axes."""
        leaves, treedef = jax.tree_util.tree_flatten(value)
        if treedef != self.treedef:
            raise ValueError("Pytree structure does not match Layout.")
        self._validate_leaves(leaves)
        batch = None
        flat = []
        for i, (leaf, shape, size) in enumerate(
            zip(leaves, self.shapes, self.sizes, strict=True)
        ):
            ndim = len(shape)
            split = leaf.ndim - ndim
            if split < 0 or tuple(leaf.shape[split:]) != shape:
                raise ValueError(
                    f"Layout leaf {i} must have trailing shape {shape}; got {leaf.shape}."
                )
            leading = tuple(leaf.shape[:split])
            if batch is not None and batch != leading:
                raise ValueError(
                    "Layout leaves must have identical leading batch axes."
                )
            batch = leading
            flat.append(jnp.reshape(leaf, (*leading, size)))
        return jnp.concatenate(flat, axis=-1)

    def unflatten(self, value):
        """Restore the pytree from an array ending in ``state_dim``."""
        value = jnp.asarray(value)
        if value.ndim == 0 or value.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected trailing flat axis {self.state_dim}; got {value.shape}."
            )
        self._validate_leaves([value])
        leaves = [
            value[..., offset : offset + size].reshape((*value.shape[:-1], *shape))
            for shape, size, offset in zip(
                self.shapes, self.sizes, self.offsets, strict=True
            )
        ]
        return jax.tree_util.tree_unflatten(self.treedef, leaves)
