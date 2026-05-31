"""Retinal sampling via ganglion-cell (or cone) density mapping.

The algorithm maps each output pixel to its source location in the input via
inverse mapping through the integrated cell-density function, then applies
precomputed bilinear interpolation.

The three-stage pipeline:

    intermediate = sampler.compress(image)           # full circular mapping
    output       = sampler.crop(intermediate)        # remove empty corners
    preview      = sampler.decompress(intermediate)  # invert for visualisation

Keeping compress, crop, and decompress separate means decompress always
receives the full intermediate (not a crop), so its coordinate map needs no
crop-offset correction and the result is clean at the periphery.

The crop eliminates the empty corners that arise because a circular FOV maps
into a square buffer: the intermediate has size ceil(output_size * sqrt(2)),
whose inscribed square is exactly output_size × output_size.

Interpolation weights and integer source-pixel indices are precomputed once in
build_mapping so each compress/decompress call reduces to four numpy fancy-index
lookups and a channel-wise weighted sum — no scipy in the hot path.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

from .density import (
    ganglion_cumulative, ganglion_cumulative_inv,
    cone_cumulative, cone_cumulative_inv,
)

_SQRT2 = math.sqrt(2.0)

_CellType = Literal['ganglion', 'cone']

_DENSITY_FUNCTIONS: dict[str, tuple[Callable, Callable]] = {
    'ganglion': (ganglion_cumulative, ganglion_cumulative_inv),
    'cone':     (cone_cumulative,     cone_cumulative_inv),
}


class RetinalSampler:
    """Biologically plausible retinal resampling based on cell-density distributions.

    Typical usage::

        sampler = RetinalSampler()
        sampler.build_mapping(input_size=512, output_size=256, fov=20.0)

        intermediate  = sampler.compress(image)           # full circular mapping
        compressed    = sampler.crop(intermediate)        # remove empty corners → 256×256
        reconstructed = sampler.decompress(intermediate)  # invert for visualisation

    ``build_mapping`` is the expensive step; call it once per geometry and reuse
    across a batch.

    Args:
        cell_type: ``'ganglion'`` (default) or ``'cone'``.
    """

    def __init__(self, cell_type: _CellType = 'ganglion') -> None:
        if cell_type not in _DENSITY_FUNCTIONS:
            raise ValueError(f"cell_type must be one of {list(_DENSITY_FUNCTIONS)}")
        self._cumulative, self._cumulative_inv = _DENSITY_FUNCTIONS[cell_type]

        self._input_size: int = 0
        self._output_size: int = 0
        self._compress_intermediate: int = 0
        self._compress_pad: int = 0
        self._decomp_intermediate: int = 0
        self._decomp_pad: int = 0
        # Precomputed bilinear data: (4, N) int32 indices and (4, N) float32 weights
        self._compress_indices: np.ndarray | None = None
        self._compress_weights: np.ndarray | None = None
        self._decomp_indices: np.ndarray | None = None
        self._decomp_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_mapping(
        self,
        input_size: int,
        output_size: int,
        fov: float,
    ) -> None:
        """Pre-compute bilinear indices and weights for compress and decompress.

        Must be called before any of ``compress``, ``crop``, or ``decompress``.
        Can be reused for all images that share the same *input_size*.

        Args:
            input_size:  Side length (pixels) of the square input.  Non-square
                         images are zero-padded to this size inside ``compress``.
            output_size: Side length (pixels) of the cropped output.
            fov:         Total field of view in degrees of visual angle (1–100).
        """
        self._input_size = input_size
        self._output_size = output_size

        eccentricity      = fov / 2.0
        cells_at_edge     = self._cumulative(eccentricity)
        degrees_per_pixel = eccentricity / (input_size / 2.0)

        (self._compress_indices,
         self._compress_weights,
         self._compress_intermediate,
         self._compress_pad) = _build_compress_mapping(
            output_size, cells_at_edge, degrees_per_pixel,
            input_size, self._cumulative_inv,
        )

        (self._decomp_indices,
         self._decomp_weights,
         self._decomp_intermediate,
         self._decomp_pad) = _build_decompress_mapping(
            input_size, eccentricity, cells_at_edge,
            self._compress_intermediate, self._cumulative,
        )

    def compress(self, image: np.ndarray) -> np.ndarray:
        """Map *image* into cell-count space.

        Returns the full intermediate buffer, which has a circular region of
        valid data and zero-filled corners.  Pass the result to ``crop`` for the
        final output, or to ``decompress`` for a visualisation of information loss.

        Args:
            image: H × W or H × W × 3 uint8 array.  The longer side must equal
                   *input_size*; shorter sides are zero-padded to form a square.

        Returns:
            ``compress_intermediate × compress_intermediate × 3`` uint8 array.
        """
        self._require_mapping()
        image = _to_square_rgb(image)
        if image.shape[0] != self._input_size:
            raise ValueError(
                f"Image side {image.shape[0]} px does not match "
                f"input_size={self._input_size} px."
            )
        return _apply_bilinear(
            image, self._compress_indices, self._compress_weights,
            self._compress_intermediate,
        )

    def crop(self, intermediate: np.ndarray) -> np.ndarray:
        """Crop the central ``output_size × output_size`` region from *intermediate*.

        The crop eliminates the zero-filled corners left by the circular FOV
        mapping.  Every pixel in the cropped result lies within the FOV circle.

        Args:
            intermediate: Output of ``compress``.

        Returns:
            ``output_size × output_size × 3`` uint8 array.
        """
        self._require_mapping()
        p = self._compress_pad
        return intermediate[p : p + self._output_size, p : p + self._output_size]

    def decompress(self, intermediate: np.ndarray) -> np.ndarray:
        """Invert *intermediate* back toward the original image space.

        Operates on the full compress intermediate (not the crop) so no
        crop-offset correction is needed and the peripheral reconstruction is
        clean.  Uses the same intermediate→crop pattern as ``compress`` to avoid
        empty corners in the output.

        Args:
            intermediate: Output of ``compress``
                          (``compress_intermediate × compress_intermediate × 3``).

        Returns:
            ``input_size × input_size × 3`` uint8 array.
        """
        self._require_mapping()
        if intermediate.ndim == 2:
            intermediate = np.repeat(intermediate[:, :, None], 3, axis=2)
        expected = (self._compress_intermediate, self._compress_intermediate)
        if intermediate.shape[:2] != expected:
            raise ValueError(
                f"Expected compress intermediate shape {expected}, "
                f"got {intermediate.shape[:2]}. Pass the output of compress(), "
                f"not crop()."
            )
        result = _apply_bilinear(
            intermediate, self._decomp_indices, self._decomp_weights,
            self._decomp_intermediate,
        )
        p = self._decomp_pad
        return result[p : p + self._input_size, p : p + self._input_size]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_mapping(self) -> None:
        if self._compress_indices is None:
            raise RuntimeError(
                "Call build_mapping() before compress(), crop(), or decompress()."
            )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_compress_mapping(
    output_size: int,
    cells_at_edge: float,
    degrees_per_pixel: float,
    input_size: int,
    cumulative_inv: Callable,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute bilinear data for the compress (input → cell-count space) transform."""
    intermediate = math.ceil(output_size * _SQRT2)
    pad = (intermediate - output_size) // 2

    cell_coords = np.linspace(-cells_at_edge, cells_at_edge, intermediate)
    cx, cy = np.meshgrid(cell_coords, cell_coords)
    cx, cy = cx.ravel(), cy.ravel()

    angle        = np.arctan2(cy, cx)
    radius_cells = np.hypot(cx, cy)

    with np.errstate(divide='ignore', invalid='ignore'):
        radius_deg = cumulative_inv(radius_cells)
    radius_pix = np.nan_to_num(
        radius_deg / degrees_per_pixel, nan=0.0, posinf=float(input_size)
    )

    center = input_size / 2.0
    source_row = np.sin(angle) * radius_pix + center
    source_col = np.cos(angle) * radius_pix + center

    indices, weights = _precompute_bilinear(source_row, source_col, input_size)
    return indices, weights, intermediate, pad


def _build_decompress_mapping(
    input_size: int,
    eccentricity: float,
    cells_at_edge: float,
    compress_intermediate: int,
    cumulative: Callable,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Compute bilinear data for the decompress (cell-count space → input) transform.

    Looks up directly in the full compress intermediate (no crop-offset correction).
    """
    intermediate = math.ceil(input_size * _SQRT2)
    pad = (intermediate - input_size) // 2

    # Scale: converts cell count to pixel coordinate in the compress intermediate
    compress_scale  = (compress_intermediate - 1) / (2.0 * cells_at_edge)
    compress_center = (compress_intermediate - 1) / 2.0

    deg_coords = np.linspace(-eccentricity, eccentricity, intermediate)
    dx, dy = np.meshgrid(deg_coords, deg_coords)
    dx, dy = dx.ravel(), dy.ravel()

    angle_d       = np.arctan2(dy, dx)
    radius_deg_d  = np.hypot(dx, dy)
    radius_cells_d = cumulative(radius_deg_d)

    source_row = np.sin(angle_d) * radius_cells_d * compress_scale + compress_center
    source_col = np.cos(angle_d) * radius_cells_d * compress_scale + compress_center

    indices, weights = _precompute_bilinear(source_row, source_col, compress_intermediate)
    return indices, weights, intermediate, pad


def _precompute_bilinear(
    source_row: np.ndarray,
    source_col: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute integer neighbour indices and bilinear weights for a set of source coords.

    Returns:
        indices: (4, N) int32  — r0, r1, c0, c1 rows of neighbour pixel indices.
        weights: (4, N) float32 — corresponding bilinear weights (sum to 1 per column).
    """
    s = image_size - 1
    r0 = np.clip(np.floor(source_row).astype(np.int32), 0, s)
    r1 = np.clip(np.ceil(source_row).astype(np.int32),  0, s)
    c0 = np.clip(np.floor(source_col).astype(np.int32), 0, s)
    c1 = np.clip(np.ceil(source_col).astype(np.int32),  0, s)

    dr = np.clip(source_row - r0, 0.0, 1.0)
    dc = np.clip(source_col - c0, 0.0, 1.0)

    indices = np.stack([r0, r1, c0, c1])                              # (4, N) int32
    weights = np.stack([
        (1.0 - dr) * (1.0 - dc),
        (1.0 - dr) * dc,
        dr          * (1.0 - dc),
        dr          * dc,
    ], dtype=np.float32)                                              # (4, N) float32

    return indices, weights


def _apply_bilinear(
    image: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Apply precomputed bilinear interpolation to all channels simultaneously.

    Args:
        image:       H × W × 3 uint8 source image.
        indices:     (4, N) int32 — r0, r1, c0, c1 neighbour indices.
        weights:     (4, N) float32 — bilinear weights.
        output_size: side length of the square output.

    Returns:
        output_size × output_size × 3 uint8 array.
    """
    r0, r1, c0, c1 = indices
    w = weights[:, :, None]   # (4, N, 1) — broadcast over 3 channels
    result = (
        w[0] * image[r0, c0] +
        w[1] * image[r0, c1] +
        w[2] * image[r1, c0] +
        w[3] * image[r1, c1]
    )
    return result.reshape(output_size, output_size, 3).clip(0, 255).astype(np.uint8)


def _to_square_rgb(image: np.ndarray) -> np.ndarray:
    """Zero-pad to square and ensure 3 channels (no copy if already correct)."""
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    row_off = (size - h) // 2
    col_off = (size - w) // 2
    square[row_off : row_off + h, col_off : col_off + w] = image
    return square
