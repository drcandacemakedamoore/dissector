"""Body oval detection for MRI slices.

Copied from dissector.diffusion so onzediff is self-contained on Lambda.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening
from scipy.ndimage import generic_filter
from scipy.ndimage import label as ndlabel


def find_body_oval(
    slice_2d: np.ndarray,
    kernel_size: int = 5,
    edge_percentile: float = 75.0,
    closing_iterations: int = 4,
    opening_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the body cross-section in a 2-D MRI slice.

    Uses local standard deviation as an edge detector, closes gaps in the
    detected edge ring, fills the interior, then discards any region touching
    the image border.  The largest surviving filled region is the body.

    Returns:
        body_mask: boolean (H, W), True inside body.
        contour:   boolean (H, W), True on boundary of body_mask.
    """
    arr = np.asarray(slice_2d, dtype=float)
    h, w = arr.shape

    std    = generic_filter(arr, np.std, size=kernel_size)
    edges  = std >= np.percentile(std, edge_percentile)
    closed = binary_closing(edges, iterations=closing_iterations)
    filled = binary_fill_holes(closed)

    labeled, n = ndlabel(filled)

    border_labels: set[int] = set()
    for strip in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(strip.tolist())
    border_labels.discard(0)

    best_lbl, best_size = 0, 0
    for lbl in range(1, n + 1):
        if lbl in border_labels:
            continue
        size = int((labeled == lbl).sum())
        if size > best_size:
            best_size, best_lbl = size, lbl

    empty = np.zeros((h, w), dtype=bool)
    if best_lbl == 0:
        return empty, empty

    body_mask = labeled == best_lbl
    if opening_iterations > 0:
        body_mask = binary_opening(body_mask, iterations=opening_iterations)

    contour = body_mask & ~binary_erosion(body_mask)
    return body_mask, contour


def get_body_mask(slice_2d: np.ndarray) -> np.ndarray:
    """Return a body mask for slice_2d, falling back to all-True if detection fails."""
    body_mask, _ = find_body_oval(slice_2d)
    if not body_mask.any():
        return np.ones(slice_2d.shape, dtype=bool)
    return body_mask
