"""Documentation about the dissector module."""

from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening
from scipy.ndimage import generic_filter
from scipy.ndimage import label as ndlabel


def estimate_dixon_noise(
    water: np.ndarray,
    fat: np.ndarray,
    background_percentile: float = 10.0,
    foreground_percentile: float = 50.0,
) -> dict:
    """Estimate noise in Dixon water and fat MRI sequences.

    This function is experimental andd currently dysfunctional.
    The idea was that background voxels — where both water and fat signal fall below
    ``background_percentile`` — contain no true anatomical signal and
    therefore reflect pure noise.  Their standard deviation gives the
    noise floor for each channel.  SNR is the ratio of the median
    foreground signal to that noise floor.

    For a well-behaved Dixon acquisition the water and fat noise floors
    should be similar (``noise_ratio`` ≈ 1).  A large deviation flags
    reconstruction artefacts or mismatched dynamic ranges.

    Args:
        water: 3-D array of water-channel signal (slices, H, W).
        fat: 3-D array of fat-channel signal, same shape as *water*.
        background_percentile: signal percentile below which a voxel is
            treated as background in both channels (default 10).
        foreground_percentile: signal percentile used as a representative
            foreground value for SNR (default 50 = median foreground).

    Returns:
        dict with keys:

        * ``water_noise_std``    - noise std in the water channel
        * ``fat_noise_std``      - noise std in the fat channel
        * ``water_snr``          - SNR of the water channel
        * ``fat_snr``            - SNR of the fat channel
        * ``noise_ratio``        - water_noise_std / fat_noise_std
        * ``n_background_voxels``- number of voxels used for estimation

    Raises:
        ValueError: if *water* and *fat* have different shapes, or if no
            background voxels are found.
    """
    water = np.asarray(water, dtype=float)
    fat   = np.asarray(fat,   dtype=float)

    if water.shape != fat.shape:
        msg = f"water and fat must have the same shape, got {water.shape} vs {fat.shape}"
        raise ValueError(msg)

    w_thresh = np.percentile(water, background_percentile)
    f_thresh = np.percentile(fat,   background_percentile)
    bg_mask  = (water <= w_thresh) & (fat <= f_thresh)

    n_bg = int(bg_mask.sum())
    if n_bg == 0:
        msg = "No background voxels found; lower background_percentile."
        raise ValueError(msg)

    water_noise = float(np.std(water[bg_mask]))
    fat_noise   = float(np.std(fat[bg_mask]))

    fg_mask     = ~bg_mask
    water_fg    = water[fg_mask]
    fat_fg      = fat[fg_mask]

    water_signal = float(np.percentile(water_fg, foreground_percentile)) if water_fg.size else 0.0
    fat_signal   = float(np.percentile(fat_fg,   foreground_percentile)) if fat_fg.size   else 0.0

    water_snr = water_signal / water_noise if water_noise > 0 else float("inf")
    fat_snr   = fat_signal   / fat_noise   if fat_noise   > 0 else float("inf")
    ratio     = water_noise  / fat_noise   if fat_noise   > 0 else float("inf")

    return {
        "water_noise_std":     water_noise,
        "fat_noise_std":       fat_noise,
        "water_snr":           water_snr,
        "fat_snr":             fat_snr,
        "noise_ratio":         ratio,
        "n_background_voxels": n_bg,
    }


def local_std_map(volume: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Compute a local noise map as the rolling standard deviation.

    Each voxel in the output is the standard deviation of a cubic
    neighbourhood of side ``kernel_size`` in the input *volume*.  High
    values indicate spatially varying noise or artefacts.

    Args:
        volume: 3-D signal array (slices, H, W).
        kernel_size: side length of the cubic neighbourhood (default 5).

    Returns:
        Array of the same shape as *volume* containing local noise estimates.
    """
    volume = np.asarray(volume, dtype=float)
    return generic_filter(volume, np.std, size=kernel_size)


def find_body_oval(
    slice_2d: np.ndarray,
    kernel_size: int = 5,
    edge_percentile: float = 75.0,
    closing_iterations: int = 4,
    opening_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the body cross-section in a 2-D MRI slice.

    Uses local standard deviation as an edge detector, closes gaps in the
    detected edge ring, fills the interior, then discards any region that
    touches the image border (noise, table, coil artefacts).  The largest
    remaining filled region is the body.  Its boundary is traced directly
    from the local-std edges — the returned contour follows the actual
    tissue boundary, not a fitted geometric shape.

    Rules applied:
    * Any connected region that touches the image border is rejected.
    * The contour is guaranteed to lie strictly inside the image — it
      never reaches the border pixels.

    Args:
        slice_2d: 2-D array (H, W), e.g. one fat-fraction slice.
        kernel_size: side length of the local-std neighbourhood (default 5).
        edge_percentile: percentile of the std map used as the binarisation
            threshold (default 75).
        closing_iterations: morphological closing iterations to bridge small
            gaps in the edge ring (default 4).
        opening_iterations: morphological opening iterations applied to the
            selected body region to remove thin protrusions before the contour
            is traced (default 2).

    Returns:
        body_mask: boolean array (H, W), True inside the body region.
        contour: boolean array (H, W), True on the boundary of body_mask.
            Empty (all False) if no body region is found.
    """
    arr = np.asarray(slice_2d, dtype=float)
    h, w = arr.shape

    # edge map from local std
    std = generic_filter(arr, np.std, size=kernel_size)

    # binarise, close gaps, fill interior
    edges  = std >= np.percentile(std, edge_percentile)
    closed = binary_closing(edges, iterations=closing_iterations)
    filled = binary_fill_holes(closed)

    # label connected components
    labeled, n = ndlabel(filled)

    # discard any component touching the image border
    border_labels: set[int] = set()
    for row_or_col in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(row_or_col.tolist())
    border_labels.discard(0)

    # largest surviving component = body
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

    # remove thin protrusions so the contour follows smooth tissue boundaries
    if opening_iterations > 0:
        body_mask = binary_opening(body_mask, iterations=opening_iterations)

    # contour = body boundary traced from the local-std edges:
    # pixels that are inside the body but lost when the mask is eroded by 1
    contour = body_mask & ~binary_erosion(body_mask)

    return body_mask, contour


# these lines are total junk
def hello(name: str) -> str:
    """Say hello.

    Function docstring using Google docstring style. Will be upgraded to sphinx style soon.

    Args:
        name (str): Name to say hello to

    Returns:
        str: Hello message

    Raises:
        ValueError: If `name` is equal to `nobody`

    Example:
        This function can be called with `Jane Smith` as argument using

        >>> from dissector.diffusion import hello
        >>> hello('Jane Smith')
        'Hello Jane Smith!'

    """
    if name == "nobody":
        msg = "Can not say hello to nobody"
        raise ValueError(msg)

    return f"Hello {name}!"
