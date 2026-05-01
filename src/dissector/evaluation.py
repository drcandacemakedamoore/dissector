from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def extract_boundary_2d(mask: np.ndarray) -> np.ndarray:
    """Get boundary pixels of a binary mask on 2d image."""
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue

            # check 4-neighborhood
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= h or nj < 0 or nj >= w or not mask[ni, nj]:
                    boundary[i, j] = True
                    break

    return boundary


def extract_boundary_3d(mask: np.ndarray) -> np.ndarray:
    """Get boundary voxels of a 3D binary mask."""
    d, h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    # 6-neighborhood (faces only)
    neighbors = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    for z in range(d):
        for i in range(h):
            for j in range(w):
                if not mask[z, i, j]:
                    continue

                for dz, di, dj in neighbors:
                    nz, ni, nj = z + dz, i + di, j + dj

                    if (
                        nz < 0 or nz >= d or
                        ni < 0 or ni >= h or
                        nj < 0 or nj >= w or
                        not mask[nz, ni, nj]
                    ):
                        boundary[z, i, j] = True
                        break

    return boundary


def dilate_2d(mask: np.ndarray, distance: int) -> np.ndarray:
    """Simple square dilation on a 2D mask using NumPy only."""
    h, w = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)

    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue

            for di in range(-distance, distance + 1):
                for dj in range(-distance, distance + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        dilated[ni, nj] = True

    return dilated


def dilate_3d(mask: np.ndarray, distance: int) -> np.ndarray:
    """Simple dilation using NumPy only on a 3D matrix."""
    h, w, d = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)

    for i in range(h):
        for j in range(w):
            for k in range(d):
                if not mask[i, j, k]:
                    continue

                for di in range(-distance, distance + 1):
                    for dj in range(-distance, distance + 1):
                        for dk in range(-distance, distance + 1):
                            ni, nj, nk = i + di, j + dj, k + dk
                            if (
                                0 <= ni < h and
                                0 <= nj < w and
                                0 <= nk < d
                            ):
                                dilated[ni, nj, nk] = True

    return dilated


def boundary_iou_3d(distance: int, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Boundary iou.

    From metric definition same as boundary_iou_2d but done in 3d.
    """
    b_a = extract_boundary_3d(mask_a)
    b_b = extract_boundary_3d(mask_b)
    d_a = dilate_3d(b_a, distance)
    d_b = dilate_3d(b_b, distance)
    intersection = np.logical_and(d_a, d_b).sum()
    union =  d_a.sum() + d_b.sum() - intersection

    if union == 0:
        return 0.0

    return intersection / union

def boundary_iou_2d(distance: int, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Extract boundary iou.

    According to
    formula from https://metrics-reloaded.dkfz.de/metric-library/
    boundary_intersection_over_union_local definition.
    """
    b_a = extract_boundary_2d(mask_a)
    b_b = extract_boundary_2d(mask_b)

    d_a = dilate_2d(b_a, distance)
    d_b = dilate_2d(b_b, distance)

    intersection = np.logical_and(d_a, d_b).sum()
    union =  d_a.sum() + d_b.sum() - intersection

    if union == 0:
        return 0.0

    return intersection / union


def single_row_viz(metrics: list, dataset: pd.dataframe, row: int, title: str ) -> None:
    """Graph a single image.

    This visualizes a single row of metrics on a radar plot. metrics is a
    list, dataset is the dataframe, row is the row number.
    """
    row = dataset.iloc[row]
    values = row[metrics].to_numpy(dtype=float, copy=True)#.astype(float)
    values = np.append(values, values[0])
    # Angles based on number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    angles = np.append(angles, angles[0])

    # auto-generate readable labels
    labels = [m.replace("_", " ").strip(":") for m in metrics]
    # Plot
    _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()


def radarplot_single_dataset(
    metrics: list[str],
    dataset: pd.DataFrame,
    title: str,
    labels: list[str] | None = None,
) -> None:
    """Plot a radar chart for all rows in a dataset.

    If labels is None, metrics are used as axis labels.
    """
    if labels is None:
        labels = metrics
    # Create angles
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    angles = np.append(angles, angles[0])

    # Create figure + polar axis
    _, ax = plt.subplots(figsize=(6,6), subplot_kw={"polar": True})

    # Plot multiple samples
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        values = row[metrics].to_numpy(dtype=float, copy=True)#.astype(float)
        values = np.append(values, values[0])
        ax.plot(angles, values, color="blue", alpha=0.4)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()
