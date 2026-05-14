from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    for z in range(d):
        for i in range(h):
            for j in range(w):
                if not mask[z, i, j]:
                    continue

                for dz, di, dj in neighbors:
                    nz, ni, nj = z + dz, i + di, j + dj

                    if nz < 0 or nz >= d or ni < 0 or ni >= h or nj < 0 or nj >= w or not mask[nz, ni, nj]:
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
                            if 0 <= ni < h and 0 <= nj < w and 0 <= nk < d:
                                dilated[ni, nj, nk] = True

    return dilated


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross entropy between two binary segmentation masks.

    Clips y_pred to [eps, 1-eps] so that exact 0/1 voxels don't produce log(0).
    """
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))


def boundary_iou_3d(distance: int, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Boundary iou.

    From metric definition same as boundary_iou_2d but done in 3d.
    """
    b_a = extract_boundary_3d(mask_a)
    b_b = extract_boundary_3d(mask_b)
    d_a = dilate_3d(b_a, distance)
    d_b = dilate_3d(b_b, distance)
    intersection = np.logical_and(d_a, d_b).sum()
    union = d_a.sum() + d_b.sum() - intersection

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
    union = d_a.sum() + d_b.sum() - intersection

    if union == 0:
        return 0.0

    return intersection / union


def single_row_viz(metrics: list, dataset: pd.dataframe, row: int, title: str) -> None:
    """Graph a single image.

    This visualizes a single row of metrics on a radar plot. metrics is a
    list, dataset is the dataframe, row is the row number.
    """
    row = dataset.iloc[row]
    values = row[metrics].to_numpy(dtype=float, copy=True)  # .astype(float)
    values = np.append(values, values[0])
    # Angles based on number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
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
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.append(angles, angles[0])

    # Create figure + polar axis
    _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    # Plot multiple samples
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        values = row[metrics].to_numpy(dtype=float, copy=True)
        values = np.append(values, values[0])
        ax.plot(angles, values, color="blue", alpha=0.4)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()


def side_by_side_comp(
    datasets: list,
    metrics: list,
    titles: list,
    colors: list,
    labels: list,
) -> None:
    """Two radarplots.

    Warning, titles and colors must go in order of dataset1, dataset2
    """
    mpl.rcParams.update(
        {
            # --- CRITICAL: makes SVG self-contained ---
            "svg.fonttype": "none",  # keep text as text (NOT paths)
            "text.usetex": False,  # NEVER use LaTeX rendering
            "font.family": "DejaVu Sans",  # safe default font
            # --- avoids weird Inkscape flow elements ---
            "path.simplify": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metrics = ["L_gracilis_lower_dice:", "L_gracilis_jaccard", "one_minus_falseNegative", "one_minus_falsePositive"]
    title1 = titles[0]
    title2 = titles[1]
    color1 = colors[0]
    color2 = colors[1]
    dataset1 = datasets[0]
    dataset2 = datasets[1]
    # Angles based on number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.append(angles, angles[0])

    # Create ONE figure with TWO polar subplots
    _, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"polar": True})

    ax = axes[0]
    for i in range(len(dataset1)):
        row = dataset1.iloc[i]
        values = row[metrics].to_numpy(dtype=float, copy=True)
        values = np.append(values, values[0])
        ax.plot(angles, values, color=color1, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_y(label.get_position()[1] - 0.08)
    ax.set_ylim(0, 1)
    ax.set_title(title1)

    # ---- RIGHT: ----
    ax = axes[1]
    for i in range(len(dataset2)):
        row = dataset2.iloc[i]
        values = row[metrics].to_numpy(dtype=float, copy=True)
        values = np.append(values, values[0])
        ax.plot(angles, values, color=color2, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_y(label.get_position()[1] - 0.08)
    ax.set_ylim(0, 1)
    ax.set_title(title2)
    plt.tight_layout()
    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig("comprison_plot.png", dpi=300, bbox_inches="tight", transparent=False)
    plt.show()


def violin_compare(
    csv_list_a: list[str],
    csv_list_b: list[str],
    label_a: str = "Algorithm A",
    label_b: str = "Algorithm B",
    metrics: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Violin plots comparing two sets of segmentation results across CSV files.

    Each CSV list comes from one segmentation algorithm. Shared numeric columns
    (or an explicit metrics list) are plotted side by side as paired violins.
    """

    def _load(csv_list: list[str]) -> pd.DataFrame:
        frames = [pd.read_csv(p, index_col=0) for p in csv_list]
        return pd.concat(frames, ignore_index=True)

    df_a = _load(csv_list_a)
    df_b = _load(csv_list_b)

    if metrics is None:
        metrics = [
            c for c in df_a.columns
            if c in df_b.columns
            and pd.api.types.is_numeric_dtype(df_a[c])
            and pd.api.types.is_numeric_dtype(df_b[c])
        ]

    if not metrics:
        raise ValueError("No shared numeric columns found between the two CSV sets.")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "tomato"]

    for ax, metric in zip(axes, metrics):
        data_a = df_a[metric].dropna().to_numpy()
        data_b = df_b[metric].dropna().to_numpy()

        parts = ax.violinplot([data_a, data_b], positions=[1, 2], showmedians=True)
        for patch, color in zip(parts["bodies"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([label_a, label_b], rotation=15, ha="right")
        ax.set_title(metric.replace("_", " ").strip(":"), fontsize=9)

    fig.suptitle(f"{label_a} vs {label_b}", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
