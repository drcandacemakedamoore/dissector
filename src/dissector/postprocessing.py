# Copyright (c) dissector contributors
#
# Portions of this file are derived from the MONAI project:
#   https://github.com/Project-MONAI/MONAI
#
# Original copyright notice (preserved as required by the licence):
#   Copyright (c) MONAI Consortium
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Modifications made relative to the MONAI originals
# (monai/transforms/post/array.py and monai/transforms/utils.py):
#   - Removed all MONAI-specific types (MetaTensor, NdarrayOrTensor, TransformBackends).
#   - Removed GPU / CuPy code paths; CPU-only.
#   - Input and output are plain numpy arrays throughout.
#   - Removed the abstract Transform base class; classes are plain Python objects.
#   - `get_unique_labels` is inlined into KeepLargestConnectedComponent.__call__.
#   - `_get_unique_labels` helper added for the non-onehot case.
#   - `remove_small_objects` reimplemented without MetaTensor pixdim lookup.
#   - Import guards added so that missing scikit-image raises a clear RuntimeError.

"""Post-processing utilities for segmentation masks.

Provides two classes ported from MONAI (Apache 2.0):

* :class:`KeepLargestConnectedComponent` -- for each label, remove all but
  the *N* largest connected blobs.
* :class:`RemoveSmallObjects` -- remove blobs below a minimum voxel count
  (or physical volume when *by_measure* is True).

Both operate on **channel-first numpy arrays** of shape
``(C, H, W[, D])``.  For an ordinary integer label map of shape ``(H, W, D)``
wrap it first::

    seg = seg[np.newaxis, ...]   # -> (1, H, W, D)
    seg = KeepLargestConnectedComponent()(seg)
    seg = seg[0]                 # -> (H, W, D)

Requires ``scikit-image``::

    conda install -c conda-forge scikit-image
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

try:
    from skimage import measure, morphology

    _has_skimage = True
except ImportError:
    _has_skimage = False

__all__ = [
    "get_largest_connected_component_mask",
    "KeepLargestConnectedComponent",
    "RemoveSmallObjects",
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_largest_connected_component_mask(
    img: np.ndarray,
    connectivity: Optional[int] = None,
    num_components: int = 1,
) -> np.ndarray:
    """Return a boolean mask containing only the *num_components* largest
    connected components of *img*.

    Ported from ``monai.transforms.utils.get_largest_connected_component_mask``
    (MONAI, Apache 2.0, Copyright (c) MONAI Consortium).  GPU/CuPy path
    removed; CPU-only.

    Args:
        img: Binary spatial array of shape ``(H[, W, D])``.
        connectivity: Maximum number of orthogonal hops considered as
            neighbours.  ``None`` uses full connectivity for the image rank.
            See :func:`skimage.measure.label`.
        num_components: Number of largest components to keep.

    Returns:
        Boolean array of the same shape as *img*.
    """
    if not _has_skimage:
        raise RuntimeError(
            "scikit-image is required for get_largest_connected_component_mask. "
            "Install with: conda install -c conda-forge scikit-image"
        )

    features, num_features = measure.label(img, connectivity=connectivity, return_num=True)

    if num_features <= num_components:
        return img.astype(bool)

    nonzeros = features[np.nonzero(features)]
    # argsort of bincount gives ascending size; reverse to get largest first
    features_to_keep = np.argsort(np.bincount(nonzeros))[::-1][:num_components]
    return np.isin(features, features_to_keep)


# ---------------------------------------------------------------------------
# KeepLargestConnectedComponent
# ---------------------------------------------------------------------------


class KeepLargestConnectedComponent:
    """Keep only the *N* largest connected component(s) per label.

    Ported from ``monai.transforms.post.array.KeepLargestConnectedComponent``
    (MONAI, Apache 2.0, Copyright (c) MONAI Consortium).
    MONAI-specific types, GPU paths, and the abstract Transform base class
    have been removed; the class now operates on plain numpy arrays.

    The input is expected to be **channel-first**:

    * **Non-one-hot** (typical label map): shape ``(1, H, W[, D])``.
      Voxel values are integer label IDs; 0 is background.
    * **One-hot**: shape ``(C, H, W[, D])``, one channel per class.

    Examples::

        # Label map, keep largest blob per label (independent=True, default)
        transform = KeepLargestConnectedComponent()
        out = transform(seg[np.newaxis])   # seg shape (H,W,D) -> (1,H,W,D)

        # One-hot, process labels 1 and 2 jointly (independent=False)
        transform = KeepLargestConnectedComponent(
            applied_labels=[1, 2], is_onehot=True, independent=False
        )

    Args:
        applied_labels: Label(s) to process.  ``None`` processes all
            non-zero labels.
        is_onehot: ``True`` for one-hot input; ``False`` for label-map input;
            ``None`` (default) auto-detects: multi-channel → one-hot,
            single-channel → label map.
        independent: If ``True`` (default), each label is processed
            independently.  If ``False``, labels are treated as a union
            and a single largest component is kept across all of them.
        connectivity: Neighbourhood connectivity passed to
            :func:`skimage.measure.label`.
        num_components: Number of largest components to preserve per label.
    """

    def __init__(
        self,
        applied_labels: Optional[Union[Sequence[int], int]] = None,
        is_onehot: Optional[bool] = None,
        independent: bool = True,
        connectivity: Optional[int] = None,
        num_components: int = 1,
    ) -> None:
        if applied_labels is not None:
            if isinstance(applied_labels, int):
                applied_labels = (applied_labels,)
            self.applied_labels: Optional[tuple[int, ...]] = tuple(applied_labels)
        else:
            self.applied_labels = None
        self.is_onehot = is_onehot
        self.independent = independent
        self.connectivity = connectivity
        self.num_components = num_components

    def _unique_labels(self, img: np.ndarray, is_onehot: bool) -> tuple[int, ...]:
        if is_onehot:
            return tuple(i for i in range(img.shape[0]) if np.any(img[i] > 0))
        return tuple(int(v) for v in np.unique(img[0]) if v != 0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: Shape ``(C, spatial_dim1[, spatial_dim2, ...])``.

        Returns:
            Array of the same shape and dtype as *img*.
        """
        is_onehot = img.shape[0] > 1 if self.is_onehot is None else self.is_onehot
        applied_labels = (
            self.applied_labels
            if self.applied_labels is not None
            else self._unique_labels(img, is_onehot)
        )

        img = img.copy()

        if self.independent:
            for i in applied_labels:
                foreground = img[i] > 0 if is_onehot else img[0] == i
                mask = get_largest_connected_component_mask(
                    foreground, self.connectivity, self.num_components
                )
                if is_onehot:
                    img[i][foreground != mask] = 0
                else:
                    img[0][foreground != mask] = 0
            return img

        if not is_onehot:
            foreground = np.isin(img[0], applied_labels)
            mask = get_largest_connected_component_mask(
                foreground, self.connectivity, self.num_components
            )
            img[0][foreground != mask] = 0
            return img

        # one-hot, union of labels
        foreground = np.any(img[list(applied_labels), ...] == 1, axis=0)
        mask = get_largest_connected_component_mask(
            foreground, self.connectivity, self.num_components
        )
        for i in applied_labels:
            img[i][foreground != mask] = 0
        return img


# ---------------------------------------------------------------------------
# RemoveSmallObjects
# ---------------------------------------------------------------------------


class RemoveSmallObjects:
    """Remove objects below a minimum size from a segmentation mask.

    Ported from ``monai.transforms.post.array.RemoveSmallObjects`` and the
    underlying ``monai.transforms.utils.remove_small_objects`` helper
    (MONAI, Apache 2.0, Copyright (c) MONAI Consortium).
    MONAI-specific types (MetaTensor, NdarrayOrTensor) and the abstract
    Transform base class have been removed; the class now operates on plain
    numpy arrays.

    Input should be **channel-first** and **one-hotted**
    ``(C, H, W[, D])``.

    Examples::

        # Remove blobs smaller than 100 voxels
        transform = RemoveSmallObjects(min_size=100)
        out = transform(seg_onehot)

        # Remove blobs smaller than 50 mm³ given 1.5×1.5×3 mm voxels
        transform = RemoveSmallObjects(min_size=50, by_measure=True,
                                       pixdim=(1.5, 1.5, 3.0))
        out = transform(seg_onehot)

    Args:
        min_size: Objects strictly smaller than this are removed.  In voxels
            unless *by_measure* is ``True``, in which case it is a physical
            volume in the same units as *pixdim* (e.g. mm³).
        connectivity: Neighbourhood connectivity for connected-component
            labelling (passed to :func:`skimage.morphology.remove_small_objects`).
        independent_channels: If ``True`` (default), each channel is
            processed independently.  If ``False``, the union of all
            non-background voxels is used.
        by_measure: If ``True``, treat *min_size* as a physical volume and
            convert to voxels using *pixdim*.
        pixdim: Voxel dimensions (mm or any consistent unit).  A single
            float is broadcast to all spatial axes.  Required when
            ``by_measure=True`` and the input is not a MONAI MetaTensor.
    """

    def __init__(
        self,
        min_size: int = 64,
        connectivity: int = 1,
        independent_channels: bool = True,
        by_measure: bool = False,
        pixdim: Optional[Union[Sequence[float], float]] = None,
    ) -> None:
        self.min_size = min_size
        self.connectivity = connectivity
        self.independent_channels = independent_channels
        self.by_measure = by_measure
        self.pixdim = pixdim

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: Shape ``(C, spatial_dim1[, spatial_dim2, ...])``.
                Data should be one-hotted.

        Returns:
            Array of the same shape as *img*.
        """
        if not _has_skimage:
            raise RuntimeError(
                "scikit-image is required for RemoveSmallObjects. "
                "Install with: conda install -c conda-forge scikit-image"
            )

        if len(np.unique(img)) == 1:
            return img

        min_size = self.min_size

        if self.by_measure:
            sr = len(img.shape[1:])
            if self.pixdim is not None:
                if isinstance(self.pixdim, (int, float)):
                    _pixdim: tuple[float, ...] = (float(self.pixdim),) * sr
                else:
                    _pixdim = tuple(float(p) for p in self.pixdim)
            else:
                warnings.warn(
                    "`pixdim` is None when `by_measure=True`; assuming 1 mm isotropic voxels."
                )
                _pixdim = (1.0,) * sr
            voxel_volume = float(np.prod(_pixdim))
            if voxel_volume == 0:
                warnings.warn("Invalid `pixdim` (product is 0); treating as 1.")
                voxel_volume = 1.0
            min_size = int(np.ceil(self.min_size / voxel_volume))
        elif self.pixdim is not None:
            warnings.warn("`pixdim` is set but ignored when `by_measure=False`.")

        img_np = img.copy()

        if not self.independent_channels:
            mask = img_np > 0
            cleaned = morphology.remove_small_objects(mask, min_size, self.connectivity)
            return img_np * cleaned

        img_work = img_np.astype(bool if img_np.max() <= 1 else np.int32)
        return morphology.remove_small_objects(img_work, min_size, self.connectivity).astype(
            img_np.dtype
        )
