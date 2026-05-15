"""Tests for the dissector.diffusion module.

Please note some of these tests were developed with the aid of generative AI, but
have been checked by the creators.
"""


import numpy as np
from dissector.evaluation import binary_cross_entropy
from dissector.evaluation import extract_boundary_2d
from dissector.evaluation import extract_boundary_3d


def test_extract_boundary_2d():
    """Example 2d boundary extract."""
    pic = np.zeros((10, 10), dtype=int)
    for z in range(2, 8):
        pic[5, z] = 1
        pic[6, z] = 1
        pic[7, z] = 1
    extracted = extract_boundary_2d(pic)
    total_voxels = pic.sum()
    interior_voxels = 1 * 1 * 4
    expected_boundary_voxels = total_voxels - interior_voxels

    assert sum(sum(extracted)) == expected_boundary_voxels


def test_extract_boundary_3d():
    """Example 3d boundary extract."""
    vol = np.zeros((10, 10, 10), dtype=int)
    for z in range(10):
        vol[5, 5, z] = 1
        vol[5, 6, z] = 1
        vol[5, 7, z] = 1
        vol[6, 5, z] = 1
        vol[6, 6, z] = 1
        vol[6, 7, z] = 1
        vol[7, 5, z] = 1
        vol[7, 6, z] = 1
        vol[7, 7, z] = 1

        extracted3 = extract_boundary_3d(vol)

    total_voxels = vol.sum()
    interior_voxels = 1 * 1 * 8
    expected_boundary_voxels = total_voxels - interior_voxels

    assert sum(sum(sum(extracted3))) == expected_boundary_voxels


def test_binary_cross_entropy():
    """BCE between two partially-overlapping 3D binary segmentation masks."""
    # 6x6x6 grid gives room for a same-size non-overlapping block
    ground_truth = np.zeros((6, 6, 6), dtype=float)
    ground_truth[1:3, 1:3, 1:3] = 1.0  # 2x2x2 = 8 voxels at indices 1-2

    # predicted mask overlaps partially — 1 shared voxel at [2,2,2]
    predicted = np.zeros((6, 6, 6), dtype=float)
    predicted[2:4, 2:4, 2:4] = 1.0  # 8 voxels at indices 2-3

    # mask with no overlap at all — same size, far corner
    zero_overlap_predicted = np.zeros((6, 6, 6), dtype=float)
    zero_overlap_predicted[4:6, 4:6, 4:6] = 1.0  # 8 voxels at indices 4-5

    result = binary_cross_entropy(ground_truth, predicted)
    zero_overlap_result = binary_cross_entropy(ground_truth, zero_overlap_predicted)
    assert result > 0
    # perfect prediction should yield lower BCE than an imperfect one
    perfect_result = binary_cross_entropy(ground_truth, ground_truth)
    assert perfect_result < result
    assert result < zero_overlap_result

def test_binary_cross_entropy_probalities():
    """BCE on a small binary array with near-perfect predictions."""
    y_true = np.array([1, 0, 1, 0, 1], dtype=float)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    result = binary_cross_entropy(y_true, y_pred)
    expected = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)
    assert result > 0

