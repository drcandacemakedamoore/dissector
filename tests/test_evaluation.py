"""Tests for the dissector.diffusion module.

Please note some of these tests were developed with the aid of generative AI, but
have been checked by the creators.
"""


import numpy as np
import pytest
from dissector.evaluation import binary_cross_entropy
from dissector.evaluation import compare_folders
from dissector.evaluation import extract_boundary_2d
from dissector.evaluation import extract_boundary_3d
from dissector.evaluation import inter_slice_dice


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

def test_inter_slice_dice():
    """Inter-slice Dice is higher for smooth masks than for discontinuous ones."""
    mask = np.zeros((10, 8, 8), dtype=np.uint8)

    # slices 0-4: identical filled square — perfect continuity
    mask[0:5, 2:6, 2:6] = 1

    # slices 5-9: square shifts by 4 pixels each slice — poor continuity
    mask[5, 2:6, 2:6] = 1
    mask[6, 3:7, 3:7] = 1  # shifted 1
    mask[7, 4:8, 4:8] = 1  # shifted 2, partial overlap
    mask[8, 0:2, 0:2] = 1  # no overlap with slice 7
    mask[9, 6:8, 6:8] = 1  # no overlap with slice 8

    # score over the smooth region (slices 0-4)
    smooth_score = inter_slice_dice(mask[0:5])

    # score over the discontinuous region (slices 5-9)
    discontinuous_score = inter_slice_dice(mask[5:10])

    assert smooth_score == 1.0
    assert discontinuous_score < smooth_score


def test_compare_folders_identical(tmp_path):
    """compare_folders returns True when two folders have identical files."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "file1.txt").write_bytes(b"hello")
    (a / "file2.txt").write_bytes(b"world")
    (b / "file1.txt").write_bytes(b"hello")
    (b / "file2.txt").write_bytes(b"world")
    assert compare_folders(str(a), str(b)) is True


def test_compare_folders_content_differs(tmp_path):
    """compare_folders raises AssertionError when file content differs."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "file.txt").write_bytes(b"hello")
    (b / "file.txt").write_bytes(b"HELLO")
    with pytest.raises(AssertionError, match="hash diff"):
        compare_folders(str(a), str(b))


def test_compare_folders_missing_file(tmp_path):
    """compare_folders raises AssertionError when a file exists only in one folder."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "only_in_a.txt").write_bytes(b"data")
    with pytest.raises(AssertionError, match="only in A"):
        compare_folders(str(a), str(b))


def test_binary_cross_entropy_probalities():
    """BCE on a small binary array with near-perfect predictions."""
    y_true = np.array([1, 0, 1, 0, 1], dtype=float)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    result = binary_cross_entropy(y_true, y_pred)
    expected = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)
    assert result > 0

