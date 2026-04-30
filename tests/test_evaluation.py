"""Tests for the dissector.diffusion module."""

import numpy as np
import matplotlib.pyplot as plt
from dissector.evaluation import extract_boundary_2d, extract_boundary_3d


def test_extract_boundary_2d():
    """Example 2d boundary extract."""
    pic = np.zeros((10,10), dtype=int)
    for z in range(2,8):
        pic[5, z] = 1
        pic[6, z] = 1
        pic[7, z] = 1
    extracted = extract_boundary_2d(pic)
    
    total_voxels = pic.sum()
    print("total voxels", total_voxels)
    interior_voxels = 1*1*4
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
    # print("total voxels", total_voxels)
    interior_voxels = 1*1*8 
    expected_boundary_voxels = total_voxels - interior_voxels
        

    assert sum(sum(sum(extracted3))) == expected_boundary_voxels

