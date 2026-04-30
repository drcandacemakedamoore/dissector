import SimpleITK as sitk
import numpy as np 


# # Read 3D images (e.g., .nii, .mha, .nrrd)
# fixed = sitk.ReadImage("fixed.nii", sitk.sitkFloat32)
# moving = sitk.ReadImage("moving.nii", sitk.sitkFloat32)

# # Initialize transform (important!)
# initial_transform = sitk.CenteredTransformInitializer(
#     fixed,
#     moving,
#     sitk.Euler3DTransform(),
#     sitk.CenteredTransformInitializerFilter.GEOMETRY
# )

# # Registration setup
# registration_method = sitk.ImageRegistrationMethod()

# # Similarity metric (great for multi-modal)
# registration_method.SetMetricAsMattesMutualInformation(50)

# # Sampling (speeds things up for large volumes)
# registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# registration_method.SetMetricSamplingPercentage(0.01)

# # Interpolator
# registration_method.SetInterpolator(sitk.sitkLinear)

# # Optimizer
# registration_method.SetOptimizerAsGradientDescent(
#     learningRate=1.0,
#     numberOfIterations=200,
#     convergenceMinimumValue=1e-6,
#     convergenceWindowSize=10
# )

# registration_method.SetOptimizerScalesFromPhysicalShift()

# # Multi-resolution (VERY important for 3D)
# registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
# registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
# registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# # Set initial transform
# registration_method.SetInitialTransform(initial_transform, inPlace=False)

# # Run registration
# final_transform = registration_method.Execute(fixed, moving)

# print("Final metric value:", registration_method.GetMetricValue())
# print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())

# # Apply transform
# resampled = sitk.Resample(
#     moving,
#     fixed,
#     final_transform,
#     sitk.sitkLinear,
#     0.0,
#     moving.GetPixelID()
# )

# # Save result
# sitk.WriteImage(resampled, "aligned.nii")

# boundaru iou

import numpy as np

def extract_boundary_2d(mask: np.ndarray) -> np.ndarray:
    """Get boundary pixels of a binary mask on 2d image."""
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue

            # Check 4-neighborhood
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= h or nj < 0 or nj >= w or not mask[ni, nj]:
                    boundary[i, j] = True
                    break

    return boundary

import numpy as np

def extract_boundary_3d(mask: np.ndarray) -> np.ndarray:
    """Get boundary voxels of a 3D binary mask."""
    d, h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)

    # 6-neighborhood (faces only)
    neighbors = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
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
    """Simple dilation using NumPy only on a 3D matrix"""
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


def boundary_overlap_2d(distance: int, mask_a: np.ndarray, mask_b: np.ndarray):
    """
    Boundary overlap for binary segmentation masks using NumPy only.
    """

    # extract boundaries
    b_a = extract_boundary(mask_a)
    b_b = extract_boundary(mask_b)

    # dilate boundaries (tolerance)
    d_a = dilate(b_a, distance)
    d_b = dilate(b_b, distance)

    # Overlap
    overlap = np.logical_and(d_a, d_b).sum()

    # Normalize (average boundary size)
    size_a = b_a.sum()
    size_b = b_b.sum()
    denom = (size_a + size_b) / 2

    if denom == 0:
        return 0.0

    return overlap / denom
# single row visualization function

def single_row_viz(metrics, dataset, row, title):

    """
    This visualizes a single row of metrics on a radar plot. metrics is a list, dataset is the dataframe, row is the row number
    """
    
    row = dataset.iloc[row]
    values = row[metrics].values.astype(float)
    values = np.append(values, values[0])
    # Angles based on number of metrics
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    angles = np.append(angles, angles[0])

    # auto-generate readable labels
    labels = [m.replace("_", " ").strip(":") for m in metrics]
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()
#single_row_viz(["L_gracilis_jaccard","one_minus_falseNegative","one_minus_falsePositive"], thigh_left_grac_dataset, 0, "Visualize example")
