# creation.py — synthesise and augment Dixon MRI data using TorchIO
from __future__ import annotations
import logging
import pathlib
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torchio as tio
from scipy.ndimage import gaussian_filter

_log = logging.getLogger(__name__)

_AUGMENT = tio.Compose([
    tio.RandomFlip(axes=("LR",)),
    tio.RandomAffine(scales=0.1, degrees=5),
    tio.RandomElasticDeformation(num_control_points=7),
    tio.RandomBiasField(coefficients=0.3),
    tio.RandomNoise(std=(0, 0.05)),
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),
])


def generate_augmented(  # noqa: PLR0913
    water_path: str | pathlib.Path,
    fat_path: str | pathlib.Path,
    n: int,
    output_dir: str | pathlib.Path,
    stem: str | None = None,
    seg_path: str | pathlib.Path | None = None,
) -> list[tuple[pathlib.Path, pathlib.Path, pathlib.Path | None]]:
    """Generate *n* augmented Dixon MRI pairs from one real scan pair.

    Spatial augmentations (flip, affine, elastic deformation) are applied
    jointly to the images and — when *seg_path* is provided — to the
    segmentation mask, so they remain perfectly aligned.  Intensity
    augmentations (bias field, noise, gamma) are applied only to the images
    because TorchIO automatically skips them for ``LabelMap`` channels.

    Args:
        water_path:  Path to the water NIfTI file (.nii or .nii.gz).
        fat_path:    Path to the fat-fraction NIfTI file.
        n:           Number of synthetic output pairs to generate.
        output_dir:  Directory where outputs are saved (created if absent).
        stem:        Base name for output files. Defaults to the water
                     filename stem (without extension).
        seg_path:    Optional path to a segmentation mask (.nii, .nii.gz,
                     or .mha).  Receives the same spatial transforms as the
                     images and is saved as ``{stem}_augmented{i:03d}_seg.nii.gz``.

    Returns:
        List of (water_out, fat_out, seg_out) path triples.
        *seg_out* is ``None`` when *seg_path* is not supplied.
    """
    water_path = pathlib.Path(water_path)
    fat_path   = pathlib.Path(fat_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = water_path.name.replace(".nii.gz", "").replace(".nii", "")

    subject_kwargs: dict = {
        "water": tio.ScalarImage(water_path),
        "fat":   tio.ScalarImage(fat_path),
    }
    if seg_path is not None:
        subject_kwargs["seg"] = tio.LabelMap(pathlib.Path(seg_path))

    subject = tio.Subject(**subject_kwargs)

    outputs = []
    for i in range(n):
        augmented = _AUGMENT(subject)
        water_out = output_dir / f"{stem}_augmented{i:03d}_water.nii.gz"
        fat_out   = output_dir / f"{stem}_augmented{i:03d}_fat.nii.gz"
        augmented["water"].save(water_out)
        augmented["fat"].save(fat_out)

        seg_out = None
        if seg_path is not None:
            seg_out = output_dir / f"{stem}_augmented{i:03d}_seg.nii.gz"
            augmented["seg"].save(seg_out)

        outputs.append((water_out, fat_out, seg_out))
        _log.info("[%d/%d] saved %s", i + 1, n, water_out.name)

    return outputs


# ── Intensity priors for SynthSeg-style synthesis ─────────────────────────────
# Each label maps to (water_mu_lo, water_mu_hi, water_sig, ff_mu_lo, ff_mu_hi, ff_sig).
# All values are normalised [0, 1] intensity space.
# Labels follow the myosegmenTUM combined_gt convention:
#   0=background, 1=L_Gracilis, 2=L_Hamstrings, 3=L_Quadriceps,
#   4=L_Sartorius, 5=R_Gracilis, 6=R_Hamstrings, 7=R_Quadriceps, 8=R_Sartorius
_LABEL_PRIORS: dict[int, tuple[float, float, float, float, float, float]] = {
    0: (0.05, 0.60, 0.08, 0.10, 0.80, 0.10),  # background: fat, bone, air mixed
    1: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # L_Gracilis
    2: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # L_Hamstrings
    3: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # L_Quadriceps
    4: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # L_Sartorius
    5: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # R_Gracilis
    6: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # R_Hamstrings
    7: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # R_Quadriceps
    8: (0.40, 0.75, 0.05, 0.03, 0.20, 0.03),  # R_Sartorius
}


def _random_bias_field(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative bias field simulating MRI inhomogeneity."""
    order = 3
    coeffs = rng.standard_normal((order + 1,) * len(shape)) * 0.1
    field  = np.zeros(shape, dtype=np.float32)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grids  = np.meshgrid(*coords, indexing="ij")
    for idx in np.ndindex(coeffs.shape):
        term = float(coeffs[idx])
        for dim, i in enumerate(idx):
            term = term * grids[dim] ** i
        field += term
    return np.exp(field).astype(np.float32)


def _muscle_texture(shape: tuple[int, ...], rng: np.random.Generator, strength: float = 0.06) -> np.ndarray:
    """Anisotropic fiber-like texture for muscle tissue.

    Generates noise at three scales and blends them, with one scale heavily
    elongated along a random axis to mimic myofibril bundles.  The result
    is zero-mean and scaled so it can be added directly to a [0,1] volume.
    """
    fiber_axis = rng.integers(0, 3)
    sigmas: dict[str, list[float]] = {
        "fine":   [0.5, 0.5, 0.5],
        "fiber":  [0.5, 0.5, 0.5],
        "coarse": [2.0, 2.0, 2.0],
    }
    sigmas["fiber"][fiber_axis] = rng.uniform(4.0, 8.0)

    texture = np.zeros(shape, dtype=np.float32)
    weights = {"fine": 0.3, "fiber": 0.5, "coarse": 0.2}
    for key, sigma in sigmas.items():
        noise = rng.standard_normal(shape).astype(np.float32)
        texture += weights[key] * gaussian_filter(noise, sigma=sigma)

    texture -= texture.mean()
    std = texture.std()
    if std > 0:
        texture = texture / std * strength
    return texture


def _synthesise_volume(
    seg: np.ndarray,
    channel: str,
    rng: np.random.Generator,
    smoothing_sigma: float = 0.8,
) -> np.ndarray:
    """Build one synthetic MRI volume from a label map."""
    vol = np.zeros(seg.shape, dtype=np.float32)
    muscle_labels = set(_LABEL_PRIORS) - {0}

    for label, (wm_lo, wm_hi, w_sig, fm_lo, fm_hi, f_sig) in _LABEL_PRIORS.items():
        mask = seg == label
        if not mask.any():
            continue
        mu  = rng.uniform(wm_lo, wm_hi) if channel == "water" else rng.uniform(fm_lo, fm_hi)
        sig = w_sig if channel == "water" else f_sig
        vol[mask] = rng.normal(mu, sig, size=int(mask.sum())).clip(0, 1)

        if label in muscle_labels:
            texture = _muscle_texture(seg.shape, rng)
            vol[mask] += texture[mask]

    vol = gaussian_filter(vol, sigma=smoothing_sigma).astype(np.float32)
    vol *= _random_bias_field(vol.shape, rng=rng)
    return np.clip(vol, 0, 1)


def generate_synthetic(
    seg_path: str | pathlib.Path,
    n: int,
    output_dir: str | pathlib.Path,
    stem: str | None = None,
    seed: int | None = None,
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Generate *n* truly synthetic Dixon MRI pairs from a segmentation mask.

    Uses a SynthSeg-inspired approach: each tissue label in the mask is
    assigned randomly sampled Gaussian intensity statistics, then the volume
    is smoothed (partial-volume simulation) and corrupted with a random
    multiplicative bias field.  Every call should produce genuinely different-looking
    images because intensities are re-sampled from per-label priors — no real
    image pixels are reused.

    Args:
        seg_path:   Path to a segmentation mask (.nii, .nii.gz, or .mha).
                    Use combined_gt_stack*.mha from myosegmenTUM for thigh data.
        n:          Number of synthetic pairs to generate.
        output_dir: Directory where outputs are saved (created if absent).
        stem:       Base name for output files.  Defaults to the seg filename stem.
        seed:       Optional integer seed for reproducibility.

    Returns:
        List of (water_out, fat_out) path pairs for each synthetic scan.
    """
    seg_path   = pathlib.Path(seg_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = seg_path.name
        for ext in (".nii.gz", ".nii", ".mha"):
            stem = stem.replace(ext, "")

    seg_sitk = sitk.ReadImage(str(seg_path))
    seg_arr  = sitk.GetArrayFromImage(seg_sitk).astype(np.int16)

    rng = np.random.default_rng(seed)
    outputs = []

    for i in range(n):
        water_arr = _synthesise_volume(seg_arr, "water", rng)
        fat_arr   = _synthesise_volume(seg_arr, "fat",   rng)

        def _save(arr: np.ndarray, suffix: str, idx: int = i) -> pathlib.Path:
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(seg_sitk)
            path = output_dir / f"{stem}_synth{idx:03d}_{suffix}.nii.gz"
            sitk.WriteImage(img, str(path))
            return path

        water_out = _save(water_arr, "water")
        fat_out   = _save(fat_arr,   "fat")
        outputs.append((water_out, fat_out))
        _log.info("[%d/%d] saved %s", i + 1, n, water_out.name)

    return outputs


# ── Registration and blending ─────────────────────────────────────────────────

def register_mri(
    fixed_path: str | pathlib.Path,
    moving_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    transform: str = "affine",
) -> sitk.Image:
    """Register *moving* onto *fixed* and save the result.

    Uses SimpleITK intensity-based registration with Mattes mutual information —
    appropriate for same-modality or cross-modality MRI.

    Args:
        fixed_path:  Reference image (the space to register into).
        moving_path: Image to be warped.
        output_path: Where to save the registered image (.nii.gz).
        transform:   'rigid' (rotation + translation) or
                     'affine' (adds scale + shear; better for cross-subject).

    Returns:
        The registered SimpleITK image (also saved to output_path).
    """
    fixed  = sitk.ReadImage(str(fixed_path),  sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(fixed.GetDimension()) if transform == "affine"
        else sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_tx, inPlace=False)

    _log.info("Registering (%s): %s -> %s", transform,
              pathlib.Path(moving_path).name, pathlib.Path(fixed_path).name)
    final_tx = reg.Execute(fixed, moving)
    _log.info("Metric: %.4f  |  Iterations: %d",
              reg.GetMetricValue(), reg.GetOptimizerIteration())

    registered = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(registered, str(output_path))
    _log.info("Saved -> %s", output_path)
    return registered


def blend_mris(
    image1: str | pathlib.Path | sitk.Image,
    image2: str | pathlib.Path | sitk.Image,
    output_path: str | pathlib.Path,
    alpha: float = 0.5,
) -> sitk.Image:
    """Blend two MRI images as a weighted average.

    Both images must share the same physical space.  Register first with
    `register_mri` if they do not.

    Args:
        image1:      First image (path or SimpleITK image).  Weight = alpha.
        image2:      Second image (path or SimpleITK image). Weight = 1 - alpha.
        output_path: Where to save the blended image (.nii.gz).
        alpha:       Blend weight for image1 in [0, 1].
                     0.5 = equal mix; 0.0 = image2 only; 1.0 = image1 only.

    Returns:
        The blended SimpleITK image (also saved to output_path).
    """
    if not isinstance(image1, sitk.Image):
        image1 = sitk.ReadImage(str(image1), sitk.sitkFloat32)
    if not isinstance(image2, sitk.Image):
        image2 = sitk.ReadImage(str(image2), sitk.sitkFloat32)

    blended = sitk.Add(
        sitk.Multiply(sitk.Cast(image1, sitk.sitkFloat32), alpha),
        sitk.Multiply(sitk.Cast(image2, sitk.sitkFloat32), 1.0 - alpha),
    )

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(blended, str(output_path))
    _log.info("Blended (alpha=%.2f) -> %s", alpha, output_path)
    return blended


def register_and_blend(
    fixed_path: str | pathlib.Path,
    moving_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    alpha: float = 0.5,
    transform: str = "affine",
) -> sitk.Image:
    """Register *moving* onto *fixed*, then blend the two.

    Convenience wrapper around `register_mri` + `blend_mris`.

    Args:
        fixed_path:  Reference image.
        moving_path: Image to register and blend in.
        output_path: Where to save the final blended image.
        alpha:       Weight for the fixed image (0.5 = equal mix).
        transform:   'rigid' or 'affine'.

    Returns:
        The blended SimpleITK image.
    """
    output_path = pathlib.Path(output_path)
    reg_tmp     = output_path.parent / f"_reg_tmp_{output_path.name}"

    registered = register_mri(fixed_path, moving_path, reg_tmp, transform=transform)
    blended    = blend_mris(fixed_path, registered, output_path, alpha=alpha)

    reg_tmp.unlink(missing_ok=True)
    return blended
