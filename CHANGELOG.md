# Changelog



## [v0.0.7] - 2026-06-21

### Added

- **Sheffield MRI dataset support** 
- Augmented Myosegmentum dataset support
- additional image augmentation functionality
- GPU metrics notebooks
- onzediff folder to train up our own diffusion model in 2.5D


## [v0.0.6] - 2026-06-06

### Added

- **Asian MRI dataset support** (Kway 2026, doi:10.6084/m9.figshare.31042489.v1) — 25 subjects, Thigh + Calf, 13 unilateral thigh muscle GT labels
- `asian_mri_viewer.ipynb` — slice viewer for the Asian dataset with modality switching and GT overlay
- `asian_scrollie.ipynb` — algorithm comparison viewer with GT overlay (supports NIfTI and NPZ algorithms)
- `asian_algos_avg_results.ipynb` — per-muscle and overall metrics aggregation for Asian dataset results
- Lambda notebooks to run all segmentation algorithms on the Asian dataset water images:
  - MuscleMap WB (`musclemap_asian_water_lambda.ipynb`)
  - MuscleMap Thigh (`musclemap_asian_water_thigh_lambda.ipynb`)
  - MedCLIP-SAMv2 Text+Boxes (`lambda_medclipsamv2_textboxes_asian.ipynb`)
  - Dafne Thigh (`dafne_asian_water_lambda.ipynb`)
- Evaluation notebooks comparing all algorithm outputs to the Asian GT (L/R evaluated separately to reflect unilateral GT labelling)
- `creation.py` — Dixon MRI synthesis and augmentation library: `generate_augmented` (TorchIO-based augmentation), `generate_synthetic` (SynthSeg-style synthesis from segmentation masks), `register_mri`, `blend_mris`, `register_and_blend`
- Synthetic MRI example notebooks (`synthetic_mri_example.ipynb`, `synthetic_mri_huashanmyo.ipynb`)

### Fixed

- `cross_dataset_registration.ipynb` — fat stack discovery regex was matching `_FATFRACTION_` instead of `_FAT_`; MyosegmenTUM fat stacks now correctly discovered
- MuscleMap label maps corrected throughout:
  - WB model: added previously missing Biceps Femoris Short Head (7191/7192), Adductor Longus (7211/7212), Adductor Brevis (7221/7222)
  - Thigh model: corrected to its own 1–28 label scheme (Zenodo record 19633000), distinct from WB 7xxx labels
- `.gitignore` `**` patterns added to exclude nested `ImageData`/`SegmentationMasks` dirs at any depth under `myosegmenTUM/`


## [v0.0.5] - 2026-05-29

### Added
- Experimental noise finder for Dixon
- Edge finder function local_std_map function
- fInd_body_oval function to find body in MRI
- FOlder comparison and file comparison functions by hashing
- Thigh version of MedSegDiff diffusion algorithm


### Changed

- DaFne segmentation evaluation and comparison notebooks all in Dafne folder, same for with medSAM on top, etc.
- pyproject toml has scipy
- `.gitignore` extended to exclude more files

---
## [v0.0.4] - 2026-05-18

### Added
- MedSAM refinement notebooks: bounding-box prompt, whole-mask prompt, logit-scaled mask prompt
- SLM-SAM2 video propagation notebook (`slmsam2_musclemap.ipynb`)
- Interactive slice viewers for MuscleMap + MedSAM results
- DaFne segmentation evaluation and comparison notebooks
- Inter-slice Dice evaluation

### Changed
- `.gitignore` extended to exclude all `eval_notebooks/**/*.npz` output files

---

## [0.0.2] — 2026-04-30

Last tagged release. Evaluation library with 3D metrics (boundary IoU, Dice), and graphing utilities.
