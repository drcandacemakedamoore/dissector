# Changelog

## [v0.0.3] - 2026-4-18

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
