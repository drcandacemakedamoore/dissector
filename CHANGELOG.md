# Changelog

## [v0.0.5] - 2026-05-24

### Added
- Experimental noise finder for Dixon
- Edge finder function local_std_map function
- fInd_body_oval function to find body in MRI
- FOlder comparison by hashing

### Changed

- DaFne segmentation evaluation and comparison notebooks all in Dafne folder, same for with medSAM on top
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
