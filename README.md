<p align="center">
    <img style="width: 30%; height: 30%" src="dissector-logo.png">
</p>

# Dissector


(Badges to be updated soon.)

| fair-software.eu recommendations | |
| :-- | :--  |
|  code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/drcandacemakedamoore/dissector) |
|  license                      | [![github license badge](https://img.shields.io/github/license/drcandacemakedamoore/dissector)](https://github.com/drcandacemakedamoore/dissector) |
| community registry           | [![RSD](https://img.shields.io/badge/rsd-dissector-00a3e3.svg)](https://www.research-software.nl/software/dissector) [![workflow pypi badge](https://img.shields.io/pypi/v/dissector.svg?colorB=blue)](https://pypi.python.org/project/dissector/) |
| citation | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18909057.svg)](https://doi.org/10.5281/zenodo.18909057) |
| howfairis                       | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices** | &nbsp; |
| Documentation | [![Documentation Status](https://readthedocs.org/projects/dissector/badge/?version=latest)](https://dissector.readthedocs.io/en/latest/?badge=latest) |
| Build | [![build](https://github.com/drcandacemakedamoore/dissector/actions/workflows/build.yml/badge.svg)](https://github.com/drcandacemakedamoore/dissector/actions/workflows/build.yml) |
| Citation data consistency | [![cffconvert](https://github.com/drcandacemakedamoore/dissector/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/drcandacemakedamoore/dissector/actions/workflows/cffconvert.yml) |
| Link checker | [![link-check](https://github.com/drcandacemakedamoore/dissector/actions/workflows/link-check.yml/badge.svg)](https://github.com/drcandacemakedamoore/dissector/actions/workflows/link-check.yml) |

## How to use dissector

Dissector is is an open-source python library which contains methods for medical image segmentation and the evaluation of medical image segmentation. 
The module for evaluation contains methods to compare a segmentation to a 'ground truth' segmentation. This module should be run in it's own dedicated environment, which can be made in conda (see yaml environment_evaluation), pip or uv. Currently it has only been tested in conda, but the requirements are loose.  
The metrics beyond the expected (hausdorf, jaccard, fp, fn, dice) implemented are : boundary intersection over union, binary cross entropy (for comparing two binary segmentation masks). Please note at present a much more robust group of metrics can be created with libraries like MONAI ([metrics doc link](https://docs.monai.io/en/stable/metrics.html)). Metrics here are implemented for convenience i.e. no need to load pytorch, have lot of GPU space or cry as dependancies clash.
Images
can be extracted from [DICOM](https://www.dicomstandard.org/) files or used 
directly (nifti or arrays). 
The long term goal of the dissector project is to compare existing segmentation methods to a to be released state of the art method currently being built but not publicly released yet. This new method exploits the various types of noise in different MRI sequences...so computer science enthousiasts can probably guess the kind of model on the way. The primary authors are Candace Makeda H. Moore and Morris Alper.

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation

To install dissector from GitHub repository, do:

```console
git clone git@github.com:drcandacemakedamoore/dissector.git
cd dissector
python -m pip install .
```
Instructions for uv and conda soon to come...


## Documentation

One source should be https://drcandacemakedamoore.github.io/dissector/. Pypi and build the docs may take longer. Until then it should be online and buid in your docs/_build folder.

## Contributing

If you want to contribute to the development of dissector,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).


## Generative AI disclosure

This package has some code created with the aid of generative AI. All code on the main branch and in releases is human author written and/or reviewed. 