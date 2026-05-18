Dissector
=========

**Dissector** is an open-source Python library for evaluating medical image segmentations,
with a focus on MRI-derived muscle segmentations.

It provides metrics for comparing a predicted segmentation against a ground-truth mask,
including standard measures (Dice, Jaccard, Hausdorff, FP, FN) and additional metrics
(boundary IoU, binary cross-entropy, inter-slice Dice) that do not require heavy
dependencies such as PyTorch or a GPU.

The long-term goal of the project is to benchmark existing segmentation methods —
including MuscleMap whole-body segmentation and MedSAM-based refinement pipelines —
against a forthcoming state-of-the-art method that exploits multi-sequence MRI noise structure.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
