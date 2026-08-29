# Particle-filter localisation

Group 11 coursework for Intelligent Robotics at the University of Birmingham, 2021. The project implements particle-filter localisation in ROS and compares its behavior qualitatively with AMCL.

The work explores particle-cloud initialisation, roulette-wheel and systematic resampling, adaptive random-particle injection, and pose estimation using averaging, a nearest-neighbour heuristic, and K-means clustering.

## Report

Read the submitted assignment report as [PDF](report/assignment-report.pdf) or [Word](report/assignment-report.docx). Both files are unchanged from the coursework material.

The report describes the development of the methods and the observed trade-off between recovery from poor initial poses and stability while moving. The observations come from the coursework trials; no experiments have been rerun for this repository.

## Implementation

The main implementation is [PFLocaliser](src/pf_localisation/localiser.py). It initialises the cloud, scores particles, adds recovery candidates, resamples, and estimates a pose using the largest K-means cluster.

- [parameters.py](src/pf_localisation/parameters.py): the selected implementation's constants.
- [pose.py](src/pf_localisation/pose.py): pose copying and noise generation.
- [resampling.py](src/pf_localisation/resampling.py): roulette-wheel and systematic sampling.
- [estimation.py](src/pf_localisation/estimation.py): spatial cluster assignment and distance calculations.

This is a structural extraction of `pf_final_actual.py`, the latest named combined version in the recovered files. It combines the report's final methods, but the exact configuration used for every trial is unknown. Calculations and existing defects are retained, including the mismatched cluster-helper call in pose estimation.

[Earlier experiments](experiments/) retain the contributor files and preceding combined versions unchanged. Abdullah's directly evidenced contribution is the [nearest-neighbour pose estimator](experiments/abdullah/pf_kNN.py); the final combined filter uses K-means instead.

## Runtime

These implementations belong to the ROS coursework framework. They import `PFLocaliserBase`, utility functions, and ROS message types; the base class, sensor model, and complete launch environment are not included. The files therefore do not form a standalone runnable package. To integrate with that framework, its `pf_base.py` and `util.py` must be available alongside `localiser.py` in the package; the framework entry point is `PFLocaliser`.

Inspect the code alongside the report for the implementation and design decisions. Reproducing the simulator trials requires the missing coursework framework and environment. There is no replacement synthetic experiment in this repository.

## Attribution

This was Group 11 coursework. The contributor folders and combined implementations preserve the organisation of the recovered source material. The report describes the group's work collectively.
