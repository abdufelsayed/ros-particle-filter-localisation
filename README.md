# Particle-filter localisation in ROS

This project implements particle-filter localisation in ROS and compares its
behaviour with AMCL. It was developed for the Intelligent Robotics module at
the University of Birmingham in 2021.

The filter maintains a cloud of possible robot poses. It weights those
particles with the sensor model, resamples them, adds random particles when
confidence drops, and estimates the robot's pose from the largest K-means
cluster.

## Implementation

[`PFLocaliser`](src/pf_localisation/localiser.py) ties the filter together. The
supporting modules separate the main calculations:

- [`parameters.py`](src/pf_localisation/parameters.py) defines the constants
  used by the final implementation.
- [`pose.py`](src/pf_localisation/pose.py) copies poses and applies positional
  and angular noise.
- [`resampling.py`](src/pf_localisation/resampling.py) implements roulette-wheel
  and systematic resampling.
- [`estimation.py`](src/pf_localisation/estimation.py) assigns particles to
  spatial clusters and calculates distances.

The final localiser comes from `pf_final_actual.py`, the latest combined
version in the coursework files. Its calculations are unchanged.

## Coursework history

[`experiments/`](experiments/) contains earlier contributor versions and the
steps that led to the combined filter. My nearest-neighbour pose estimator is in
[`experiments/abdullah/pf_kNN.py`](experiments/abdullah/pf_kNN.py). The final
combined filter estimates pose with K-means instead.

The [assignment report](report/assignment-report.pdf) explains the design and
the coursework trials. It discusses the trade-off between recovering from a
poor initial pose and keeping the estimate stable while the robot moves. The
original [Word document](report/assignment-report.docx) is included as well.

## Running the code

The source depends on the coursework ROS framework, including
`PFLocaliserBase`, `pf_base.py`, `util.py`, the sensor model, and ROS message
types. Those framework files and the simulator setup are not part of this
repository, so the project does not run by itself.

To use the filter in that framework, place `pf_base.py` and `util.py` beside the
localiser package and configure `PFLocaliser` as the framework entry point.

## AI assistance

I used AI tools in 2026 to organise this repository, restructure the existing
code, and edit the documentation. My teammates and I completed the original
coursework in 2021 without AI assistance. Our original code, reports,
experiments, and recorded results are preserved in this repository or its Git
history.
