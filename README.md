[![PyPI version](https://badge.fury.io/py/desdeo-mcdm.svg)](https://badge.fury.io/py/desdeo-mcdm)
[![Documentation Status](https://readthedocs.org/projects/desdeo-mcdm/badge/?version=latest)](https://desdeo-mcdm.readthedocs.io/en/latest/?badge=latest)

# desdeo-mcdm

Contains interactive optimization methods for solving multiobjective optimization problems. This package is part of
the [DESDEO framework](https://github.com/industrial-optimization-group/DESDEO).

## Installation

### For regular users
You can install the `desdeo-mcdm` package from the Python package index by issuing the command `pip install desdeo-mcdm`.

### For developers
Requires [poetry](https://python-poetry.org/). See `pyproject.toml` for Python package requirements. To install and use the this package with poetry, issue the following command:

1. `git clone https://github.com/industrial-optimization-group/desdeo-mcdm`
2. `cd desdeo-mcdm`
3. `poetry shell`
4. `poetry install`

## Documentation

Documentation for this package can be found [here](https://desdeo-mcdm.readthedocs.io/en/latest/)

## Currently implemented interactive methods

- Synchronous NIMBUS
- NAUTILUS Navigator
- E-NAUTILUS
- NAUTILUS
- NAUTILUSv2
- Reference point method
- (Convex) pareto navigator
- PAINT

## Citation

If you decide to use DESDEO is any of your works or research, we would appreciate you citing the appropiate paper published in [IEEE Access](https://doi.org/10.1109/ACCESS.2021.3123825) (open access).
