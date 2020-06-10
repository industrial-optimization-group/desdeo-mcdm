# desdeo-mcdm

Contains interactive optimization methods for solving multiobjective optimizaton problems. This package is part of the DESDEO framework.

## Installation

Requires [poetry](https://python-poetry.org/). See `pyproject.toml` for Python package requirements. To install and use the this package:

1. `git clone https://github.com/industrial-optimization-group/desdeo-mcdm`
2. `cd desdeo-mcdm`
3. `poetry init`
4. `poetry install`

## Documentation

Documentation for this package can be found [here](https://desdeo-mcdm.readthedocs.io/en/latest/)

## Currently implemented methods

- Synchronous NIMBUS
- NAUTILUS Navigator
- E-NAUTILUS

## Coming soon

- Pareto Navigator
- NAUTILUSv2

## Demonstrations

### NAUTILUS Navigator

The implementation of NAUTILUS Navigator has been used to build a web based graphical user interface found [online](https://dash.misitano.xyz). Feel free to try it! The source code for this demo is available [here](https://github.com/gialmisi/desdeo-dash).

Here is a video of the webapp in action:
![NAUTILUS Navigator in action.](https://github.com/industrial-optimization-group/desdeo-mcdm/blob/master/assets/nautilus_nav_demo.gif "A gif of the demo in action.")
