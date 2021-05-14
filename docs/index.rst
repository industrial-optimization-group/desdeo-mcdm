.. desdeo-mcdm documentation master file, created by
   sphinx-quickstart on Mon Jun  1 14:25:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to desdeo-mcdm's documentation
======================================

Contains interactive optimization methods for solving multiobjective optimizaton problems. This package is part of the DESDEO framework.


Installation
============

To install and use this package on a \*nix-based system, follow one of the following procedures.


For users
---------


First, create a new virtual environment for the project. Then install the package using the following command:

::

    $ pip install desdeo-mcdm




For developers
--------------
It requires `poetry <https://python-poetry.org/>`__  to be installed. See `pyproject.toml` for Python package requirements.

Download the code or clone it with the following command:

::

    $ git clone https://github.com/industrial-optimization-group/desdeo-mcdm

Then, create a new virtual environment for the project and install the package in it:

::

    $ cd desdeo-mcdm
    $ poetry init
    $ poetry install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   background
   api
   examples
   


Currently implemented methods
=============================

- Synchronous NIMBUS
- NAUTILUS Navigator
- E-NAUTILUS
- NAUTILUS

Coming soon
===========

- Pareto Navigator
- NAUTILUSv2
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
