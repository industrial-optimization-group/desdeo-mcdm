.. desdeo-mcdm documentation master file, created by
   sphinx-quickstart on Mon Jun  1 14:25:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to desdeo-mcdm's documentation
======================================

Contains interactive optimization methods for solving multiobjective optimizaton problems. This package is part of the DESDEO framework.


Requirements
============

* Python 3.7 or newer.
* `Poetry dependency manager <https://python-poetry.org/>`__ : Only for developers.

See `pyproject.toml` for Python package requirements.


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

.. table:: 
   :widths: 20 80

   =============================  ========================================================================================================================================
     Algorithm                    Reference
   =============================  ========================================================================================================================================
   **Synchronous NIMBUS**         Miettinen, K., Mäkelä, M.M.: Synchronous approach in interactive multiobjective optimization. 
                                  Eur. J. Oper. Res. 170(3), 909–922 (2006)
   **NAUTILUS Navigator**         Ruiz, A. B., Ruiz, F., Miettinen, K., Delgado-Antequera, L., & Ojalehto, V. (2019). 
                                  NAUTILUS Navigator : free search interactive multiobjective optimization without trading-off. 
                                  Journal of Global Optimization, 74 (2), 213-231. doi:10.1007/s10898-019-00765-2
   **E-NAUTILUS**                 Ruiz, A., Sindhya, K., Miettinen, K., Ruiz, F., & Luque, M. (2015). 
                                  E-NAUTILUS: A decision support system for complex multiobjective optimization problems based on the NAUTILUS method. 
                                  European Journal of Operational Research, 246 (1), 218-231. doi:10.1016/j.ejor.2015.04.027
   **NAUTILUS**                   Kaisa Miettinen, Petri Eskelinen, Francisco Ruiz, Mariano Luque,
                                  NAUTILUS method: An interactive technique in multiobjective optimization based on the nadir point,
                                  European Journal of Operational Research, Volume 206, Issue 2, 2010,
                                  Pages 426-434, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2010.02.041.
   **Reference Point Method**     Andrzej P. Wierzbicki, A mathematical basis for satisficing decision making,
                                  Mathematical Modelling, Volume 3, Issue 5,1982,
                                  Pages 391-405, ISSN 0270-0255, https://doi.org/10.1016/0270-0255(82)90038-0.
   **NAUTILUSv2**                 Miettinen, K., Podkopaev, D., Ruiz, F. et al. 
                                  A new preference handling technique for interactive multiobjective optimization without trading-off. 
                                  J Glob Optim 63, 633–652 (2015). https://doi.org/10.1007/s10898-015-0301-8
   =============================  ========================================================================================================================================

Coming soon
===========

- Pareto Navigator


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
