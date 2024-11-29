.. BRC documentation master file, created by
   sphinx-quickstart on Fri Oct 18 11:50:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contact Mosaic documentation
============================

What is Contact Mosaic?
-----------------------
Contact Mosaic (``cntmosaic``) is a Python package for analysing social contact patterns from 
social contact data. It provides a set of tools to process, analyse, simulate, and visualise social contact data.
It also provides a set of models to infer social contact matrices from real world social contact data.
The models in ``cntmosaic`` are implemented using the probabilistic programming language `Numpyro <https://num.pyro.ai/en/stable/index.html>`_ which allows for
both Hamiltonian Monte Carlo (HMC) based full Bayesian inference and fast stochastic variational inference (SVI).

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   usage/setup
   usage/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Introductory Tutorial

.. toctree::
   :maxdepth: 1
   :caption: API and Developer Reference

   documentation/preprocess
   documentation/models
   documentation/simulation

