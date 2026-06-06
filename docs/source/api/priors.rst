Priors
======

The priors module provides spatial prior classes for 2D fields used in contact matrix models.
All priors live under ``cntmosaic.models.numpyro.priors``.

Base Class
----------

Prior2D
^^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.Prior2D
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Gaussian Process Priors
-----------------------

HSGP2D
^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.HSGP2D
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Intrinsic Gaussian Markov Random Field Priors
---------------------------------------------

IGMRF2D
^^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.IGMRF2D
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Spline-Based Priors
-------------------

Spline2D
^^^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.Spline2D
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

PSpline2D
^^^^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.PSpline2D
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Special Effect Priors
---------------------

Hill
^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.Hill
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

vdKassteele2D
^^^^^^^^^^^^^

.. autoclass:: cntmosaic.models.numpyro.priors.vdKassteele2D
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
