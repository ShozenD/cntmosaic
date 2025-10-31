Models
======

The models module contains classes for estimating contact matrices from survey data using various Bayesian and frequentist approaches.

Contact Matrix Models
---------------------

SocialMix
^^^^^^^^^

.. autoclass:: cntmosaic.models.SocialMix
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Prem
^^^^

.. autoclass:: cntmosaic.models.Prem
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Bayesian Rate Consistency Models
---------------------------------

BRC (Base Class)
^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.models.BRC
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

BRCfine
^^^^^^^

.. autoclass:: cntmosaic.models.BRCfine
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

BRCrefine
^^^^^^^^^

.. autoclass:: cntmosaic.models.BRCrefine
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

HiBRCfine
^^^^^^^^^

.. autoclass:: cntmosaic.models.HiBRCfine
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

HiBRCrefine
^^^^^^^^^^^

.. autoclass:: cntmosaic.models.HiBRCrefine
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Utility Functions
-----------------

.. autofunction:: cntmosaic.models.to_inference_data
