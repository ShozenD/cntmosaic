from ._Prior2D import Prior2D
from ._HSGP2D import HSGP2D
from ._TensorSpline2D import TensorSpline2D
from ._PenalisedTensorSpline2D import PenalisedTensorSpline2D
from ._IGMRF2D import IGMRF2D
from ._GMRF2D import GMRF2D
from ._vdKassteele import vdKassteele
from ._Hill import Hill

__all__ = [
    'Prior2D',
    'HSGP2D',
    'TensorSpline2D',
    'PenalisedTensorSpline2D'
    'IGMRF2D',
    'GMRF2D',
    'Hill'
]