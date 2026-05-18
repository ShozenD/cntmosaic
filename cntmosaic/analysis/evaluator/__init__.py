"""
Model evaluators for cntmosaic.

Each evaluator pairs with a matching summariser to compute error metrics and
uncertainty-quantification statistics for estimated contact intensity matrices:

+-----------------------------+-------------------------------+
| Summariser                  | Evaluator                     |
+=============================+===============================+
| ModelSummariser             | ModelEvaluatorBRC             |
+-----------------------------+-------------------------------+
| ModelSummariserPrem         | ModelEvaluatorPrem            |
+-----------------------------+-------------------------------+
| ModelSummariserSocialMix    | ModelEvaluatorSocialMix       |
+-----------------------------+-------------------------------+

All three concrete evaluators inherit from :class:`BaseModelEvaluator`, which
provides shared metric helpers (``validate_alpha``, ``interval_score``,
``compute_metrics``, ``aggregate_metrics``) and a common public interface
(``evaluate()``, ``evaluate_cint()``, ``evaluate_mcint()``, ``clear_cache()``).

:class:`SummariserProtocol` is a ``typing.Protocol`` that captures the minimal
interface a summariser must expose; it can be used for type annotations or
``isinstance`` checks where duck-typing is preferred over explicit inheritance.
"""

from ._base import BaseModelEvaluator, SummariserProtocol
from ._ModelEvaluatorBRC import ModelEvaluatorBRC
from ._ModelEvaluatorPrem import ModelEvaluatorPrem
from ._ModelEvaluatorSocialMix import ModelEvaluatorSocialMix

__all__ = [
    "BaseModelEvaluator",
    "SummariserProtocol",
    "ModelEvaluatorBRC",
    "ModelEvaluatorPrem",
    "ModelEvaluatorSocialMix",
]
