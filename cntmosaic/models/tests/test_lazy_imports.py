"""
Verify that ``to_inference_data`` is exposed via lazy ``__getattr__`` in
``cntmosaic.models`` without being loaded at import time from __init__.py.

Note: full numpyro lazy-loading is not yet achieved because the priors
package imports numpyro at module level.  This test verifies the specific
item from Stage 5: to_inference_data is no longer an eager top-level import
in models/__init__.py.
"""
import importlib
import sys


def test_to_inference_data_accessible_via_lazy_getattr():
    """to_inference_data is reachable through __getattr__ on the package."""
    import cntmosaic.models as m

    fn = m.to_inference_data
    assert callable(fn), "to_inference_data should be callable"


def test_to_inference_data_not_in_module_dict_before_access():
    """
    After a fresh import of cntmosaic.models the 'to_inference_data' name
    should not appear in ``vars(module)`` until it is first accessed — proving
    it is not an eager top-level import.
    """
    # Remove the module if already cached so we get a fresh import
    for key in list(sys.modules):
        if key.startswith("cntmosaic.models") and key != "cntmosaic.models.tests":
            del sys.modules[key]

    m = importlib.import_module("cntmosaic.models")

    # 'to_inference_data' should NOT be in the module's __dict__ yet
    assert "to_inference_data" not in vars(m), (
        "'to_inference_data' was eagerly loaded into cntmosaic.models.__dict__; "
        "it should only appear after first access via __getattr__."
    )

    # Once accessed, it should work
    fn = m.to_inference_data
    assert callable(fn)
