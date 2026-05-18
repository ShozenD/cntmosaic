"""
Verify that importing cntmosaic.analysis does not eagerly load NumPyro.

After item 1.8 all numpyro imports in analysis/_arviz.py are deferred to
the body of svi_to_inference_data, so the analysis package must import
cleanly without NumPyro being present.
"""

import importlib
import sys


def _purge_modules(*prefixes: str) -> None:
    """Remove all loaded modules whose names start with any of *prefixes*."""
    for key in list(sys.modules):
        if any(key == p or key.startswith(p + ".") for p in prefixes):
            del sys.modules[key]


def test_analysis_import_does_not_load_numpyro():
    """Importing cntmosaic.analysis must not trigger a top-level numpyro import."""
    # Snapshot sys.modules so the purge+reimport doesn't corrupt module identity
    # for the rest of the test suite (numpyro._PYRO_STACK is a module-level
    # singleton; purging numpyro creates a new one, breaking existing references).
    modules_snapshot = dict(sys.modules)
    try:
        _purge_modules("cntmosaic.analysis", "numpyro")
        importlib.import_module("cntmosaic.analysis")
        numpyro_loaded = any(
            k == "numpyro" or k.startswith("numpyro.") for k in sys.modules
        )
        assert not numpyro_loaded, (
            "numpyro was loaded as a side-effect of `import cntmosaic.analysis`. "
            "Check for top-level `import numpyro` / `from numpyro` in analysis/_arviz.py "
            "or any module it imports at module scope."
        )
    finally:
        sys.modules.clear()
        sys.modules.update(modules_snapshot)


def test_svi_to_inference_data_accessible():
    """svi_to_inference_data must be callable via the analysis package."""
    import cntmosaic.analysis as a

    assert callable(a.svi_to_inference_data)
