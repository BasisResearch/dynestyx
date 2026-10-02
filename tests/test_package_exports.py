"""Regression tests for curated package export surfaces."""

import importlib


def test_curated_package_all_entries_are_bound():
    module_names = [
        "dynestyx",
        "dynestyx.inference.latent",
        "dynestyx.inference.state_paths",
        "dynestyx.models",
        "dynestyx.simulation",
        "dynestyx.solvers",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        exported = getattr(module, "__all__", ())
        missing = [name for name in exported if not hasattr(module, name)]
        assert not missing, f"{module_name} has unbound __all__ entries: {missing}"
