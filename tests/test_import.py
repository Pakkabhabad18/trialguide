"""Smoke test: the package and every component folder can be imported."""

import importlib

COMPONENTS = ["data", "llm", "retrieval", "checker", "tools", "simulator",
              "selector", "policy", "baselines", "finetune", "eval"]


def test_package_imports():
    # If this fails, `pip install -e .` was not run, or a folder lost its __init__.py
    import trialguide
    assert trialguide.__version__


def test_components_import():
    # Each component is a sub-package with an owner noted in its docstring
    for name in COMPONENTS:
        mod = importlib.import_module(f"trialguide.{name}")
        assert "Owner:" in (mod.__doc__ or "")
