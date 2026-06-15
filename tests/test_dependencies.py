"""Smoke-test that required runtime dependencies are importable in the venv."""


def test_import_pandas():
    import pandas as _  # noqa: F401


def test_import_pyarrow():
    import pyarrow as _  # noqa: F401


def test_import_requests():
    import requests as _  # noqa: F401


def test_import_pyproj():
    import pyproj as _  # noqa: F401


def test_import_sklearn():
    import sklearn as _  # noqa: F401


def test_import_optuna():
    import optuna as _  # noqa: F401


def test_import_pytest():
    import pytest as _  # noqa: F401
