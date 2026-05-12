# Root-level conftest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]
