import pytest

from serve_dashboard import allowed


@pytest.mark.parametrize("path", [
    "/dashboard/",
    "/dashboard/index.html",
    "/live_runs/latest.txt",
    "/live_runs/20261109-004005/state.json?t=1",
])
def test_the_dashboard_and_snapshots_are_served(path):
    assert allowed(path)


@pytest.mark.parametrize("path", [
    "/",
    "/.env",
    "/src/serve_dashboard.py",
    "/data/processed/train.parquet",
    "/dashboard/../.env",
    "/dashboard/%2e%2e/.env",               # .. written in code: still blocked
    "/live_runs/%2E%2E/%2E%2E/etc/passwd",
    "/dashboard/%2e%2e%2f.env",
    "/dashboard/..%5c.env",                 # a Windows-style separator
    "/dashboardx/../.env",
])
def test_nothing_else_in_the_repo_is_served(path):
    assert not allowed(path)
