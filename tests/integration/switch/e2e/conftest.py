"""Configuration for Switch E2E tests

This conftest.py isolates Switch E2E tests from the parent conftest.py files
that have PySpark dependencies. Switch E2E tests don't require PySpark and
should run independently.
"""
import os
import logging

import pytest

# Configure logging for Switch E2E tests
logging.getLogger("tests").setLevel("DEBUG")
logging.getLogger("databricks.labs.lakebridge").setLevel("DEBUG")
logging.getLogger("switch").setLevel("DEBUG")

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def e2e_test_environment():
    """Basic test environment validation for Switch E2E tests"""
    required_env_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        pytest.skip(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        "host": os.getenv("DATABRICKS_HOST"),
        "token": os.getenv("DATABRICKS_TOKEN"),
        "e2e_enabled": os.getenv("LAKEBRIDGE_SWITCH_E2E") == "true"
    }


# Prevent pytest from loading parent conftest files that have PySpark dependencies
def pytest_configure(config):
    """Configure pytest to avoid PySpark dependency issues"""
    if not os.getenv("LAKEBRIDGE_SWITCH_E2E") == "true":
        logger.info("Switch E2E tests are disabled. Set LAKEBRIDGE_SWITCH_E2E=true to enable.")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for Switch E2E specific needs"""
    if not os.getenv("LAKEBRIDGE_SWITCH_E2E") == "true":
        # If E2E is not enabled, mark all items as skipped
        skip_marker = pytest.mark.skip(reason="Switch E2E tests disabled")
        for item in items:
            item.add_marker(skip_marker)
