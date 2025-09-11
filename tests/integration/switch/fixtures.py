"""Shared test fixtures and configuration for Switch integration tests"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


@pytest.fixture
def switch_config_data():
    """Shared Switch configuration data matching actual config.yml"""
    return {
        "remorph": {
            "version": 1,
            "name": "switch",
            "command_line": ["echo", "Switch uses Jobs API, not LSP"],
            "dialects": [
                "mssql", "mysql", "netezza", "oracle", "postgresql", 
                "redshift", "snowflake", "synapse", "teradata",
                "python", "scala", "airflow"
            ]
        },
        "options": {
            "all": [
                {"flag": "source_format", "method": "CHOICE", "prompt": "Source file format", "choices": ["sql", "generic"], "default": "sql"},
                {"flag": "target_type", "method": "CHOICE", "prompt": "Target output type", "choices": ["notebook", "file"], "default": "notebook"},
                {"flag": "output_extension", "method": "QUESTION", "prompt": "Output file extension (for file target type) - press <enter> for none", "default": "<none>"},
                {"flag": "endpoint_name", "method": "QUESTION", "prompt": "Model endpoint name", "default": "databricks-claude-sonnet-4"},
                {"flag": "concurrency", "method": "QUESTION", "prompt": "Concurrency level", "default": 4},
                {"flag": "max_fix_attempts", "method": "QUESTION", "prompt": "Maximum fix attempts", "default": 1},
                {"flag": "log_level", "method": "CHOICE", "prompt": "Log level", "choices": ["DEBUG", "INFO", "WARNING", "ERROR"], "default": "INFO"},
                {"flag": "token_count_threshold", "method": "QUESTION", "prompt": "Token count threshold", "default": 20000},
                {"flag": "comment_lang", "method": "CHOICE", "prompt": "Comment language", "choices": ["English", "Japanese", "Chinese", "French", "German", "Italian", "Korean", "Portuguese", "Spanish"], "default": "English"},
                {"flag": "conversion_prompt_yaml", "method": "QUESTION", "prompt": "Custom conversion prompt YAML file path - press <enter> for default", "default": "<none>"},
                {"flag": "sql_output_dir", "method": "QUESTION", "prompt": "SQL output directory - press <enter> for none", "default": "<none>"},
                {"flag": "request_params", "method": "QUESTION", "prompt": "Additional request parameters (JSON) - press <enter> for none", "default": "<none>"},
                {"flag": "wait_for_completion", "method": "CHOICE", "prompt": "Wait for job completion?", "choices": ["true", "false"], "default": "false"}
            ]
        },
        "custom": {
            "execution_type": "jobs_api",
            "job_id": 12345,
            "job_name": "lakebridge-switch",
            "job_url": "https://test.databricks.com/jobs/12345",
            "switch_home": "/Workspace/Users/test/.lakebridge-switch",
            "created_by": "test@databricks.com"
        }
    }


@pytest.fixture
def switch_config_path(switch_config_data):
    """Create a temporary Switch config file for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create Switch config structure
        switch_dir = temp_path / "switch" / "lib"
        switch_dir.mkdir(parents=True)

        config_path = switch_dir / "config.yml"
        with config_path.open("w") as f:
            yaml.dump(switch_config_data, f)

        # Patch TranspilerRepository to use temp directory
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository.transpilers_path", return_value=temp_path):
            with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository.all_transpiler_names", return_value={"switch"}):
                # Create mock LSPConfig object
                mock_switch_config = MagicMock()
                mock_switch_config.custom = switch_config_data.get("custom", {})
                mock_switch_config.options = switch_config_data.get("options", {})
                with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository.all_transpiler_configs", return_value={"switch": mock_switch_config}):
                    yield config_path


@pytest.fixture
def mock_application_context():
    """Create a mock ApplicationContext for testing"""
    mock_ctx = MagicMock()
    mock_ctx.workspace_client = MagicMock()
    mock_ctx.workspace_client.config.host = "https://test.databricks.com"
    return mock_ctx


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing"""
    mock_ws = MagicMock()
    mock_ws.config.host = "https://test.databricks.com"
    return mock_ws


@pytest.fixture
def mock_install_result():
    """Mock SwitchInstaller.install() result"""
    result = MagicMock()
    result.job_id = 123456789
    result.job_name = "lakebridge-switch"
    result.job_url = "https://test.databricks.com/jobs/123456789"
    result.switch_home = "/Workspace/Users/test/.lakebridge-switch"
    result.created_by = "test@databricks.com"
    return result


def setup_transpiler_repository_mock(mock_repo, tmp_path, config_path, switch_config_data):
    """Setup common TranspilerRepository mock configuration"""
    mock_repo.transpilers_path.return_value = tmp_path
    mock_repo.transpiler_config_path.return_value = config_path

    mock_switch_config = MagicMock()
    mock_switch_config.custom = switch_config_data.get("custom", {})
    mock_repo.all_transpiler_configs.return_value = {"switch": mock_switch_config}

    return mock_repo
