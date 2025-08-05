"""Integration tests for Switch transpiler CLI functionality

Tests how Lakebridge CLI handles Switch transpiler differently from local transpilers.
Switch uses Databricks Jobs API instead of local execution, requiring special handling.

Test coverage:
- test_switch_detection:
    Validates Switch availability detection
    Mocks: TranspilerInstaller.all_transpiler_names()
    
- test_switch_routing_decision:
    Tests decision logic to use Jobs API path
    Mocks: None (pure logic test)
    
- test_switch_parameter_mapping:
    Validates TranspileConfig to SwitchJobParameters conversion
    Mocks: switch.api.job_parameters.SwitchJobParameters
    
- test_switch_job_execution_async:
    Tests Switch job execution through Jobs API in async mode
    Mocks: switch.api.job_runner.SwitchJobRunner, _get_switch_job_id

- test_switch_job_execution_sync:
    Tests Switch job execution through Jobs API in sync mode with wait_for_completion
    Mocks: switch.api.job_runner.SwitchJobRunner, _get_switch_job_id
"""
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from databricks.labs.lakebridge import cli
from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.install import TranspilerInstaller


logger = logging.getLogger(__name__)


# Fixtures for common test setup
@pytest.fixture
def switch_config_path():
    """Create a temporary Switch config for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create Switch config structure
        switch_dir = temp_path / "switch" / "lib"
        switch_dir.mkdir(parents=True)
        
        # Create config matching actual Switch format
        config_data = {
            "remorph": {
                "version": 1,
                "name": "switch",
                "dialects": ["mysql", "netezza", "oracle", "postgresql", 
                           "redshift", "snowflake", "teradata", "tsql"],
                "command_line": ["echo", "Switch uses Jobs API, not LSP"]
            },
            "options": {
                "all": [
                    {"flag": "endpoint_name", "method": "QUESTION", "prompt": "Model endpoint name", "default": "databricks-claude-sonnet-4"},
                    {"flag": "token_count_threshold", "method": "QUESTION", "prompt": "Token count threshold", "default": 20000},
                    {"flag": "concurrency", "method": "QUESTION", "prompt": "Concurrency level", "default": 4},
                    {"flag": "comment_lang", "method": "CHOICE", "prompt": "Comment language", "choices": ["English", "Japanese", "Chinese", "French", "German", "Italian", "Korean", "Portuguese", "Spanish"], "default": "English"},
                    {"flag": "max_fix_attempts", "method": "QUESTION", "prompt": "Maximum fix attempts", "default": 1},
                    {"flag": "log_level", "method": "CHOICE", "prompt": "Log level", "choices": ["DEBUG", "INFO", "WARNING", "ERROR"], "default": "INFO"},
                    {"flag": "conversion_prompt_yaml", "method": "QUESTION", "prompt": "Custom conversion prompt YAML file path", "default": "<none>"},
                    {"flag": "existing_result_table", "method": "QUESTION", "prompt": "Existing result table name", "default": "<none>"},
                    {"flag": "sql_output_dir", "method": "QUESTION", "prompt": "SQL output directory", "default": "<none>"},
                    {"flag": "wait_for_completion", "method": "CHOICE", "prompt": "Wait for job completion?", "choices": ["true", "false"], "default": "false"}
                ]
            },
            "custom": {
                "execution_type": "jobs_api",
                "job_id": 12345,
                "job_name": "switch-sql-converter",
                "switch_home": "/Workspace/Users/test/.lakebridge-switch"
            }
        }
        
        config_path = switch_dir / "config.yml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)
        
        # Patch TranspilerInstaller to use temp directory
        with patch.object(TranspilerInstaller, "transpilers_path", return_value=temp_path):
            yield config_path


@pytest.fixture
def mock_application_context():
    """Create a mock ApplicationContext for testing"""
    mock_ctx = MagicMock()
    mock_ctx.workspace_client = MagicMock()
    mock_ctx.workspace_client.config.host = "https://test.databricks.com"
    return mock_ctx


@pytest.fixture
def valid_transpile_config(switch_config_path):
    """Create a valid TranspileConfig for Switch execution"""
    return TranspileConfig(
        transpiler_config_path=str(switch_config_path),
        source_dialect="snowflake",
        input_source="/Workspace/Users/test/sql_input",
        output_folder="/Workspace/Users/test/notebooks_output",
        catalog_name="test_catalog",
        schema_name="test_schema"
    )


class TestSwitchCLIIntegration:
    """Test how Lakebridge CLI integrates with Switch transpiler"""

    def test_switch_detection(self, switch_config_path):
        """Test that Switch is properly detected when installed"""
        # When Switch config exists, it should be detected
        is_switch_request = cli._is_switch_transpiler_request(str(switch_config_path))
        assert is_switch_request is True, "Switch should be detected when config exists"
        
        # Verify Switch appears in available transpilers
        all_transpilers = TranspilerInstaller.all_transpiler_names()
        assert 'switch' in all_transpilers, "Switch should appear in transpiler list"

    def test_switch_routing_decision(self, switch_config_path):
        """Test the decision logic for using Switch Jobs API path"""
        # Test with Switch config path
        switch_config = TranspileConfig(
            transpiler_config_path=str(switch_config_path),
            source_dialect="snowflake",
            input_source="/test/input",
            output_folder="/test/output"
        )
        
        # Should route to Switch when available and requested
        assert cli._is_switch_transpiler_request(switch_config.transpiler_config_path) is True
        
        # Test with non-Switch transpiler
        non_switch_config = TranspileConfig(
            transpiler_config_path="/path/to/morpheus/config.yml",
            source_dialect="snowflake",
            input_source="/test/input",
            output_folder="/test/output"
        )
        
        assert cli._is_switch_transpiler_request(non_switch_config.transpiler_config_path) is False

    def test_switch_parameter_mapping(self, mock_application_context, valid_transpile_config, switch_config_path):
        """Test conversion from TranspileConfig to SwitchJobParameters using new implementation"""
        try:
            # Test the new _create_switch_job_parameters function
            params, wait_for_completion, job_id = cli._create_switch_job_parameters(valid_transpile_config)
            
            # Verify mapping from TranspileConfig
            assert params.input_dir == "/Workspace/Users/test/sql_input"
            assert params.output_dir == "/Workspace/Users/test/notebooks_output"
            assert params.result_catalog == "test_catalog"
            assert params.result_schema == "test_schema"
            assert params.prompt_template.value == "snowflake"
            
            # Verify parameters from Switch config defaults
            assert params.endpoint_name == "databricks-claude-sonnet-4"
            assert params.token_count_threshold == 20000
            assert params.concurrency == 4
            assert params.comment_lang == "English"
            assert params.max_fix_attempts == 1
            assert params.log_level == "INFO"
            
            # Verify wait_for_completion and job_id extraction
            assert wait_for_completion == False  # Default from config
            assert job_id == 12345  # From config fixture
            
        except ImportError:
            pytest.skip("Switch package not available for parameter mapping test")

    def test_switch_job_execution_async(self, mock_application_context, valid_transpile_config, switch_config_path):
        """Test Switch job execution through Jobs API (async mode)"""
        # Mock the switch.api components inside _execute_switch_directly
        with patch('switch.api.job_runner.SwitchJobRunner') as mock_job_runner_class:
            mock_job_runner = MagicMock()
            mock_job_runner.run_async.return_value = 987654321  # Mock run ID
            mock_job_runner_class.return_value = mock_job_runner
            
            with patch('databricks.labs.lakebridge.cli._create_switch_job_parameters') as mock_create_params:
                from switch.api.job_parameters import SwitchJobParameters
                mock_params = SwitchJobParameters(
                    input_dir="/Workspace/Users/test/sql_input",
                    output_dir="/Workspace/Users/test/notebooks_output",
                    result_catalog="test_catalog",
                    result_schema="test_schema",
                    prompt_template="snowflake"
                )
                mock_create_params.return_value = (mock_params, False, 12345)  # async mode, job_id=12345
                
                # Execute Switch directly
                result = cli._execute_switch_directly(mock_application_context, valid_transpile_config)
                
                # Verify result structure (returns list with single dict)
                assert isinstance(result, list)
                assert len(result) == 1
                result_item = result[0]
                assert result_item["transpiler"] == "switch"
                assert result_item["job_id"] == 12345  # From config
                assert result_item["run_id"] == 987654321
                assert "run_url" in result_item
                
                # Verify job runner was called correctly
                mock_job_runner_class.assert_called_once_with(
                    mock_application_context.workspace_client,
                    12345  # job_id from config
                )
                mock_job_runner.run_async.assert_called_once_with(mock_params)

    def test_switch_job_execution_sync(self, mock_application_context, switch_config_path):
        """Test Switch job execution through Jobs API (sync mode with wait_for_completion)"""
        # Create config for sync execution
        config_for_sync = TranspileConfig(
            transpiler_config_path=str(switch_config_path),
            source_dialect="snowflake",
            input_source="/Workspace/Users/test/sql_input",
            output_folder="/Workspace/Users/test/notebooks_output",
            catalog_name="test_catalog",
            schema_name="test_schema"
        )

        # Mock the switch.api components for synchronous execution
        with patch('switch.api.job_runner.SwitchJobRunner') as mock_job_runner_class:
            mock_job_runner = MagicMock()

            # Mock run_sync return value
            mock_run = MagicMock()
            mock_run.run_id = 987654321
            mock_run.state.life_cycle_state.value = "TERMINATED"
            mock_run.state.result_state.value = "SUCCESS"
            mock_job_runner.run_sync.return_value = mock_run
            mock_job_runner_class.return_value = mock_job_runner

            with patch('databricks.labs.lakebridge.cli._create_switch_job_parameters') as mock_create_params:
                from switch.api.job_parameters import SwitchJobParameters
                mock_params = SwitchJobParameters(
                    input_dir="/Workspace/Users/test/sql_input",
                    output_dir="/Workspace/Users/test/notebooks_output",
                    result_catalog="test_catalog",
                    result_schema="test_schema",
                    prompt_template="snowflake"
                )
                mock_create_params.return_value = (mock_params, True, 12345)  # sync mode, job_id=12345

                # Execute Switch with wait option (via config)
                result = cli._execute_switch_directly(mock_application_context, config_for_sync)

                # Verify synchronous execution results (returns list with single dict)
                assert isinstance(result, list)
                assert len(result) == 1
                result_item = result[0]
                assert result_item["transpiler"] == "switch"
                assert result_item["job_id"] == 12345
                assert result_item["run_id"] == 987654321
                assert result_item["state"] == "TERMINATED"
                assert result_item["result_state"] == "SUCCESS"
                assert "run_url" in result_item

                # Verify run_sync was called instead of run_async
                mock_job_runner.run_sync.assert_called_once_with(mock_params)
                mock_job_runner.run_async.assert_not_called()


class TestSwitchHelperFunctions:
    """Test Switch-specific helper functions"""
    
    def test_get_switch_job_id_from_config(self, switch_config_path):
        """Test job ID extraction from config using new implementation"""
        switch_config = TranspilerInstaller.read_switch_config()
        job_id = switch_config.get('custom', {}).get('job_id') if switch_config else None
        assert job_id == 12345, "Should extract job ID from config"
    
    def test_get_switch_job_id_missing(self):
        """Test job ID extraction when config is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(TranspilerInstaller, "transpilers_path", return_value=Path(temp_dir)):
                switch_config = TranspilerInstaller.read_switch_config()
                job_id = switch_config.get('custom', {}).get('job_id') if switch_config else None
                assert job_id is None, "Should return None when config missing"
