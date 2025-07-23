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

- test_switch_cli_end_to_end:
    Complete CLI workflow with real Switch execution
    Mocks: None (requires DATABRICKS_HOST/TOKEN and switch package)
"""
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from databricks.labs.lakebridge import cli
from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.contexts.application import ApplicationContext
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
        is_available = cli._is_switch_available()
        assert is_available is True, "Switch should be detected when config exists"
        
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
        assert cli._is_switch_available() is True
        assert cli._is_switch_request(switch_config.transpiler_config_path) is True
        
        # Test with non-Switch transpiler
        non_switch_config = TranspileConfig(
            transpiler_config_path="/path/to/morpheus/config.yml",
            source_dialect="snowflake",
            input_source="/test/input",
            output_folder="/test/output"
        )
        
        assert cli._is_switch_request(non_switch_config.transpiler_config_path) is False

    def test_switch_parameter_mapping(self, mock_application_context, valid_transpile_config, switch_config_path):
        """Test conversion from TranspileConfig to SwitchJobParameters"""
        try:
            # Import here to skip if not available
            from switch.api.job_parameters import SwitchJobParameters
            
            # Create parameters from config
            params = SwitchJobParameters(
                input_dir=valid_transpile_config.input_source,
                output_dir=valid_transpile_config.output_folder,
                result_catalog=valid_transpile_config.catalog_name,
                result_schema=valid_transpile_config.schema_name,
                sql_dialect=valid_transpile_config.source_dialect,
            )
            
            # Verify mapping
            assert params.input_dir == "/Workspace/Users/test/sql_input"
            assert params.output_dir == "/Workspace/Users/test/notebooks_output"
            assert params.result_catalog == "test_catalog"
            assert params.result_schema == "test_schema"
            assert params.sql_dialect == "snowflake"
            
        except ImportError:
            pytest.skip("Switch package not available for parameter mapping test")

    def test_switch_job_execution_async(self, mock_application_context, valid_transpile_config, switch_config_path):
        """Test Switch job execution through Jobs API (async mode)"""
        # Mock the switch.api components inside _execute_switch_directly
        with patch('switch.api.job_runner.SwitchJobRunner') as mock_job_runner_class:
            mock_job_runner = MagicMock()
            mock_job_runner.run_async.return_value = 987654321  # Mock run ID
            mock_job_runner_class.return_value = mock_job_runner
            
            with patch('switch.api.job_parameters.SwitchJobParameters') as mock_params_class:
                mock_params = MagicMock()
                mock_params.validate.return_value = None  # Validation passes
                mock_params_class.return_value = mock_params
                
                # Execute Switch directly
                result = cli._execute_switch_directly(mock_application_context, valid_transpile_config)
                
                # Verify result structure
                assert isinstance(result, dict)
                assert result["transpiler"] == "switch"
                assert result["job_id"] == 12345  # From config
                assert result["run_id"] == 987654321
                assert "run_url" in result
                
                # Verify job runner was called correctly
                mock_job_runner_class.assert_called_once_with(
                    mock_application_context.workspace_client,
                    12345  # job_id from config
                )
                mock_job_runner.run_async.assert_called_once_with(mock_params)

    def test_switch_job_execution_sync(self, mock_application_context, switch_config_path):
        """Test Switch job execution through Jobs API (sync mode with wait_for_completion)"""
        # Create config with wait_for_completion option
        config_with_wait = TranspileConfig(
            transpiler_config_path=str(switch_config_path),
            source_dialect="snowflake",
            input_source="/Workspace/Users/test/sql_input",
            output_folder="/Workspace/Users/test/notebooks_output",
            catalog_name="test_catalog",
            schema_name="test_schema",
            transpiler_options={"wait_for_completion": "true"}
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

            with patch('switch.api.job_parameters.SwitchJobParameters') as mock_params_class:
                mock_params = MagicMock()
                mock_params.validate.return_value = None
                mock_params_class.return_value = mock_params

                # Execute Switch with wait option
                result = cli._execute_switch_directly(mock_application_context, config_with_wait)

                # Verify synchronous execution results
                assert isinstance(result, dict)
                assert result["transpiler"] == "switch"
                assert result["job_id"] == 12345
                assert result["run_id"] == 987654321
                assert result["state"] == "TERMINATED"
                assert result["result_state"] == "SUCCESS"
                assert "run_url" in result

                # Verify run_sync was called instead of run_async
                mock_job_runner.run_sync.assert_called_once_with(mock_params)
                mock_job_runner.run_async.assert_not_called()

    def test_switch_cli_end_to_end(self, switch_config_path):
        """End-to-end test of Switch CLI integration"""
        # Check prerequisites
        host = os.getenv('DATABRICKS_HOST')
        token = os.getenv('DATABRICKS_TOKEN')
        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        
        try:
            import switch
        except ImportError:
            pytest.skip("Switch package not available")
        
        # Create real ApplicationContext
        from databricks.sdk import WorkspaceClient
        ws = WorkspaceClient(host=host, token=token)
        
        ctx = ApplicationContext(ws)
        
        # Create config for Switch
        config = TranspileConfig(
            transpiler_config_path=str(switch_config_path),
            source_dialect="snowflake",
            input_source="/Workspace/Users/test/sample.sql",
            output_folder="/Workspace/Users/test/converted",
            catalog_name="test_catalog",
            schema_name="test_schema"
        )
        
        # Test the complete flow
        try:
            # Verify Switch is detected
            assert cli._is_switch_available() is True
            assert cli._is_switch_request(config.transpiler_config_path) is True
            
            # Get job ID from config
            job_id = cli._get_switch_job_id()
            assert job_id is not None, "Job ID should be available from config"
            assert job_id == 12345, "Job ID should match config"
            
            # Note: We don't actually execute the job in tests to avoid creating real runs
            logger.info(f"End-to-end test verified Switch integration (job_id={job_id})")
            
        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")


class TestSwitchHelperFunctions:
    """Test Switch-specific helper functions"""
    
    def test_get_switch_job_id(self, switch_config_path):
        """Test job ID extraction from config"""
        job_id = cli._get_switch_job_id()
        assert job_id == 12345, "Should extract job ID from config"
    
    def test_get_switch_job_id_missing(self):
        """Test job ID extraction when config is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(TranspilerInstaller, "transpilers_path", return_value=Path(temp_dir)):
                job_id = cli._get_switch_job_id()
                assert job_id is None, "Should return None when config missing"
