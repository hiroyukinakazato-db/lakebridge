"""Integration tests for Switch transpiler CLI functionality

Tests how Lakebridge CLI handles Switch transpiler differently from local transpilers.
Switch uses Databricks Jobs API instead of local execution, requiring special handling.

Focus: CLI integration layer (not end-to-end SQL conversion)
Approach: Lightweight mocking to test interface contracts
Execution time: ~5-10 seconds

Test coverage:
- test_switch_detection:
    Validates Switch availability detection and routing logic

- test_switch_parameter_mapping:
    Validates TranspileConfig to SwitchJobParameters conversion with new features

- test_switch_parameter_integration:
    Tests new parameters (source_format, target_type, builtin_prompt)

- test_switch_cli_integration_flow:
    Tests CLI correctly delegates to Switch job execution (lightweight, async only)

- test_switch_error_handling:
    Tests error handling for missing Switch package and configuration issues
"""
import logging
from unittest.mock import patch, MagicMock

import pytest

from databricks.labs.lakebridge import cli
from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.install import TranspilerInstaller


logger = logging.getLogger(__name__)


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

    def test_switch_parameter_mapping(self, valid_transpile_config):
        """Test conversion from TranspileConfig to SwitchJobParameters using new implementation"""
        try:
            # Test the new _create_switch_job_parameters function
            params, wait_for_completion, job_id = cli._create_switch_job_parameters(valid_transpile_config)

            # Verify mapping from TranspileConfig
            assert params.input_dir == "/Workspace/Users/test/sql_input"
            assert params.output_dir == "/Workspace/Users/test/notebooks_output"
            assert params.result_catalog == "test_catalog"
            assert params.result_schema == "test_schema"
            assert params.builtin_prompt.value == "snowflake"

            # Verify parameters from Switch config defaults
            assert params.endpoint_name == "databricks-claude-sonnet-4"
            assert params.token_count_threshold == 20000
            assert params.concurrency == 4
            assert params.comment_lang == "English"
            assert params.max_fix_attempts == 1
            assert params.log_level == "INFO"

            # Verify new parameters from defaults
            assert params.source_format == "sql"
            assert params.target_type == "notebook"
            assert params.output_extension is None

            # Verify wait_for_completion and job_id extraction
            assert wait_for_completion == False  # Default from config
            assert job_id == 12345  # From config fixture

        except ImportError:
            pytest.skip("Switch package not available for parameter mapping test")

    def test_switch_cli_integration_flow(self, mock_application_context, valid_transpile_config):
        """Test CLI integration flow with Switch (lightweight, mocked execution)"""
        # Mock the switch.api components for CLI integration testing
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
                    builtin_prompt="snowflake"
                )
                mock_create_params.return_value = (mock_params, False, 12345)  # async mode, job_id=12345

                # Execute Switch CLI integration (focus on interface contract)
                result = cli._execute_switch_directly(mock_application_context, valid_transpile_config)

                # Verify CLI interface contract (returns list with single dict)
                assert isinstance(result, list)
                assert len(result) == 1
                result_item = result[0]
                assert result_item["transpiler"] == "switch"
                assert result_item["job_id"] == 12345  # From config
                assert result_item["run_id"] == 987654321
                assert "run_url" in result_item

                # Verify correct API delegation (CLI integration responsibility)
                mock_job_runner_class.assert_called_once_with(
                    mock_application_context.workspace_client,
                    12345  # job_id from config
                )
                mock_job_runner.run_async.assert_called_once_with(mock_params)

                # Verify parameter validation was called
                mock_create_params.assert_called_once_with(valid_transpile_config)

    def test_switch_parameter_integration(self, valid_transpile_config):
        """Test new parameter integration (source_format, target_type, output_extension)"""
        try:
            # Test the new parameters in _create_switch_job_parameters
            params, _, _ = cli._create_switch_job_parameters(valid_transpile_config)

            # Verify conversion parameters
            assert hasattr(params, 'source_format')
            assert hasattr(params, 'target_type') 
            assert hasattr(params, 'output_extension')

            # Verify default values from config
            assert params.source_format.value == "sql"
            assert params.target_type.value == "notebook"
            assert params.output_extension is None

            # Verify request_params parameter
            assert hasattr(params, 'request_params')
            assert params.request_params is None

        except ImportError:
            pytest.skip("Switch package not available for parameter test")


    def test_switch_error_handling(self, mock_application_context):
        """Test CLI error handling for Switch integration issues"""
        # Test missing Switch package
        with patch('databricks.labs.lakebridge.cli._create_switch_job_parameters', 
                   side_effect=ImportError("Switch package not available")):
            config = TranspileConfig(
                transpiler_config_path="/fake/switch/config.yml",
                source_dialect="snowflake",
                input_source="/test/input",
                output_folder="/test/output"
            )

            with pytest.raises(ValueError, match="Switch transpiler is not properly installed"):
                cli._execute_switch_directly(mock_application_context, config)

        # Test missing job ID in config
        with patch('databricks.labs.lakebridge.cli._create_switch_job_parameters', 
                   side_effect=ValueError("Switch job not found")):
            with pytest.raises(RuntimeError, match="Switch execution failed"):
                cli._execute_switch_directly(mock_application_context, config)

    def test_config_job_id_extraction(self):
        """Test job ID extraction from Switch config (essential for CLI integration)"""
        switch_config = TranspilerInstaller.read_switch_config()
        assert switch_config is not None, "Switch config should exist"
        assert 'custom' in switch_config, "Config should have custom section"
        
        # job_id can be None if Switch hasn't been deployed yet, or a positive integer if deployed
        job_id = switch_config.get('custom', {}).get('job_id')
        if job_id is not None:
            assert isinstance(job_id, int), "Job ID should be an integer when present"
            assert job_id > 0, "Job ID should be positive when present"
