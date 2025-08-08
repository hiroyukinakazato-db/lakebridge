"""Integration tests for Switch transpiler installation process

Tests Switch installation workflow focusing on workspace deployment and job management.
Uses lightweight mocking to avoid external dependencies while testing core functionality.

Focus: Workspace operations and configuration management
Approach: Mock-heavy for workspace operations, fast execution
Execution time: ~10-15 seconds

Test coverage:
- test_workspace_deployment:
    Tests Databricks job creation via SwitchInstaller
    
- test_config_update:
    Verifies job information persistence in config.yml
    
- test_installation_with_cleanup:
    Tests cleanup of previous installations during new installation
    
- test_installation_error_handling:
    Tests error handling during installation process
"""
import logging
from unittest.mock import patch, MagicMock

import pytest
import yaml

from databricks.labs.lakebridge.install import WorkspaceInstaller


logger = logging.getLogger(__name__)


@pytest.fixture
def minimal_workspace_installer():
    """Create a minimal WorkspaceInstaller for testing"""
    return WorkspaceInstaller(
        ws=None, prompts=None, installation=None, install_state=None,
        product_info=None, resource_configurator=None, workspace_installation=None
    )


class TestSwitchInstallationProcess:
    """Test Switch installation process focusing on workspace operations"""

    def test_workspace_deployment(self, tmp_path, switch_config_data, mock_install_result, mock_workspace_client):
        """Test workspace deployment creates Databricks job"""

        # Setup Switch config structure
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Create initial config file
            with config_path.open("w") as f:
                yaml.dump(switch_config_data, f)

            # Mock SwitchInstaller to return job information
            with patch('switch.api.installer.SwitchInstaller') as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create WorkspaceInstaller with minimal dependencies
                installer = WorkspaceInstaller(
                    ws=mock_workspace_client,
                    prompts=None, installation=None, install_state=None,
                    product_info=None, resource_configurator=None, workspace_installation=None
                )

                # Mock the display method to avoid stdout during tests
                with patch.object(installer, '_display_switch_installation_details') as mock_display:
                    # Execute workspace deployment
                    installer.install_switch()

                    # Verify display method was called with install result
                    mock_display.assert_called_once_with(mock_install_result)

                # Verify SwitchInstaller was called correctly
                mock_switch_installer_class.assert_called_once_with(mock_workspace_client)
                mock_installer.install.assert_called_once_with(
                    previous_job_id=12345,
                    previous_switch_home="/Workspace/Users/test/.lakebridge-switch"
                )

                logger.info(f"Workspace deployment completed. Job ID: {mock_install_result.job_id}")

    def test_config_update(self, tmp_path, switch_config_data, mock_install_result, minimal_workspace_installer):
        """Test config update writes job information to config.yml"""

        # Setup config environment
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Write initial config
            with config_path.open("w") as f:
                yaml.dump(switch_config_data, f)

            # Execute config update
            minimal_workspace_installer._update_switch_config(mock_install_result)

            # Verify config was updated with job information
            with config_path.open('r') as f:
                updated_config = yaml.safe_load(f)

            assert 'custom' in updated_config, "Config missing 'custom' section after update"
            custom = updated_config['custom']

            assert custom['job_id'] == mock_install_result.job_id
            assert custom['job_name'] == mock_install_result.job_name
            assert custom['job_url'] == mock_install_result.job_url
            assert custom['switch_home'] == mock_install_result.switch_home
            assert custom['created_by'] == mock_install_result.created_by

            # Verify original config was preserved
            assert updated_config['remorph']['name'] == 'switch'
            assert 'snowflake' in updated_config['remorph']['dialects']
            assert 'options' in updated_config

            logger.info(f"Config updated with job ID {mock_install_result.job_id}")

    def test_installation_with_cleanup(self, tmp_path, mock_install_result, mock_workspace_client):
        """Test Switch installation with cleanup of previous installation"""
        # Setup environment with existing Switch installation
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Create config with previous installation info
            previous_config_data = {
                "remorph": {
                    "version": 1,
                    "name": "switch",
                    "dialects": ["mysql", "snowflake", "teradata"],
                    "command_line": ["echo", "Switch uses Jobs API, not LSP"]
                },
                "options": {"all": []},
                "custom": {
                    "execution_type": "jobs_api",
                    "job_id": 999888777,
                    "job_name": "old-switch-job",
                    "switch_home": "/Workspace/Users/test/.lakebridge-switch-old"
                }
            }

            with config_path.open("w") as f:
                yaml.dump(previous_config_data, f)

            # Mock workspace deployment
            with patch('switch.api.installer.SwitchInstaller') as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create WorkspaceInstaller
                installer = WorkspaceInstaller(
                    ws=mock_workspace_client,
                    prompts=None, installation=None, install_state=None,
                    product_info=None, resource_configurator=None, workspace_installation=None
                )

                # Mock the display method to avoid stdout during tests
                with patch.object(installer, '_display_switch_installation_details') as mock_display:
                    # Execute installation - should detect and pass previous installation info
                    installer.install_switch()

                    # Verify display method was called
                    mock_display.assert_called_once_with(mock_install_result)

                # Verify SwitchInstaller.install was called with previous installation info
                mock_installer.install.assert_called_once_with(
                    previous_job_id=999888777,
                    previous_switch_home="/Workspace/Users/test/.lakebridge-switch-old"
                )

                # Verify config was updated with new job info
                with config_path.open('r') as f:
                    updated_config = yaml.safe_load(f)

                assert updated_config['custom']['job_id'] == mock_install_result.job_id
                assert updated_config['custom']['job_name'] == mock_install_result.job_name
                assert updated_config['custom']['job_url'] == mock_install_result.job_url
                assert updated_config['custom']['switch_home'] == mock_install_result.switch_home
                assert updated_config['custom']['created_by'] == mock_install_result.created_by

                logger.info(f"Previous installation (job_id=999888777) cleaned up and new installation completed (job_id={mock_install_result.job_id})")

    def test_installation_error_handling(self, tmp_path, switch_config_data, mock_workspace_client, caplog):
        """Test error handling during installation process"""
        # Setup config environment
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"
            
            # Create initial config file
            with config_path.open("w") as f:
                yaml.dump(switch_config_data, f)

            installer = WorkspaceInstaller(
                ws=mock_workspace_client,
                prompts=None, installation=None, install_state=None,
                product_info=None, resource_configurator=None, workspace_installation=None
            )

            # Test SwitchInstaller import error (implementation: output warning and continue normally)
            with patch('switch.api.installer.SwitchInstaller', side_effect=ImportError("Switch package not available")):
                with caplog.at_level(logging.WARNING):
                    installer.install_switch()  # should complete normally
                    
                # verify warning message is logged
                assert "Switch package not available for workspace deployment" in caplog.text
                logger.info("ImportError handling verified: warning logged, no exception raised")

            # Test SwitchInstaller.install() failure (implementation: re-raise as RuntimeError)
            with patch('switch.api.installer.SwitchInstaller') as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.side_effect = Exception("Job creation failed")
                mock_switch_installer_class.return_value = mock_installer

                with pytest.raises(RuntimeError, match="Switch workspace deployment failed"):
                    installer.install_switch()
                
                logger.info("General exception handling verified: RuntimeError raised")

            logger.info("Error handling tests completed successfully")

    def test_config_validation(self, tmp_path):
        """Test config file validation and error handling"""
        from databricks.labs.lakebridge.install import TranspilerInstaller
        
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib" 
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Test missing config file (implementation: return None)
            config = TranspilerInstaller.read_switch_config()
            assert config is None, "Config file not found should return None"
            logger.info("Missing config file handling verified: None returned")

            # Test invalid config file (implementation: return None)
            with config_path.open("w") as f:
                f.write("invalid: yaml: content: [")

            config = TranspilerInstaller.read_switch_config()
            assert config is None, "Invalid YAML should return None"
            logger.info("Invalid YAML handling verified: None returned")

            # Test valid config file (implementation: return parsed result)
            # Use actual Switch config structure (version: 1 is required)
            valid_config = {
                "remorph": {
                    "version": 1,
                    "name": "switch",
                    "dialects": ["snowflake"],
                    "command_line": ["echo", "test"]
                },
                "options": {"all": []},
                "custom": {}
            }
            with config_path.open("w") as f:
                yaml.dump(valid_config, f)

            config = TranspilerInstaller.read_switch_config()
            assert config is not None, "Valid config should return parsed data"
            assert config["remorph"]["name"] == "switch", "Config content should be correctly parsed"
            logger.info("Valid config handling verified: parsed data returned")

            logger.info("Config validation tests completed successfully")
