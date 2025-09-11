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

from databricks.labs.lakebridge.transpiler.installers import SwitchInstaller
from databricks.labs.lakebridge.transpiler.repository import TranspilerRepository
from .fixtures import switch_config_data, mock_install_result, mock_workspace_client, setup_transpiler_repository_mock


logger = logging.getLogger(__name__)


@pytest.fixture
def mock_transpiler_repository(tmp_path):
    """Create a mock TranspilerRepository for testing"""
    mock_repo = MagicMock(spec=TranspilerRepository)
    mock_repo.transpilers_path.return_value = tmp_path
    return mock_repo


class TestSwitchInstallationProcess:
    """Test Switch installation process focusing on workspace operations"""

    def test_workspace_deployment(self, tmp_path, switch_config_data, mock_install_result, mock_workspace_client):
        """Test workspace deployment creates Databricks job"""

        # Setup Switch config structure
        switch_dir = tmp_path / "switch" / "lib"
        switch_dir.mkdir(parents=True)
        config_path = switch_dir / "config.yml"

        # Create initial config file
        with config_path.open("w") as f:
            yaml.dump(switch_config_data, f)

        # Mock TranspilerRepository
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository") as mock_repo_class:
            mock_repo = MagicMock()
            setup_transpiler_repository_mock(mock_repo, tmp_path, config_path, switch_config_data)
            mock_repo_class.return_value = mock_repo

            # Mock SwitchAPIInstaller to return job information
            with patch("databricks.labs.lakebridge.transpiler.installers.SwitchAPIInstaller") as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create SwitchInstaller directly
                switch_installer = SwitchInstaller(mock_repo, mock_workspace_client)

                # Mock the display method to avoid stdout during tests
                with patch.object(switch_installer, "_display_switch_installation_details") as mock_display:
                    # Mock Switch version (now imported as SWITCH_VERSION)
                    with patch("databricks.labs.lakebridge.transpiler.installers.SWITCH_VERSION", "0.1.0"):
                        # Execute workspace deployment
                        switch_installer.install()

                        # Verify display method was called with install result
                        mock_display.assert_called_once_with(mock_install_result)

                # Verify SwitchAPIInstaller was called correctly
                mock_switch_installer_class.assert_called_once_with(mock_workspace_client)
                mock_installer.install.assert_called_once_with(
                    previous_job_id=12345,
                    previous_switch_home="/Workspace/Users/test/.lakebridge-switch"
                )

                # Verify version.json was created
                version_json_path = tmp_path / "switch" / "state" / "version.json"
                assert version_json_path.exists(), "version.json should be created"

                import json
                with version_json_path.open("r") as f:
                    version_data = json.load(f)

                assert version_data["version"] == "v0.1.0", "Version should match Switch version"
                assert "date" in version_data, "Version data should contain date field"

                logger.info(f"Workspace deployment completed. Job ID: {mock_install_result.job_id}")

    def test_config_update(self, tmp_path, switch_config_data, mock_install_result, mock_workspace_client):
        """Test config update writes job information to config.yml"""

        # Setup config environment
        switch_dir = tmp_path / "switch" / "lib"
        switch_dir.mkdir(parents=True)
        config_path = switch_dir / "config.yml"

        # Write initial config
        with config_path.open("w") as f:
            yaml.dump(switch_config_data, f)

        # Mock TranspilerRepository
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository") as mock_repo_class:
            mock_repo = MagicMock()
            setup_transpiler_repository_mock(mock_repo, tmp_path, config_path, switch_config_data)
            mock_repo_class.return_value = mock_repo

            # Create SwitchInstaller directly
            switch_installer = SwitchInstaller(mock_repo, mock_workspace_client)

            # Execute config update
            switch_installer._update_switch_config(mock_install_result)

            # Verify config was updated with job information
            with config_path.open("r") as f:
                updated_config = yaml.safe_load(f)

            assert "custom" in updated_config, "Config missing 'custom' section after update"
            custom = updated_config["custom"]

            assert custom["job_id"] == mock_install_result.job_id
            assert custom["job_name"] == mock_install_result.job_name
            assert custom["job_url"] == mock_install_result.job_url
            assert custom["switch_home"] == mock_install_result.switch_home
            assert custom["created_by"] == mock_install_result.created_by

            # Verify original config was preserved
            assert updated_config["remorph"]["name"] == "switch"
            assert "snowflake" in updated_config["remorph"]["dialects"]
            assert "options" in updated_config

            logger.info(f"Config updated with job ID {mock_install_result.job_id}")

    def test_installation_with_cleanup(self, tmp_path, mock_install_result, mock_workspace_client):
        """Test Switch installation with cleanup of previous installation"""
        # Setup environment with existing Switch installation
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

        # Mock TranspilerRepository
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository") as mock_repo_class:
            mock_repo = MagicMock()
            setup_transpiler_repository_mock(mock_repo, tmp_path, config_path, previous_config_data)
            mock_repo_class.return_value = mock_repo

            # Mock workspace deployment
            with patch("databricks.labs.lakebridge.transpiler.installers.SwitchAPIInstaller") as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create SwitchInstaller directly
                switch_installer = SwitchInstaller(mock_repo, mock_workspace_client)

                # Mock the display method to avoid stdout during tests
                with patch.object(switch_installer, "_display_switch_installation_details") as mock_display:
                    # Execute installation - should detect and pass previous installation info
                    switch_installer.install()

                    # Verify display method was called
                    mock_display.assert_called_once_with(mock_install_result)

                # Verify SwitchAPIInstaller.install was called with previous installation info
                mock_installer.install.assert_called_once_with(
                    previous_job_id=999888777,
                    previous_switch_home="/Workspace/Users/test/.lakebridge-switch-old"
                )

                # Verify config was updated with new job info
                with config_path.open("r") as f:
                    updated_config = yaml.safe_load(f)

                assert updated_config["custom"]["job_id"] == mock_install_result.job_id
                assert updated_config["custom"]["job_name"] == mock_install_result.job_name
                assert updated_config["custom"]["job_url"] == mock_install_result.job_url
                assert updated_config["custom"]["switch_home"] == mock_install_result.switch_home
                assert updated_config["custom"]["created_by"] == mock_install_result.created_by

                logger.info(f"Previous installation (job_id=999888777) cleaned up and new installation completed (job_id={mock_install_result.job_id})")

    def test_installation_error_handling(self, tmp_path, switch_config_data, mock_workspace_client, caplog):
        """Test error handling during installation process"""
        # Setup config environment
        switch_dir = tmp_path / "switch" / "lib"
        switch_dir.mkdir(parents=True)
        config_path = switch_dir / "config.yml"

        # Create initial config file
        with config_path.open("w") as f:
            yaml.dump(switch_config_data, f)

        # Mock TranspilerRepository
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository") as mock_repo_class:
            mock_repo = MagicMock()
            setup_transpiler_repository_mock(mock_repo, tmp_path, config_path, switch_config_data)
            mock_repo_class.return_value = mock_repo

            # Create SwitchInstaller directly
            switch_installer = SwitchInstaller(mock_repo, mock_workspace_client)

            # Mock version state setup (not relevant to error handling tests)
            with patch.object(switch_installer, "_setup_switch_version_state"):
                # Test SwitchAPIInstaller.install() failure (implementation: re-raise as RuntimeError)
                with patch("databricks.labs.lakebridge.transpiler.installers.SwitchAPIInstaller") as mock_switch_installer_class:
                    mock_installer = MagicMock()
                    mock_installer.install.side_effect = Exception("Job creation failed")
                    mock_switch_installer_class.return_value = mock_installer

                    with pytest.raises(RuntimeError, match="Switch workspace deployment failed"):
                        switch_installer.install()

                    logger.info("General exception handling verified: RuntimeError raised")

            logger.info("Error handling tests completed successfully")

    def test_config_validation(self, tmp_path):
        """Test config file validation and error handling through all_transpiler_configs"""
        with patch("databricks.labs.lakebridge.transpiler.repository.TranspilerRepository.transpilers_path", return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib" 
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Test missing config file (implementation: switch not in all_transpiler_configs)
            all_configs = TranspilerRepository.user_home().all_transpiler_configs()
            switch_config = all_configs.get("switch")
            assert switch_config is None, "Config file not found should result in switch not available"
            logger.info("Missing config file handling verified: None returned")

            # Test invalid config file (implementation: YAML parsing error raises exception)
            with config_path.open("w") as f:
                f.write("invalid: yaml: content: [")

            # Invalid YAML will raise an exception during all_transpiler_configs() call
            import yaml
            with pytest.raises(yaml.scanner.ScannerError):
                all_configs = TranspilerRepository.user_home().all_transpiler_configs()
            logger.info("Invalid YAML handling verified: ScannerError raised as expected")

            # Test valid config file (implementation: switch available in all_transpiler_configs)
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

            all_configs = TranspilerRepository.user_home().all_transpiler_configs()
            switch_config = all_configs.get("switch")
            assert switch_config is not None, "Valid config should be available in all_transpiler_configs"
            assert switch_config.name == "switch", "Config should have correct transpiler name"
            logger.info("Valid config handling verified: LSPConfig object returned")

            logger.info("Config validation tests completed successfully")
