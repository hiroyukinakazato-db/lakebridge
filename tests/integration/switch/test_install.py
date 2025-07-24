"""Integration tests for Switch transpiler installation process

Switch differs from other transpilers (morpheus, bladebridge) as it runs as a Databricks job
rather than locally, providing better scalability for large-scale SQL conversions.
The job ID must be persisted in config.yml for subsequent transpile commands.

Test coverage:
- test_switch_pypi_installation: 
    Validates PyPI package installation and config structure
    Mocks: TranspilerInstaller.install_from_pypi (uses real TestPyPI config)
    
- test_switch_workspace_deployment: 
    Tests Databricks job creation via SwitchInstaller
    Mocks: switch.api.installer.SwitchInstaller (for fast execution)
    
- test_switch_config_update: 
    Verifies job information persistence in config.yml
    Mocks: None (direct file operations)
    
- test_switch_with_previous_installation: 
    Tests cleanup of previous installations
    Mocks: switch.api.installer.SwitchInstaller
"""
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, create_autospec, MagicMock

import pytest
import yaml

from databricks.labs.lakebridge.install import WorkspaceInstaller


logger = logging.getLogger(__name__)


# Fixtures for common test setup
@pytest.fixture
def testpypi_install():
    """Install Switch package from TestPyPI and return package info"""

    # Install from TestPyPI
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "-i", "https://test.pypi.org/simple/", 
        "databricks-switch-plugin",
        "--force-reinstall",  # Ensure we get the latest version
        "--no-deps"  # Avoid dependency conflicts in test environment
    ], capture_output=True, text=True)

    if result.returncode != 0:
        pytest.skip(f"Failed to install from TestPyPI: {result.stderr}")

    # Get installed package info
    try:
        import switch
        package_path = Path(switch.__file__).parent
        return {
            "package_path": package_path,
            "version": getattr(switch, "__version__", "unknown")
        }
    except ImportError as e:
        pytest.skip(f"Failed to import installed package: {e}")


@pytest.fixture  
def mock_install_result():
    """Mock SwitchInstaller.install() result"""
    result = MagicMock()
    result.job_id = 123456789
    result.job_name = "switch-sql-converter"
    result.switch_home = "/Workspace/Users/test/.lakebridge-switch"
    return result


@pytest.fixture
def mock_workspace_installer():
    """Create a WorkspaceInstaller with all required mocks for testing"""
    def _create_installer(ws):
        from databricks.sdk import WorkspaceClient
        from databricks.labs.blueprint.tui import MockPrompts
        from databricks.labs.blueprint.installation import MockInstallation
        from databricks.labs.blueprint.installer import InstallState
        from databricks.labs.blueprint.wheels import ProductInfo
        from databricks.labs.lakebridge.deployment.configurator import ResourceConfigurator
        from databricks.labs.lakebridge.deployment.installation import WorkspaceInstallation

        return WorkspaceInstaller(
            ws=ws,
            prompts=MockPrompts({}),
            installation=MockInstallation({}),
            install_state=InstallState.from_installation(MockInstallation({})),
            product_info=ProductInfo.from_class(WorkspaceInstaller),
            resource_configurator=create_autospec(ResourceConfigurator),
            workspace_installation=create_autospec(WorkspaceInstallation)
        )
    return _create_installer


@pytest.fixture
def minimal_workspace_installer():
    """Create a minimal WorkspaceInstaller for config-only tests"""
    return WorkspaceInstaller(
        ws=None, prompts=None, installation=None, install_state=None,
        product_info=None, resource_configurator=None, workspace_installation=None
    )


class TestSwitchInstallationProcess:
    """Test the complete Switch installation process step by step"""

    def test_switch_pypi_installation(self, tmp_path, testpypi_install):
        """Test PyPI installation creates local Switch package structure"""

        # Real PyPI installation that copies actual config.yml
        def real_pypi_install(local_name, pypi_name, artifact):
            """Simulate PyPI installation using actual TestPyPI package config"""
            # Create switch directory structure
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)

            # Copy actual config.yml from installed TestPyPI package
            package_path = testpypi_install["package_path"]
            source_config = package_path / "lsp" / "config.yml"
            target_config = switch_dir / "config.yml"

            if source_config.exists():
                import shutil
                shutil.copy(source_config, target_config)
            else:
                pytest.skip(f"Config file not found in TestPyPI package: {source_config}")

        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            with patch('databricks.labs.lakebridge.install.TranspilerInstaller.install_from_pypi', side_effect=real_pypi_install):

                # Execute PyPI installation
                from databricks.labs.lakebridge.install import TranspilerInstaller
                TranspilerInstaller.install_from_pypi("switch", "databricks-switch-plugin", None)

                # Verify package structure was created
                config_path = tmp_path / "switch" / "lib" / "config.yml"
                assert config_path.exists(), "Config file not created by PyPI installation"

                # Verify actual config.yml content from TestPyPI
                with config_path.open() as f:
                    actual_config = yaml.safe_load(f)

                # Verify it contains the real Switch configuration
                assert actual_config["remorph"]["name"] == "switch"
                assert "snowflake" in actual_config["remorph"]["dialects"]
                assert "mysql" in actual_config["remorph"]["dialects"]  # TestPyPI version has 8 dialects
                assert "custom" in actual_config
                assert actual_config["custom"]["execution_type"] == "jobs_api"

                # Verify Switch is discoverable by TranspilerInstaller
                discovered_config = TranspilerInstaller.transpiler_config_path("switch")
                assert discovered_config == config_path

                # Verify Switch appears in available transpilers
                all_transpilers = TranspilerInstaller.all_transpiler_names()
                assert 'switch' in all_transpilers

                logger.info(f"PyPI installation completed successfully (TestPyPI v{testpypi_install['version']})")

    def test_switch_workspace_deployment(self, tmp_path, testpypi_install, mock_install_result, mock_workspace_installer):
        """Test workspace deployment creates Databricks job"""

        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Check credentials after loading .env
        host = os.getenv('DATABRICKS_HOST')
        token = os.getenv('DATABRICKS_TOKEN')
        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env file.")

        # Setup mock environment with real TestPyPI config
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            # Create Switch package structure with actual TestPyPI config
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Copy actual config.yml from TestPyPI package
            package_path = testpypi_install["package_path"]
            source_config = package_path / "lsp" / "config.yml"
            if source_config.exists():
                import shutil
                shutil.copy(source_config, config_path)
            else:
                pytest.skip(f"Config file not found in TestPyPI package: {source_config}")

            # Mock workspace deployment to return job information
            with patch('switch.api.installer.SwitchInstaller') as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create WorkspaceInstaller using fixture
                from databricks.sdk import WorkspaceClient
                ws = WorkspaceClient(
                    host=os.getenv('DATABRICKS_HOST'),
                    token=os.getenv('DATABRICKS_TOKEN')
                )
                installer = mock_workspace_installer(ws)

                # Execute workspace deployment (this calls SwitchInstaller.install())
                installer.install_switch()

                # Verify SwitchInstaller was called with workspace client
                mock_switch_installer_class.assert_called_once_with(ws)
                mock_installer.install.assert_called_once_with(
                    previous_job_id=None,
                    previous_switch_home=None
                )

                logger.info(f"Workspace deployment completed. Job ID: {mock_install_result.job_id}")

    def test_switch_config_update(self, tmp_path, testpypi_install, mock_install_result, minimal_workspace_installer):
        """Test config update writes job information to config.yml"""

        # Setup initial config (after PyPI installation)
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Write initial config from actual TestPyPI package
            package_path = testpypi_install["package_path"]
            source_config = package_path / "lsp" / "config.yml"
            if source_config.exists():
                import shutil
                shutil.copy(source_config, config_path)
            else:
                pytest.skip(f"Config file not found in TestPyPI package: {source_config}")

            # Execute config update using minimal installer fixture
            minimal_workspace_installer._update_switch_config(mock_install_result)

            # Verify config was updated with job information
            with config_path.open('r') as f:
                updated_config = yaml.safe_load(f)

            assert 'custom' in updated_config, "Config missing 'custom' section after update"
            custom = updated_config['custom']

            assert custom['job_id'] == mock_install_result.job_id
            assert custom['job_name'] == mock_install_result.job_name
            assert custom['switch_home'] == mock_install_result.switch_home

            # Verify original TestPyPI config was preserved
            assert updated_config['remorph']['name'] == 'switch'
            assert 'snowflake' in updated_config['remorph']['dialects']
            assert 'mysql' in updated_config['remorph']['dialects']  # TestPyPI has 8 dialects
            assert 'options' in updated_config  # TestPyPI includes options section

            logger.info(f"Config updated with job ID {mock_install_result.job_id}")

    def test_switch_with_previous_installation(self, tmp_path, testpypi_install, mock_install_result, mock_workspace_installer):
        """Test Switch installation with cleanup of previous installation"""
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Check credentials
        host = os.getenv('DATABRICKS_HOST')
        token = os.getenv('DATABRICKS_TOKEN')
        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env file.")

        # Setup environment with existing Switch installation
        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            # Create Switch package structure with previous installation info
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Create config with previous installation info (matching actual config.yml format)
            previous_config_data = {
                "remorph": {
                    "version": 1,
                    "name": "switch",
                    "dialects": ["mysql", "netezza", "oracle", "postgresql", 
                                 "redshift", "snowflake", "teradata", "tsql"],
                    "command_line": ["echo", "Switch uses Jobs API, not LSP"]
                },
                "options": {
                    "all": []  # Empty options list is fine for testing
                },
                "custom": {
                    "execution_type": "jobs_api",
                    "job_id": 999888777,  # Previous job ID
                    "job_name": "old-switch-job",
                    "switch_home": "/Workspace/Users/test/.lakebridge-switch-old"
                }
            }
            
            with config_path.open("w") as f:
                yaml.dump(previous_config_data, f)

            # Mock workspace deployment to return new job information
            with patch('switch.api.installer.SwitchInstaller') as mock_switch_installer_class:
                mock_installer = MagicMock()
                mock_installer.install.return_value = mock_install_result
                mock_switch_installer_class.return_value = mock_installer

                # Create WorkspaceInstaller
                from databricks.sdk import WorkspaceClient
                ws = WorkspaceClient(
                    host=os.getenv('DATABRICKS_HOST'),
                    token=os.getenv('DATABRICKS_TOKEN')
                )
                installer = mock_workspace_installer(ws)

                # Execute installation - should detect and pass previous installation info
                installer.install_switch()

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
                assert updated_config['custom']['switch_home'] == mock_install_result.switch_home

                logger.info(f"Previous installation (job_id=999888777) cleaned up and new installation completed (job_id={mock_install_result.job_id})")
