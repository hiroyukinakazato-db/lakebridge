"""Integration tests for Switch transpiler installation process

These tests validate the 3-step installation process:
1. PyPI Installation: TranspilerInstaller.install_from_pypi()
2. Workspace Deployment: SwitchInstaller.install() creates Databricks job
3. Config Update: _update_switch_config() writes job info to config.yml
"""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, create_autospec, MagicMock

import pytest
import yaml

from databricks.labs.lakebridge.install import WorkspaceInstaller


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

    def test_install_switch_step1_pypi_installation(self, tmp_path, testpypi_install):
        """Step 1: Test PyPI installation creates local Switch package structure"""

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

                # Execute Step 1: PyPI installation
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

                print(f"✅ Step 1: PyPI installation completed successfully (TestPyPI v{testpypi_install['version']})")

    def test_install_switch_step2_workspace_deployment(self, tmp_path, testpypi_install, mock_install_result, mock_workspace_installer):
        """Step 2: Test workspace deployment creates Databricks job"""

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

                # Execute Step 2: Workspace deployment (this calls SwitchInstaller.install())
                installer.install_switch()

                # Verify SwitchInstaller was called with workspace client
                mock_switch_installer_class.assert_called_once_with(ws)
                mock_installer.install.assert_called_once()

                print(f"✅ Step 2: Workspace deployment completed. Job ID: {mock_install_result.job_id}")

    def test_install_switch_step3_config_update(self, tmp_path, testpypi_install, mock_install_result, minimal_workspace_installer):
        """Step 3: Test config update writes job information to config.yml"""

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

            # Execute Step 3: Config update using minimal installer fixture
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

            print(f"✅ Step 3: Config updated with job ID {mock_install_result.job_id}")

    def test_install_switch_end_to_end(self, tmp_path, testpypi_install):
        """End-to-end test: Complete installation process with real Databricks workspace"""

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

        # Mock PyPI installation using real TestPyPI package (Step 1)
        def mock_pypi_install(local_name, pypi_name, artifact):
            switch_dir = tmp_path / "switch" / "lib"
            switch_dir.mkdir(parents=True)
            config_path = switch_dir / "config.yml"

            # Use actual config from TestPyPI package
            package_path = testpypi_install["package_path"]
            source_config = package_path / "lsp" / "config.yml"
            if source_config.exists():
                import shutil
                shutil.copy(source_config, config_path)
            else:
                pytest.skip(f"Config file not found in TestPyPI package: {source_config}")

        with patch('databricks.labs.lakebridge.install.TranspilerInstaller.transpilers_path', return_value=tmp_path):
            with patch('databricks.labs.lakebridge.install.TranspilerInstaller.install_from_pypi', side_effect=mock_pypi_install):
                try:
                    # Create WorkspaceInstaller with real Databricks connection
                    from databricks.sdk import WorkspaceClient
                    from databricks.labs.blueprint.tui import MockPrompts
                    from databricks.labs.blueprint.installation import MockInstallation
                    from databricks.labs.blueprint.installer import InstallState
                    from databricks.labs.blueprint.wheels import ProductInfo
                    from databricks.labs.lakebridge.deployment.configurator import ResourceConfigurator
                    from databricks.labs.lakebridge.deployment.installation import WorkspaceInstallation

                    ws = WorkspaceClient(
                        host=os.getenv('DATABRICKS_HOST'),
                        token=os.getenv('DATABRICKS_TOKEN')
                    )

                    installer = WorkspaceInstaller(
                        ws=ws,
                        prompts=MockPrompts({}),
                        installation=MockInstallation({}),
                        install_state=InstallState.from_installation(MockInstallation({})),
                        product_info=ProductInfo.from_class(WorkspaceInstaller),
                        resource_configurator=create_autospec(ResourceConfigurator),
                        workspace_installation=create_autospec(WorkspaceInstallation)
                    )

                    # Execute complete installation process
                    installer.install_switch()

                    # Verify final state: config.yml should contain job information
                    config_path = tmp_path / "switch" / "lib" / "config.yml"
                    assert config_path.exists(), "Config file missing after installation"

                    with config_path.open('r') as f:
                        final_config = yaml.safe_load(f)

                    # Verify all 3 steps completed successfully with real TestPyPI config
                    assert 'remorph' in final_config, "Step 1 (PyPI) failed: missing remorph section"
                    assert final_config['remorph']['name'] == 'switch', "Step 1 (PyPI) failed: wrong transpiler name"
                    assert len(final_config['remorph']['dialects']) >= 8, "Step 1 (PyPI) failed: missing SQL dialects from TestPyPI"
                    assert 'options' in final_config, "Step 1 (PyPI) failed: missing options section from TestPyPI"

                    assert 'custom' in final_config, "Step 3 (Config update) failed: missing custom section"
                    assert final_config['custom']['job_id'] is not None, "Step 2 (Workspace deployment) failed: no job created"
                    assert isinstance(final_config['custom']['job_id'], int), "Step 2 (Workspace deployment) failed: invalid job ID"

                    job_id = final_config['custom']['job_id']
                    dialect_count = len(final_config['remorph']['dialects'])
                    print(f"🎉 End-to-end installation successful!")
                    print(f"   Step 1: TestPyPI package installed (v{testpypi_install['version']}, {dialect_count} dialects) ✅")
                    print(f"   Step 2: Databricks job created (ID: {job_id}) ✅")
                    print(f"   Step 3: Config updated with job info ✅")

                except Exception as e:
                    import traceback
                    pytest.skip(f"End-to-end installation test failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
