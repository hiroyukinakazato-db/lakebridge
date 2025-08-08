"""Simplified E2E tests for Switch transpiler integration with Lakebridge

This module provides focused, efficient end-to-end testing for Switch conversion functionality
through Lakebridge CLI, utilizing Switch's robust testing utilities for maximum reliability.

Test Coverage:
1. Installation Lifecycle: Test install, reinstall, and uninstall operations  
2. Basic SQL Conversion: Snowflake → Databricks with basic parameters using real Switch examples
3. Advanced SQL Conversion: TSQL → Databricks with advanced parameters (Japanese comments, DEBUG logging, sql_output_dir) using real Switch examples
4. Airflow File Conversion: Airflow DAG → YAML file with generic source format and file output using real Switch examples

Key Benefits:
- Uses real Switch example files for comprehensive testing
- Leverages proven Switch testing utilities for maximum reliability
- Unified configuration management via environment variables
- Simplified test logic with comprehensive error handling
- Async/sync execution controlled by environment variables

Environment Variables:
- LAKEBRIDGE_SWITCH_E2E=true: Enable E2E tests (disabled by default)
- LAKEBRIDGE_SWITCH_E2E_PYPI_SOURCE: "testpypi" (default) or "pypi"
- LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true: Include sync execution tests
- LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true: Clean up existing resources before tests
- LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true: Keep resources after tests (for debugging)
- LAKEBRIDGE_SWITCH_E2E_CATALOG: Custom catalog name (default: "remorph")
- LAKEBRIDGE_SWITCH_E2E_SCHEMA: Custom schema name (default: "transpiler")
- DATABRICKS_HOST & DATABRICKS_TOKEN: Workspace credentials

Usage:
    # Quick async test (4 tests, ~2-3 minutes)
    LAKEBRIDGE_SWITCH_E2E=true pytest tests/integration/switch/e2e/test_e2e.py -v

    # Full sync test (wait for completion, ~15-20 minutes)  
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true pytest tests/integration/switch/e2e/test_e2e.py -v
"""
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.contexts.application import ApplicationContext
from databricks.labs.lakebridge.install import TranspilerInstaller, WorkspaceInstaller
from databricks.labs.lakebridge import cli

# Import Switch testing utilities
try:
    from switch.testing.e2e_utils import SwitchCleanupManager, SwitchExamplesManager
    from switch.api.installer import SwitchInstaller
    from switch.notebooks.pyscripts.types.builtin_prompt import BuiltinPrompt
    SWITCH_UTILITIES_AVAILABLE = True
except ImportError:
    SWITCH_UTILITIES_AVAILABLE = False

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LakebridgeTestConfig:
    """Unified configuration for Lakebridge E2E tests"""
    pypi_source: str = "testpypi"
    catalog: str = "main"
    schema: str = "default"
    include_sync: bool = False
    clean_all_before: bool = False
    keep_after: bool = False

    @classmethod
    def from_env(cls) -> 'LakebridgeTestConfig':
        """Create configuration from environment variables"""
        return cls(
            pypi_source=os.getenv("LAKEBRIDGE_SWITCH_E2E_PYPI_SOURCE", "testpypi"),
            catalog=os.getenv("LAKEBRIDGE_SWITCH_E2E_CATALOG", "main"),
            schema=os.getenv("LAKEBRIDGE_SWITCH_E2E_SCHEMA", "default"),
            include_sync=os.getenv("LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC") == "true",
            clean_all_before=os.getenv("LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE") == "true",
            keep_after=os.getenv("LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER") == "true"
        )


class TestConstants:
    """Test configuration constants"""
    BASE_DIR_PREFIX = ".lakebridge-switch-e2e"
    EXAMPLES_DIR_SUFFIX = "-examples"
    TRANSPILERS_PATH = ".databricks/labs/remorph-transpilers"
    SWITCH_SUBDIR = "switch"
    CONFIG_FILE = "config.yml"
    LIB_SUBDIR = "lib"
    LSP_SUBDIR = "lsp"
    TESTPYPI_URL = "https://test.pypi.org/simple/"
    SWITCH_PACKAGE = "databricks-switch-plugin"


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("LAKEBRIDGE_SWITCH_E2E") != "true",
    reason="Switch E2E tests disabled. Set LAKEBRIDGE_SWITCH_E2E=true to enable"
)
class TestLakebridgeSwitchIntegration:
    """Simplified Lakebridge-Switch integration tests"""

    # ==================== Test Fixtures & Setup ====================

    @pytest.fixture(scope="class")
    def workspace_client(self):
        """Create Databricks workspace client for E2E tests"""
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")

        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")

        return WorkspaceClient(host=host, token=token)

    @pytest.fixture(scope="class")
    def test_config(self):
        """Load test configuration from environment"""
        return LakebridgeTestConfig.from_env()

    @pytest.fixture(scope="class")
    def cleanup_manager(self, workspace_client):
        """Initialize Switch cleanup manager if available"""
        if not SWITCH_UTILITIES_AVAILABLE:
            return None

        # Use a minimal warehouse ID (will be provided by environment or skipped)
        warehouse_id = os.getenv("LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID")
        if not warehouse_id:
            logger.warning("No warehouse ID provided - some cleanup features may be limited")
            return None

        return SwitchCleanupManager(workspace_client, warehouse_id)

    @pytest.fixture(scope="class")
    def current_user(self, workspace_client):
        """Cache current user for the test session"""
        return workspace_client.current_user.me().user_name

    @pytest.fixture(scope="class")
    def switch_installation(self, workspace_client, test_config):
        """Install Switch once for all conversion tests in the class"""
        # This fixture is used by conversion tests but not by test_installation_lifecycle
        # The fixture will install Switch and return the job_id
        logger.info("Installing Switch once for all conversion tests")
        job_id = self._install_switch_for_test(workspace_client, test_config.pypi_source)
        logger.info(f"Switch installed with job ID: {job_id}")
        return job_id

    @pytest.fixture(scope="class")
    def examples_manager(self, workspace_client, test_config, current_user):
        """Initialize Switch examples manager and upload examples"""
        if not SWITCH_UTILITIES_AVAILABLE:
            pytest.skip("Switch utilities not available - cannot use examples manager")

        manager = SwitchExamplesManager(workspace_client)
        base_dir = f"/Workspace/Users/{current_user}/{TestConstants.BASE_DIR_PREFIX}{TestConstants.EXAMPLES_DIR_SUFFIX}"

        # Upload examples for the dialects we'll test
        logger.info(f"Uploading Switch examples to {base_dir}")
        manager.upload_examples_to_workspace(
            base_dir=base_dir,
            sql_dialects=['snowflake', 'tsql'],  # Only upload what we need
            code_types=[],  # No code examples needed
            include_workflow=True  # For Airflow test
        )
        logger.info("Switch examples uploaded successfully")

        return manager, base_dir

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, request, workspace_client, test_config, cleanup_manager):
        """Ensure clean state before and after tests"""
        logger.info(f"Cleanup settings: clean_all_before={test_config.clean_all_before}, keep_after={test_config.keep_after}")

        # Skip cleanup for conversion tests that depend on switch_installation fixture
        conversion_test_names = ["test_sql_basic_conversion", "test_sql_advanced_conversion", "test_airflow_file_conversion"]
        skip_cleanup = any(test_name in request.node.name for test_name in conversion_test_names)
        
        if skip_cleanup:
            logger.info("Skipping cleanup for conversion test that uses switch_installation fixture")
        else:
            # Setup: Perform cleanup before tests if requested
            if test_config.clean_all_before and cleanup_manager:
                logger.info("Performing comprehensive cleanup before tests")
                cleanup_manager.cleanup_all_if_requested(test_config.catalog, clean_before=True)
            elif test_config.clean_all_before:
                logger.info("Performing basic cleanup before tests")
                self._basic_cleanup(workspace_client)

        yield

        # Teardown: Clean up test artifacts unless keeping for debugging
        if not test_config.keep_after and not skip_cleanup:
            logger.info("Cleaning up resources after test")
            if cleanup_manager:
                cleanup_manager.cleanup_switch_jobs()
            else:
                self._basic_cleanup(workspace_client)

    def _basic_cleanup(self, workspace_client):
        """Basic cleanup when Switch utilities are not available"""
        try:
            # Clean up any existing Switch configuration
            job_id = self._get_switch_job_id_from_config()
            if job_id:
                logger.info(f"Cleaning up Switch job {job_id}")
                try:
                    workspace_client.jobs.delete(job_id)
                except NotFound:
                    logger.info(f"Job {job_id} already deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete job {job_id}: {e}")

            # Clean up local Switch files
            self._cleanup_local_switch_files()
        except Exception as e:
            logger.warning(f"Basic cleanup failed: {e}")

    def _get_switch_job_id_from_config(self):
        """Get Switch job ID from config"""
        switch_config = TranspilerInstaller.read_switch_config()
        if switch_config:
            return switch_config.get('custom', {}).get('job_id')
        return None

    def _cleanup_local_switch_files(self):
        """Remove local Switch installation files"""
        transpilers_path = Path.home() / TestConstants.TRANSPILERS_PATH
        switch_path = transpilers_path / TestConstants.SWITCH_SUBDIR

        if switch_path.exists():
            import shutil
            shutil.rmtree(switch_path)
            logger.info("Removed local Switch files")

    # ==================== Main Test Methods ====================

    def test_installation_lifecycle(self, workspace_client, test_config):
        """Test complete installation workflow: install → reinstall → uninstall"""
        logger.info(f"Starting installation lifecycle test with PyPI source: {test_config.pypi_source}")

        try:
            # Step 1: Install Switch
            logger.info("Step 1: Installing Switch transpiler")
            job_id1 = self._install_switch_for_test(workspace_client, test_config.pypi_source)
            assert job_id1 is not None, "Initial installation should succeed"
            logger.info(f"Switch installed successfully with job ID: {job_id1}")

            # Step 2: Reinstall (test cleanup of previous installation)
            logger.info("Step 2: Testing reinstallation with cleanup")
            job_id2 = self._install_switch_for_test(workspace_client, test_config.pypi_source)
            assert job_id2 is not None, "Reinstallation should succeed"
            assert job_id2 != job_id1, "New job ID should be different from old one"
            logger.info(f"Reinstallation successful. Old job {job_id1} cleaned up, new job {job_id2} created")

            # Step 3: Uninstall and verify
            logger.info("Step 3: Testing uninstallation")
            self._uninstall_switch_completely(workspace_client)

            assert self._get_switch_job_id_from_config() is None, "Switch config should be removed"
            assert not self._verify_job_exists(workspace_client, job_id2), f"Job {job_id2} should be deleted"
            logger.info("Uninstallation successful")
        finally:
            # Ensure cleanup after this test since conversion tests need fresh Switch installation
            logger.info("Final cleanup after installation lifecycle test")
            self._basic_cleanup(workspace_client)

    def test_sql_basic_conversion(self, workspace_client, test_config, examples_manager, switch_installation, current_user):
        """Basic SQL conversion: Snowflake → Databricks with basic parameters"""
        manager, base_dir = examples_manager
        input_dir = manager.get_example_workspace_path(base_dir, BuiltinPrompt.SNOWFLAKE)

        result = self._execute_lakebridge_conversion(
            workspace_client=workspace_client,
            config=test_config,
            test_type="basic",
            input_source=input_dir,
            test_data={
                "source_dialect": "snowflake",
                "concurrency": 2,
                "max_fix_attempts": 0,
                "log_level": "INFO",
                "comment_lang": "English"
            },
            job_id=switch_installation,
            current_user=current_user
        )

        self._verify_conversion_result(result, test_config.include_sync)
        logger.info("Basic SQL conversion test completed successfully")

    def test_sql_advanced_conversion(self, workspace_client, test_config, examples_manager, switch_installation, current_user):
        """Advanced SQL conversion: TSQL → Databricks with advanced parameters"""
        manager, base_dir = examples_manager
        input_dir = manager.get_example_workspace_path(base_dir, BuiltinPrompt.TSQL)
        sql_output_path = f"/Workspace/Users/{current_user}/{TestConstants.BASE_DIR_PREFIX}-advanced/sql_output_{int(time.time())}"

        result = self._execute_lakebridge_conversion(
            workspace_client=workspace_client,
            config=test_config,
            test_type="advanced",
            input_source=input_dir,
            test_data={
                "source_dialect": "tsql",
                "concurrency": 2,
                "max_fix_attempts": 1,
                "log_level": "DEBUG",
                "comment_lang": "Japanese",
                "token_count_threshold": 25000,
                "sql_output_dir": sql_output_path
            },
            job_id=switch_installation,
            current_user=current_user
        )

        self._verify_conversion_result(result, test_config.include_sync)
        logger.info("Advanced SQL conversion test completed successfully")

    def test_airflow_file_conversion(self, workspace_client, test_config, examples_manager, switch_installation, current_user):
        """Airflow DAG to YAML file conversion with generic source format"""
        manager, base_dir = examples_manager
        input_dir = manager.get_example_workspace_path(base_dir, BuiltinPrompt.AIRFLOW)

        result = self._execute_lakebridge_conversion(
            workspace_client=workspace_client,
            config=test_config,
            test_type="airflow",
            input_source=input_dir,
            test_data={
                "source_dialect": "airflow",
                "source_format": "generic",
                "target_type": "file",
                "output_extension": ".yml",
                "concurrency": 2,
                "max_fix_attempts": 0,
                "log_level": "INFO",
                "comment_lang": "English"
            },
            job_id=switch_installation,
            current_user=current_user
        )

        self._verify_conversion_result(result, test_config.include_sync)
        logger.info("Airflow file conversion test completed successfully")

    # ==================== Helper Methods ====================

    def _execute_lakebridge_conversion(self, workspace_client: WorkspaceClient, 
                                     config: LakebridgeTestConfig, test_type: str, 
                                     input_source: str, test_data: Dict[str, Any],
                                     job_id: Optional[int] = None,
                                     current_user: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute Lakebridge conversion with unified interface"""
        logger.info(f"Executing {test_type} conversion test")

        # Step 1: Use provided job_id or install Switch if needed
        if not job_id:
            job_id = self._install_switch_for_test(workspace_client, config.pypi_source)
        logger.info(f"Switch ready with job ID: {job_id}")

        # Step 2: Setup output directory only (input already exists from examples)
        if not current_user:
            current_user = workspace_client.current_user.me().user_name
        output_dir = self._setup_output_directory(workspace_client, test_type, current_user)

        try:
            # Step 3: Execute conversion via Lakebridge CLI
            conversion_config = {
                "source_dialect": test_data["source_dialect"],
                "input_source": input_source,
                "output_folder": output_dir,
                "catalog_name": config.catalog,
                "schema_name": config.schema,
                "wait_for_completion": config.include_sync
            }

            # Add optional parameters
            optional_params = ["source_format", "target_type", "output_extension", 
                             "comment_lang", "token_count_threshold", "sql_output_dir", 
                             "concurrency", "max_fix_attempts", "log_level"]
            for param in optional_params:
                if param in test_data:
                    conversion_config[param] = test_data[param]

            result = self._run_lakebridge_transpile(workspace_client, conversion_config)
            logger.info(f"{test_type.capitalize()} conversion executed successfully")
            return result

        finally:
            # Step 4: Cleanup output directory
            self._cleanup_test_workspace(workspace_client, output_dir)

    def _run_lakebridge_transpile(self, workspace_client: WorkspaceClient, 
                                config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute Lakebridge transpile command via direct API call"""
        transpiler_config_path = str(self._get_switch_config_path())

        # Prepare transpiler options
        transpiler_options = {}
        if config.get("wait_for_completion"):
            transpiler_options["wait_for_completion"] = "true"

        # Add Switch-specific parameters
        switch_params = ["source_format", "target_type", "output_extension",
                        "comment_lang", "token_count_threshold", "sql_output_dir", 
                        "concurrency", "max_fix_attempts", "log_level"]
        for param in switch_params:
            if param in config:
                value = config[param]
                if isinstance(value, int):
                    transpiler_options[param] = str(value)
                else:
                    transpiler_options[param] = value

        # Create TranspileConfig
        transpile_config = TranspileConfig(
            transpiler_config_path=transpiler_config_path,
            source_dialect=config["source_dialect"],
            input_source=config["input_source"], 
            output_folder=config["output_folder"],
            catalog_name=config["catalog_name"],
            schema_name=config["schema_name"],
            skip_validation=True,  # Skip path validation for workspace paths
            transpiler_options=transpiler_options
        )

        # Execute via Lakebridge CLI
        ctx = ApplicationContext(workspace_client)
        return cli._execute_switch_directly(ctx, transpile_config)

    def _verify_conversion_result(self, result: List[Dict[str, Any]], is_sync: bool):
        """Verify conversion result with standardized checks"""
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Result should contain exactly one item"

        result_item = result[0]
        assert result_item["transpiler"] == "switch", "Transpiler should be 'switch'"
        assert "job_id" in result_item, "Result should contain job_id"

        if is_sync:
            assert "state" in result_item, "Sync result should contain state"
            assert result_item["state"] in ["TERMINATED", "SKIPPED"], f"Invalid state: {result_item['state']}"
            logger.info(f"Sync execution verified: state={result_item['state']}")
        else:
            assert "run_id" in result_item, "Async result should contain run_id"
            assert "run_url" in result_item, "Async result should contain run_url"
            logger.info(f"Async execution verified: run_id={result_item['run_id']}")

    def _install_switch_for_test(self, workspace_client: WorkspaceClient, pypi_source: str) -> Optional[int]:
        """Install Switch for testing (install + verify)"""
        logger.info(f"Setting up Switch with PyPI source: {pypi_source}")

        try:
            if pypi_source == "pypi":
                # Use standard installation
                output = self._run_cli_command("install-transpile")
                logger.info("PyPI installation completed")
            else:
                # Use TestPyPI installation
                self._install_switch_from_testpypi()
                logger.info("TestPyPI installation completed")

        except Exception as e:
            logger.error(f"Switch installation failed: {e}")
            raise ValueError(f"Failed to install Switch: {e}") from e

        # Verify installation
        job_id = self._get_switch_job_id_from_config()
        if job_id is None:
            raise ValueError("Switch job ID not found after installation")

        if not self._verify_job_exists(workspace_client, job_id):
            raise ValueError(f"Switch job {job_id} not found in workspace")

        logger.info(f"Switch setup completed with job ID: {job_id}")
        return job_id

    def _install_switch_from_testpypi(self):
        """Install Switch from TestPyPI with proper setup"""
        # Install package from TestPyPI
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-i", TestConstants.TESTPYPI_URL,
            TestConstants.SWITCH_PACKAGE, 
            "--force-reinstall", "--no-deps"
        ], check=True)

        # Setup config and install to workspace
        self._setup_switch_config_for_testpypi()
        self._install_switch_via_api_with_mock()

    def _setup_switch_config_for_testpypi(self):
        """Set up Switch config file in expected location"""
        import switch
        import shutil

        switch_package_dir = Path(switch.__file__).parent
        source_config = switch_package_dir.parent / TestConstants.LSP_SUBDIR / TestConstants.CONFIG_FILE

        transpilers_path = Path.home() / TestConstants.TRANSPILERS_PATH
        switch_path = transpilers_path / TestConstants.SWITCH_SUBDIR / TestConstants.LIB_SUBDIR
        switch_path.mkdir(parents=True, exist_ok=True)
        target_config = switch_path / TestConstants.CONFIG_FILE

        shutil.copy2(source_config, target_config)
        logger.info(f"Copied Switch config to {target_config}")

    def _install_switch_via_api_with_mock(self):
        """Install Switch using API (requires Switch package to be pre-installed)"""
        ws = WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"),
            token=os.getenv("DATABRICKS_TOKEN")
        )
        app_context = ApplicationContext(ws)
        installer = WorkspaceInstaller(
            app_context.workspace_client, app_context.prompts, app_context.installation,
            app_context.install_state, app_context.product_info, app_context.resource_configurator,
            app_context.workspace_installation,
        )

        installer.install_switch()

    def _uninstall_switch_completely(self, workspace_client: WorkspaceClient):
        """Complete Switch uninstallation"""
        try:
            job_id = self._get_switch_job_id_from_config()
            if job_id:
                if SWITCH_UTILITIES_AVAILABLE:
                    try:
                        installer = SwitchInstaller(workspace_client)
                        result = installer.uninstall(job_id=job_id)
                        logger.info(f"Switch uninstalled: {result.message}")
                    except Exception as e:
                        logger.warning(f"SwitchInstaller.uninstall failed: {e}")
                        # Fallback to direct job deletion
                        workspace_client.jobs.delete(job_id)
                        logger.info(f"Switch job {job_id} deleted directly")
                else:
                    # Fallback to direct job deletion when utilities not available
                    workspace_client.jobs.delete(job_id)
                    logger.info(f"Switch job {job_id} deleted directly")

            self._cleanup_local_switch_files()

        except Exception as e:
            logger.error(f"Uninstallation failed: {e}")
            raise

    def _setup_output_directory(self, workspace_client: WorkspaceClient, test_type: str, current_user: str) -> str:
        """Set up output directory for test results"""
        test_base_dir = f"/Workspace/Users/{current_user}/{TestConstants.BASE_DIR_PREFIX}-{test_type}"
        output_dir = f"{test_base_dir}/output_{int(time.time())}"

        # Create output directory
        workspace_client.workspace.mkdirs(output_dir)

        logger.debug(f"Output directory created: {output_dir}")
        return output_dir

    def _cleanup_test_workspace(self, workspace_client: WorkspaceClient, *paths):
        """Clean up test workspace directories"""
        for path in paths:
            if not path:
                continue
            try:
                parent_dir = str(Path(path).parent)
                workspace_client.workspace.delete(parent_dir, recursive=True)
                logger.debug(f"Cleaned up: {parent_dir}")
            except Exception as e:
                logger.debug(f"Failed to clean up {path}: {e}")

    def _run_cli_command(self, *args):
        """Execute lakebridge CLI command"""
        cmd = ["databricks", "labs", "lakebridge"] + list(args)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env={**os.environ, "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"), 
                 "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN")}
        )

        if result.returncode != 0:
            logger.error(f"CLI command failed: {' '.join(args)}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"CLI command failed: {' '.join(args)}")

        return result.stdout

    def _verify_job_exists(self, workspace_client: WorkspaceClient, job_id: int) -> bool:
        """Check if Databricks job exists"""
        try:
            job = workspace_client.jobs.get(job_id)
            return job is not None
        except NotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking job {job_id}: {e}")
            return False

    def _get_switch_config_path(self) -> Path:
        """Get the Switch config path"""
        transpilers_path = Path.home() / TestConstants.TRANSPILERS_PATH
        return transpilers_path / TestConstants.SWITCH_SUBDIR / TestConstants.LIB_SUBDIR / TestConstants.CONFIG_FILE
