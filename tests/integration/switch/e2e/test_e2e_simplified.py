"""E2E tests for Switch transpiler integration with Lakebridge

This module provides end-to-end testing for Switch conversion functionality
through Lakebridge CLI.

Test Classes:
1. TestSwitchInstallationLifecycle: Installation workflow testing
   - test_installation_lifecycle: Test install → reinstall → uninstall operations
   
2. TestLakebridgeSwitchConversion: SQL and code conversion tests  
   - test_sql_basic_conversion_async: Snowflake → Databricks (async)
   - test_sql_advanced_conversion_async: TSQL → Databricks with advanced parameters (async)  
   - test_airflow_file_conversion_async: Airflow DAG → YAML file (async)

Environment Variables:
- DATABRICKS_HOST & DATABRICKS_TOKEN: Workspace credentials (required)
- LAKEBRIDGE_SWITCH_E2E=true: Enable E2E tests (disabled by default)
- LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true: Include sync execution tests
- LAKEBRIDGE_SWITCH_E2E_CATALOG: Custom catalog name (default: "main")
- LAKEBRIDGE_SWITCH_E2E_SCHEMA: Custom schema name (default: "default")
- LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true: Clean up existing resources before tests
- LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true: Keep resources after tests (for debugging)

Usage:
    # Quick async test (4 tests, ~2-3 minutes)
    LAKEBRIDGE_SWITCH_E2E=true pytest tests/integration/switch/e2e/test_e2e_simplified.py -v

    # Full sync test (wait for completion, ~15-20 minutes)  
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true pytest tests/integration/switch/e2e/test_e2e_simplified.py -v
"""
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from databricks.labs.lakebridge import cli
from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.contexts.application import ApplicationContext
from databricks.labs.lakebridge.install import TranspilerInstaller, WorkspaceInstaller
from switch.api.installer import SwitchInstaller
from switch.notebooks.pyscripts.types.builtin_prompt import BuiltinPrompt
from switch.testing.e2e_utils import SwitchCleanupManager, SwitchExamplesManager, SwitchSchemaManager

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
    catalog: str = "main"
    include_sync: bool = False
    clean_all_before: bool = False
    keep_after: bool = False
    warehouse_id: Optional[str] = None
    test_schema: Optional[str] = None  # Generated unique schema for test isolation

    @classmethod
    def from_env(cls) -> 'LakebridgeTestConfig':
        """Create configuration from environment variables"""
        return cls(
            catalog=os.getenv("LAKEBRIDGE_SWITCH_E2E_CATALOG", "main"),
            include_sync=os.getenv("LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC") == "true",
            clean_all_before=os.getenv("LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE") == "true",
            keep_after=os.getenv("LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER") == "true",
            warehouse_id=os.getenv("LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID")
        )


# Test configuration constants
BASE_DIR_PREFIX = ".lakebridge-switch-e2e"
EXAMPLES_DIR_SUFFIX = "-examples"
TRANSPILERS_PATH = ".databricks/labs/remorph-transpilers"
SWITCH_SUBDIR = "switch"
SCHEMA_PREFIX = "e2e_lakebridge_switch"  # Schema prefix for unique test schema generation


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("LAKEBRIDGE_SWITCH_E2E") != "true",
    reason="Switch E2E tests disabled. Set LAKEBRIDGE_SWITCH_E2E=true to enable"
)
class TestSwitchInstallationLifecycle:
    """Switch installation lifecycle tests (independent execution)"""

    @pytest.fixture(scope="class")
    def workspace_client(self):
        """Create Databricks workspace client for E2E tests"""
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        
        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        
        return WorkspaceClient(host=host, token=token)

    def test_installation_lifecycle(self, workspace_client):
        """Test complete installation workflow: install → reinstall → uninstall"""
        config = LakebridgeTestConfig.from_env()
        logger.info("Starting installation lifecycle test")

        # Cleanup before test if requested
        if config.clean_all_before:
            self._basic_cleanup(workspace_client)

        try:
            # Step 1: Install Switch
            logger.info("Step 1: Installing Switch transpiler")
            job_id1 = self._install_switch(workspace_client)
            assert job_id1 is not None, "Initial installation should succeed"
            logger.info(f"Switch installed successfully with job ID: {job_id1}")

            # Step 2: Reinstall (test cleanup of previous installation)
            logger.info("Step 2: Testing reinstallation with cleanup")
            job_id2 = self._install_switch(workspace_client)
            assert job_id2 is not None, "Reinstallation should succeed"
            assert job_id2 != job_id1, "New job ID should be different from old one"
            logger.info(f"Reinstallation successful. Old job {job_id1} cleaned up, new job {job_id2} created")

            # Step 3: Uninstall and verify
            logger.info("Step 3: Testing uninstallation")
            self._uninstall_switch(workspace_client)
            
            assert _get_switch_job_id() is None, "Switch config should be removed"
            assert not _verify_job_exists(workspace_client, job_id2), f"Job {job_id2} should be deleted"
            logger.info("Uninstallation successful")
        finally:
            # Final cleanup unless keeping for debugging
            if not config.keep_after:
                self._basic_cleanup(workspace_client)

    # ==================== Helper Methods for Installation Tests ====================

    def _install_switch(self, workspace_client: WorkspaceClient) -> Optional[int]:
        """Install Switch for testing"""
        logger.info("Installing Switch transpiler")
        
        try:
            # Install to workspace using Lakebridge installer
            app_context = ApplicationContext(workspace_client)
            installer = WorkspaceInstaller(
                app_context.workspace_client, app_context.prompts, app_context.installation,
                app_context.install_state, app_context.product_info, app_context.resource_configurator,
                app_context.workspace_installation,
            )
            installer.install_switch()
            logger.info("Switch installation completed")
        except Exception as e:
            logger.error(f"Switch installation failed: {e}")
            raise ValueError(f"Failed to install Switch: {e}") from e
        
        # Verify installation
        job_id = _get_switch_job_id()
        if job_id is None:
            raise ValueError("Switch job ID not found after installation")
        
        if not _verify_job_exists(workspace_client, job_id):
            raise ValueError(f"Switch job {job_id} not found in workspace")
        
        logger.info(f"Switch installed with job ID: {job_id}")
        return job_id

    def _uninstall_switch(self, workspace_client: WorkspaceClient):
        """Uninstall Switch completely"""
        job_id = _get_switch_job_id()
        if job_id:
            try:
                installer = SwitchInstaller(workspace_client)
                result = installer.uninstall(job_id=job_id)
                logger.info(f"Switch uninstalled: {result.message}")
            except Exception as e:
                logger.warning(f"SwitchInstaller.uninstall failed: {e}")
                workspace_client.jobs.delete(job_id)
                logger.info(f"Switch job {job_id} deleted")
        
        self._cleanup_local_files()

    def _basic_cleanup(self, workspace_client: WorkspaceClient):
        """Basic cleanup of Switch resources"""
        try:
            job_id = _get_switch_job_id()
            if job_id:
                try:
                    workspace_client.jobs.delete(job_id)
                    logger.info(f"Deleted job {job_id}")
                except NotFound:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to delete job {job_id}: {e}")
            
            self._cleanup_local_files()
        except Exception as e:
            logger.warning(f"Basic cleanup failed: {e}")

    def _cleanup_local_files(self):
        """Remove local Switch installation files"""
        transpilers_path = Path.home() / TRANSPILERS_PATH
        switch_path = transpilers_path / SWITCH_SUBDIR
        
        if switch_path.exists():
            shutil.rmtree(switch_path)
            logger.info("Removed local Switch files")


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("LAKEBRIDGE_SWITCH_E2E") != "true",
    reason="Switch E2E tests disabled. Set LAKEBRIDGE_SWITCH_E2E=true to enable"
)
class TestLakebridgeSwitchConversion:
    """Lakebridge-Switch conversion tests with shared setup/teardown"""

    # Class-level shared resources
    workspace_client = None
    cleanup_manager = None
    examples_manager = None
    schema_manager = None
    job_id = None
    examples_base_dir = None
    config = None

    @classmethod
    def setup_class(cls):
        """Setup Switch and examples once for all conversion tests"""
        # Get configuration
        cls.config = LakebridgeTestConfig.from_env()
        
        # Initialize workspace client
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        
        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        
        cls.workspace_client = WorkspaceClient(host=host, token=token)
        
        # Validate warehouse_id upfront (required for schema operations)
        if not cls.config.warehouse_id:
            pytest.skip("LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID required for schema operations")
        
        # Initialize schema manager and generate unique test schema
        cls.schema_manager = SwitchSchemaManager(cls.workspace_client, cls.config.warehouse_id)
        test_schema = cls.schema_manager.generate_unique_schema_name(SCHEMA_PREFIX)
        
        # Create test schema
        logger.info(f"Creating test schema: {cls.config.catalog}.{test_schema}")
        cls.schema_manager.create_schema(cls.config.catalog, test_schema)
        cls.config.test_schema = test_schema
        
        # Initialize utilities
        cls.cleanup_manager = SwitchCleanupManager(cls.workspace_client, cls.config.warehouse_id)
        cls.examples_manager = SwitchExamplesManager(cls.workspace_client)
        
        # Cleanup before test if requested
        if cls.config.clean_all_before:
            try:
                cls.cleanup_manager.cleanup_all_if_requested(cls.config.catalog, clean_before=True)
            except Exception as e:
                logger.warning(f"Pre-setup cleanup failed: {e}")
        
        # Install Switch once
        logger.info("Setting up Switch for all conversion tests")
        cls.job_id = cls._ensure_switch_installed_class(cls.workspace_client)
        
        # Upload examples once
        if cls.examples_manager:
            current_user = cls.workspace_client.current_user.me().user_name
            cls.examples_base_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}{EXAMPLES_DIR_SUFFIX}"
            logger.info(f"Uploading examples to {cls.examples_base_dir}")
            cls.examples_manager.upload_examples_to_workspace(
                base_dir=cls.examples_base_dir,
                sql_dialects=['snowflake', 'tsql'],
                code_types=[],
                include_workflow=True
            )

    @classmethod
    def teardown_class(cls):
        """Cleanup Switch and examples once after all conversion tests"""
        if not cls.config.keep_after:
            logger.info("Cleaning up after all conversion tests")
            
            # Comprehensive cleanup including potential tables
            if cls.cleanup_manager:
                try:
                    cls.cleanup_manager.cleanup_all_if_requested(cls.config.catalog, clean_before=False)
                except Exception as e:
                    logger.warning(f"Comprehensive cleanup failed: {e}")
                    # Fallback to basic job cleanup
                    try:
                        cls.cleanup_manager.cleanup_switch_jobs()
                    except Exception:
                        pass
            
            # Drop test schema (removes all tables/views)
            if cls.schema_manager and cls.config.test_schema:
                try:
                    logger.info(f"Dropping test schema: {cls.config.catalog}.{cls.config.test_schema}")
                    cls.schema_manager.drop_schema(cls.config.catalog, cls.config.test_schema)
                except Exception as e:
                    logger.warning(f"Failed to drop test schema: {e}")
            
            # Manual job cleanup if needed
            if cls.job_id and cls.workspace_client:
                try:
                    cls.workspace_client.jobs.delete(cls.job_id)
                except Exception:
                    pass
            
            # Clean up test directories
            if cls.examples_base_dir and cls.workspace_client:
                try:
                    cls.workspace_client.workspace.delete(cls.examples_base_dir, recursive=True)
                except Exception:
                    pass

    @classmethod
    def _ensure_switch_installed_class(cls, workspace_client: WorkspaceClient) -> int:
        """Ensure Switch is installed and return job ID"""
        job_id = _get_switch_job_id()
        if job_id and _verify_job_exists(workspace_client, job_id):
            logger.info(f"Switch already installed with job ID: {job_id}")
            return job_id
        
        logger.info("Installing Switch")
        return cls._install_switch_class(workspace_client)

    @classmethod
    def _install_switch_class(cls, workspace_client: WorkspaceClient) -> Optional[int]:
        """Install Switch for testing"""
        logger.info("Installing Switch transpiler")
        
        try:
            # Install to workspace using Lakebridge installer
            app_context = ApplicationContext(workspace_client)
            installer = WorkspaceInstaller(
                app_context.workspace_client, app_context.prompts, app_context.installation,
                app_context.install_state, app_context.product_info, app_context.resource_configurator,
                app_context.workspace_installation,
            )
            installer.install_switch()
            logger.info("Switch installation completed")
        except Exception as e:
            logger.error(f"Switch installation failed: {e}")
            raise ValueError(f"Failed to install Switch: {e}") from e
        
        # Verify installation
        job_id = _get_switch_job_id()
        if job_id is None:
            raise ValueError("Switch job ID not found after installation")
        
        if not _verify_job_exists(workspace_client, job_id):
            raise ValueError(f"Switch job {job_id} not found in workspace")
        
        logger.info(f"Switch installed with job ID: {job_id}")
        return job_id

    def test_sql_basic_conversion_async(self):
        """Basic SQL conversion (async): Snowflake → Databricks with basic parameters"""
        current_user = self.workspace_client.current_user.me().user_name
        
        try:
            # Execute conversion using shared resources
            input_dir = self.examples_manager.get_example_workspace_path(
                self.examples_base_dir, BuiltinPrompt.SNOWFLAKE
            )
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-basic-output-{int(time.time())}"
            
            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="snowflake",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema,
                wait_for_completion=self.config.include_sync,
                job_id=self.job_id
            )
            
            # Verify result
            self._verify_conversion_result(result, self.config.include_sync)
            logger.info("Basic SQL conversion test completed successfully")
        except Exception as e:
            logger.error(f"Basic SQL conversion test failed: {e}")
            raise

    def test_sql_advanced_conversion_async(self):
        """Advanced SQL conversion (async): TSQL → Databricks with advanced parameters"""
        current_user = self.workspace_client.current_user.me().user_name
        
        try:
            # Execute conversion with advanced parameters using shared resources
            input_dir = self.examples_manager.get_example_workspace_path(
                self.examples_base_dir, BuiltinPrompt.TSQL
            )
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-advanced-output-{int(time.time())}"
            sql_output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-advanced-sql-{int(time.time())}"
            
            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="tsql",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema,
                wait_for_completion=self.config.include_sync,
                job_id=self.job_id,
                extra_options={
                    "log_level": "DEBUG",
                    "comment_lang": "Japanese",
                    "token_count_threshold": "25000",
                    "sql_output_dir": sql_output_dir,
                    "max_fix_attempts": "1"
                }
            )
            
            # Verify result
            self._verify_conversion_result(result, self.config.include_sync)
            logger.info("Advanced SQL conversion test completed successfully")
        except Exception as e:
            logger.error(f"Advanced SQL conversion test failed: {e}")
            raise

    def test_airflow_file_conversion_async(self):
        """Airflow DAG to YAML file conversion (async) with generic source format"""
        current_user = self.workspace_client.current_user.me().user_name
        
        try:
            # Execute conversion using shared resources
            input_dir = self.examples_manager.get_example_workspace_path(
                self.examples_base_dir, BuiltinPrompt.AIRFLOW
            )
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-airflow-output-{int(time.time())}"
            
            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="airflow",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema,
                wait_for_completion=self.config.include_sync,
                job_id=self.job_id,
                extra_options={
                    "source_format": "generic",
                    "target_type": "file",
                    "output_extension": ".yml"
                }
            )
            
            # Verify result
            self._verify_conversion_result(result, self.config.include_sync)
            logger.info("Airflow file conversion test completed successfully")
        except Exception as e:
            logger.error(f"Airflow file conversion test failed: {e}")
            raise

    # ==================== Helper Methods for Conversion Tests ====================

    def _execute_conversion(self, workspace_client: WorkspaceClient, 
                           source_dialect: str, input_source: str, output_folder: str,
                           catalog_name: str, schema_name: str, 
                           wait_for_completion: bool = False,
                           job_id: Optional[int] = None,
                           extra_options: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Execute Lakebridge conversion"""
        # Prepare transpiler options
        transpiler_options = {}
        if wait_for_completion:
            transpiler_options["wait_for_completion"] = "true"
        
        if extra_options:
            transpiler_options.update(extra_options)
        
        # Create TranspileConfig
        transpiler_config_path = str(_get_switch_config_path())
        transpile_config = TranspileConfig(
            transpiler_config_path=transpiler_config_path,
            source_dialect=source_dialect,
            input_source=input_source,
            output_folder=output_folder,
            catalog_name=catalog_name,
            schema_name=schema_name,
            skip_validation=True,
            transpiler_options=transpiler_options
        )
        
        # Execute via Lakebridge CLI
        ctx = ApplicationContext(workspace_client)
        return cli._execute_switch_directly(ctx, transpile_config)

    def _verify_conversion_result(self, result: List[Dict[str, Any]], is_sync: bool):
        """Verify conversion result"""
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


# ==================== Common Helper Functions ====================

def _get_switch_job_id() -> Optional[int]:
    """Get Switch job ID from config"""
    switch_config = TranspilerInstaller.read_switch_config()
    if switch_config:
        return switch_config.get('custom', {}).get('job_id')
    return None


def _verify_job_exists(workspace_client: WorkspaceClient, job_id: int) -> bool:
    """Check if Databricks job exists"""
    try:
        job = workspace_client.jobs.get(job_id)
        return job is not None
    except NotFound:
        return False
    except Exception as e:
        logger.error(f"Error checking job {job_id}: {e}")
        return False


def _get_switch_config_path() -> Path:
    """Get the Switch config path"""
    transpilers_path = Path.home() / TRANSPILERS_PATH
    return transpilers_path / SWITCH_SUBDIR / "lib" / "config.yml"
