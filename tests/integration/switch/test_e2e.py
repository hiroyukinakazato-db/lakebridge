"""E2E tests for Switch transpiler integration with Lakebridge

This module provides end-to-end testing for Switch conversion functionality
through Lakebridge CLI.

Test Classes:
1. TestSwitchInstallationLifecycle: Installation workflow testing
   - test_installation_lifecycle: Test install → reinstall → uninstall operations

2. TestLakebridgeSwitchConversion: SQL and code conversion tests  
   - test_sql_basic_conversion: Snowflake → Databricks
   - test_sql_advanced_conversion: TSQL → Databricks with advanced parameters  
   - test_airflow_file_conversion: Airflow DAG → YAML file

Environment Variables:
- DATABRICKS_HOST & DATABRICKS_TOKEN: Workspace credentials (required)
- LAKEBRIDGE_SWITCH_E2E=true: Enable E2E tests (disabled by default)
- LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true: Clean up existing resources before tests
- LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true: Keep resources after tests (for debugging)
- LAKEBRIDGE_SWITCH_E2E_CATALOG: Custom catalog name (default: "main")
- LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID: SQL warehouse ID for schema operations (required)

Note: Schema names are automatically generated with timestamp + random suffix to prevent conflicts.

Usage:
    # Quick async test (4 tests, ~2-3 minutes)
    LAKEBRIDGE_SWITCH_E2E=true pytest tests/integration/switch/test_e2e.py -v
"""
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import ImportFormat

from databricks.labs.lakebridge import cli
from databricks.labs.lakebridge.config import TranspileConfig
from databricks.labs.lakebridge.contexts.application import ApplicationContext
from databricks.labs.lakebridge.deployment.installation import WorkspaceInstallation
from databricks.labs.lakebridge.transpiler.installers import SwitchInstaller
from databricks.labs.lakebridge.transpiler.repository import TranspilerRepository
from switch.testing.e2e_utils import SwitchCleanupManager, SwitchSchemaManager

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
    clean_all_before: bool = False
    keep_after: bool = False
    warehouse_id: Optional[str] = None
    test_schema: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'LakebridgeTestConfig':
        """Create configuration from environment variables"""
        return cls(
            catalog=os.getenv("LAKEBRIDGE_SWITCH_E2E_CATALOG", "main"),
            clean_all_before=os.getenv("LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE") == "true",
            keep_after=os.getenv("LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER") == "true",
            warehouse_id=os.getenv("LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID")
        )


# Test configuration constants
BASE_DIR_PREFIX = ".e2e-lakebridge-switch"
EXAMPLES_DIR_SUFFIX = "-examples"
TRANSPILERS_PATH = ".databricks/labs/remorph-transpilers"
SWITCH_SUBDIR = "switch"
SCHEMA_PREFIX = "e2e_lakebridge_switch"


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
        logger.info("Starting installation lifecycle test")

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

    # ==================== Helper Methods for Installation Tests ====================

    def _install_switch(self, workspace_client: WorkspaceClient) -> Optional[int]:
        """Install Switch for testing"""
        logger.info("Installing Switch transpiler")

        try:
            # Install using new SwitchInstaller pattern
            transpiler_repository = TranspilerRepository.user_home()
            switch_installer = SwitchInstaller(transpiler_repository, workspace_client)
            switch_installer.install()
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
        """Uninstall Switch using main WorkspaceInstallation method"""
        try:
            # Create WorkspaceInstallation instance
            workspace_installation = WorkspaceInstallation(
                ws=workspace_client,
                prompts=None,
                installation=None,
                recon_deployment=None,
                product_info=None,
                upgrades=None
            )
            # Call the main uninstall method
            workspace_installation._uninstall_switch()
            logger.info("Switch uninstalled using main WorkspaceInstallation method")
        except Exception as e:
            logger.error(f"Failed to use main uninstall method: {e}")
            raise ValueError(f"Failed to uninstall Switch: {e}") from e


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
    schema_manager = None
    job_id = None
    examples_base_dir = None
    test_resources_dir = None
    config = None

    @classmethod
    def setup_class(cls):
        """Setup orchestrator following conftest.py pattern"""
        if not cls._setup_environment_and_config():
            return
        if cls.config.clean_all_before:
            cls._cleanup_resources()
        cls._setup_resources()

    @classmethod
    def teardown_class(cls):
        """Teardown orchestrator following conftest.py pattern"""
        if not cls.config.keep_after:
            cls._cleanup_resources()

    @classmethod
    def _setup_environment_and_config(cls) -> bool:
        """Setup environment, credentials, and validate requirements

        Returns:
            bool: True if setup successful, False if should skip tests
        """
        # Get configuration
        cls.config = LakebridgeTestConfig.from_env()

        # Check all requirements at once
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")

        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        if not cls.config.warehouse_id:
            pytest.skip("LAKEBRIDGE_SWITCH_E2E_WAREHOUSE_ID required for schema operations")

        # Initialize workspace client after all validations
        cls.workspace_client = WorkspaceClient(host=host, token=token)
        return True

    @classmethod
    def _setup_resources(cls) -> None:
        """Setup new resources: schema, Switch installation, examples"""
        # Setup NEW resources AFTER cleanup
        # Initialize schema manager and generate unique test schema
        cls.schema_manager = SwitchSchemaManager(cls.workspace_client, cls.config.warehouse_id)
        test_schema = cls.schema_manager.generate_unique_schema_name(SCHEMA_PREFIX)

        # Create test schema
        logger.info(f"Creating test schema: {cls.config.catalog}.{test_schema}")
        cls.schema_manager.create_schema(cls.config.catalog, test_schema)
        cls.config.test_schema = test_schema

        # Install Switch once
        logger.info("Setting up Switch for all conversion tests")
        cls.job_id = cls._ensure_switch_installed_class(cls.workspace_client)

        # Build name for examples base directory
        current_user = cls.workspace_client.current_user.me().user_name
        cls.examples_base_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}{EXAMPLES_DIR_SUFFIX}"

        # Set up test resources directory
        cls.test_resources_dir = Path(__file__).parent.parent.parent / "resources" / "switch"

        # Upload examples once
        logger.info(f"Uploading examples to {cls.examples_base_dir}")
        cls._upload_test_examples()

    @classmethod
    def _cleanup_resources(cls) -> None:
        """Cleanup resources (always runs when called)"""
        logger.info("Cleaning up resources")

        # Initialize cleanup manager if needed
        if not cls.cleanup_manager:
            cls.cleanup_manager = SwitchCleanupManager(cls.workspace_client, cls.config.warehouse_id)

        # Comprehensive cleanup (jobs, schemas, tables, etc.)
        try:
            cls.cleanup_manager.cleanup_all_if_requested(cls.config.catalog, prefix=SCHEMA_PREFIX)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        # Clean up test directories
        if cls.examples_base_dir and cls.workspace_client:
            try:
                cls.workspace_client.workspace.delete(cls.examples_base_dir, recursive=True)
            except Exception as e:
                logger.warning(f"Failed to delete examples directory: {e}")

    @classmethod
    def _upload_test_examples(cls) -> None:
        """Upload test examples from local resources to workspace"""
        # Upload entire test resources directory using switch e2e_utils pattern
        cls._upload_directory_to_workspace(cls.test_resources_dir, cls.examples_base_dir)

    @classmethod
    def _upload_directory_to_workspace(cls, local_dir: Path, workspace_dir: str) -> None:
        """Upload a local directory to workspace"""
        if not local_dir.exists():
            logger.warning(f"Local directory {local_dir} does not exist, skipping")
            return

        logger.info(f"Uploading {local_dir} -> {workspace_dir}")

        # Create workspace directory
        cls.workspace_client.workspace.mkdirs(workspace_dir)

        # Upload all files recursively
        file_count = 0
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path to maintain directory structure
                relative_path = file_path.relative_to(local_dir)
                workspace_file_path = f"{workspace_dir}/{relative_path}"
                
                # Create parent directories if needed
                workspace_parent = "/".join(workspace_file_path.split("/")[:-1])
                if workspace_parent != workspace_dir.rstrip("/"):
                    cls.workspace_client.workspace.mkdirs(workspace_parent)
                
                # Upload file
                with open(file_path, 'rb') as f:
                    content = f.read()
                cls.workspace_client.workspace.upload(
                    path=workspace_file_path,
                    content=content,
                    format=ImportFormat.AUTO,
                    overwrite=True
                )
                logger.info(f"  ✓ Uploaded: {workspace_file_path}")
                file_count += 1

        logger.info(f"  Directory upload completed: {file_count} files")

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
            # Install using new SwitchInstaller pattern
            transpiler_repository = TranspilerRepository.user_home()
            switch_installer = SwitchInstaller(transpiler_repository, workspace_client)
            switch_installer.install()
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

    def test_sql_basic_conversion(self):
        """Basic SQL conversion: Snowflake → Databricks (async)"""
        current_user = self.workspace_client.current_user.me().user_name

        try:
            # Execute conversion using shared resources
            input_dir = f"{self.examples_base_dir}/sql/snowflake"
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-basic-output-{int(time.time())}"

            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="snowflake",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema
            )

            # Verify result
            self._verify_conversion_result(result)
            logger.info("Basic SQL conversion test completed successfully")
        except Exception as e:
            logger.error(f"Basic SQL conversion test failed: {e}")
            raise

    def test_sql_advanced_conversion(self):
        """Advanced SQL conversion: TSQL → Databricks with advanced parameters (async)"""
        current_user = self.workspace_client.current_user.me().user_name

        try:
            # Execute conversion with advanced parameters using shared resources
            input_dir = f"{self.examples_base_dir}/sql/tsql"
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-advanced-output-{int(time.time())}"
            sql_output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-advanced-sql-{int(time.time())}"

            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="tsql",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema,
                extra_options={
                    "log_level": "DEBUG",
                    "comment_lang": "Japanese",
                    "token_count_threshold": "25000",
                    "sql_output_dir": sql_output_dir,
                    "max_fix_attempts": "1"
                }
            )

            # Verify result
            self._verify_conversion_result(result)
            logger.info("Advanced SQL conversion test completed successfully")
        except Exception as e:
            logger.error(f"Advanced SQL conversion test failed: {e}")
            raise

    def test_airflow_file_conversion(self):
        """Airflow DAG to YAML file conversion with generic source format (async)"""
        current_user = self.workspace_client.current_user.me().user_name

        try:
            # Execute conversion using shared resources
            input_dir = f"{self.examples_base_dir}/workflow/airflow"
            output_dir = f"/Workspace/Users/{current_user}/{BASE_DIR_PREFIX}-airflow-output-{int(time.time())}"

            result = self._execute_conversion(
                workspace_client=self.workspace_client,
                source_dialect="airflow",
                input_source=input_dir,
                output_folder=output_dir,
                catalog_name=self.config.catalog,
                schema_name=self.config.test_schema,
                extra_options={
                    "source_format": "generic",
                    "target_type": "file",
                    "output_extension": ".yml"
                }
            )

            # Verify result
            self._verify_conversion_result(result)
            logger.info("Airflow file conversion test completed successfully")
        except Exception as e:
            logger.error(f"Airflow file conversion test failed: {e}")
            raise

    # ==================== Helper Methods for Conversion Tests ====================

    def _execute_conversion(self, workspace_client: WorkspaceClient, 
                           source_dialect: str, input_source: str, output_folder: str,
                           catalog_name: str, schema_name: str,
                           extra_options: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Execute Lakebridge conversion (always async)"""
        # Prepare transpiler options
        transpiler_options = extra_options if extra_options else {}

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

        # Execute via Lakebridge CLI (always async)
        ctx = ApplicationContext(workspace_client)
        return cli._execute_switch_directly(ctx, transpile_config)

    def _verify_conversion_result(self, result: List[Dict[str, Any]]):
        """Verify conversion result (async only)"""
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Result should contain exactly one item"

        result_item = result[0]
        assert result_item["transpiler"] == "switch", "Transpiler should be 'switch'"
        assert "job_id" in result_item, "Result should contain job_id"
        assert "run_id" in result_item, "Result should contain run_id"
        assert "run_url" in result_item, "Result should contain run_url"


# ==================== Common Helper Functions ====================

def _get_switch_job_id() -> Optional[int]:
    """Get Switch job ID from config"""
    switch_config = TranspilerRepository.user_home().read_switch_config()
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
