"""End-to-end tests for Switch transpiler integration with Lakebridge

Switch is a Databricks-native SQL transpiler that uses LLMs to convert SQL between dialects.
Unlike other existing transpilers (BladeRunner, Morpheus) that use LSP, Switch runs as a 
Databricks job, making it scalable and cloud-native.

This module contains six types of tests:
1. Installation Lifecycle: Test install, reinstall, and uninstall operations
2. SQL Conversion (Async): Test SQL-to-notebook conversion with asynchronous job execution
3. SQL Conversion (Sync): Test SQL-to-notebook conversion with synchronous job execution
4. Generic Conversion: Test non-SQL file conversion (Python/Scala/Airflow) with source_format=generic
5. File Output Conversion: Test SQL-to-file conversion with target_type=file and output_extension
6. Advanced Parameters: Test custom parameters (request_params, comment_lang, token_count_threshold)

Key Challenges Addressed:
- Workspace vs Local Paths: CLI validates paths as local filesystem paths, but Switch 
  requires Databricks workspace paths. We bypass CLI validation by calling APIs directly.
- TestPyPI Limitations: Packages on TestPyPI aren't available on PyPI.org, causing
  installation failures. We mock the PyPI installation step while still testing the real
  workspace deployment.
- Job Cleanup Timing: Databricks job deletion isn't always immediate. Tests handle
  this gracefully without failing on timing issues.

Environment Variables:
- LAKEBRIDGE_SWITCH_E2E=true: Enable E2E tests of this module (disabled by default)
- LAKEBRIDGE_SWITCH_E2E_PYPI_SOURCE: "testpypi" (default) or "pypi"
- LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true: Include slow synchronous tests
- LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true: Clean up all existing Switch jobs before tests
- LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true: Keep resources after tests (for debugging)
- LAKEBRIDGE_SWITCH_E2E_CATALOG: Custom catalog name (default: "remorph")
- LAKEBRIDGE_SWITCH_E2E_SCHEMA: Custom schema name (default: "transpiler")
- DATABRICKS_HOST & DATABRICKS_TOKEN: Workspace credentials

Usage:
    # Quick test (async only, ~30 seconds)
    LAKEBRIDGE_SWITCH_E2E=true pytest tests/integration/switch/test_e2e.py -v

    # Full test including sync mode (~10 minutes)
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true pytest tests/integration/switch/test_e2e.py -v

    # Debug mode (keep resources for inspection)
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true pytest tests/integration/switch/test_e2e.py -v

    # CI mode (clean environment before tests)
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true pytest tests/integration/switch/test_e2e.py -v

    # Debug with clean start (clean before, keep after)
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER=true pytest tests/integration/switch/test_e2e.py -v

    # Custom catalog and schema
    LAKEBRIDGE_SWITCH_E2E=true LAKEBRIDGE_SWITCH_E2E_CATALOG=my_catalog LAKEBRIDGE_SWITCH_E2E_SCHEMA=my_schema pytest tests/integration/switch/test_e2e.py -v
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from databricks.labs.lakebridge.install import TranspilerInstaller


logger = logging.getLogger(__name__)

# Load environment variables at module level
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("LAKEBRIDGE_SWITCH_E2E") != "true",
    reason="Switch E2E tests disabled. Set LAKEBRIDGE_SWITCH_E2E=true to enable"
)
class TestSwitchE2E:
    """E2E tests for Switch transpiler lifecycle"""

    # ==================== Test Fixtures & Setup ====================

    @pytest.fixture(scope="class")
    def pypi_source(self):
        """Determine PyPI source from environment"""
        return os.getenv("LAKEBRIDGE_SWITCH_E2E_PYPI_SOURCE", "testpypi")

    @pytest.fixture(scope="class")
    def workspace_client(self):
        """Create Databricks workspace client"""
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")

        if not (host and token):
            pytest.skip("Databricks credentials required. Set DATABRICKS_HOST and DATABRICKS_TOKEN")

        return WorkspaceClient(host=host, token=token)

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, workspace_client):
        """Ensure clean state before and after tests"""
        clean_all_before = os.getenv("LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE") == "true"
        keep_after = os.getenv("LAKEBRIDGE_SWITCH_E2E_KEEP_AFTER") == "true"
        
        logger.info(f"Cleanup settings: clean_all_before={clean_all_before}, keep_after={keep_after}")

        # Setup: Perform cleanup before tests
        if clean_all_before:
            logger.info("Performing comprehensive Switch job cleanup before tests")
            self._cleanup_all_switch_jobs(workspace_client)
        else:
            # Standard cleanup: Clean any existing Switch installation
            job_id = self._get_switch_job_id_from_config()
            if job_id:
                logger.info(f"Cleaning up existing Switch job {job_id} before test")
                try:
                    from switch.api.installer import SwitchInstaller
                    installer = SwitchInstaller(workspace_client)
                    result = installer.uninstall(job_id=job_id)
                    logger.info(f"Cleanup result: {result.message}")
                except ImportError:
                    # Switch not yet installed, use direct cleanup
                    self._cleanup_databricks_job(workspace_client, job_id)

            # Clean local files
            self._cleanup_local_switch_files()

        yield

        # Teardown: Clean up test artifacts
        if keep_after:
            logger.info("Keeping resources for debugging")
        else:
            logger.info("Cleaning up Switch resources after test")
            self._cleanup_switch_completely(workspace_client)

    # ==================== Main Test Methods ====================

    def test_installation_lifecycle(self, workspace_client, pypi_source):
        """Complete installation workflow: install → reinstall → uninstall
        
        This test verifies:
        - Fresh installation works correctly
        - Reinstallation properly cleans up previous installation
        - Uninstallation removes all resources
        """
        logger.info(f"Starting installation lifecycle test with PyPI source: {pypi_source}")

        # Step 1: Install Switch
        logger.info("Step 1: Installing Switch transpiler")
        job_id = self._setup_switch_for_test(workspace_client, pypi_source)
        logger.info(f"Switch installed successfully with job ID: {job_id}")

        # Step 2: Reinstall (test cleanup of previous installation)
        logger.info("Step 2: Testing reinstallation with cleanup")
        old_job_id = job_id
        new_job_id = self._setup_switch_for_test(workspace_client, pypi_source)

        assert new_job_id != old_job_id, "New job ID should be different from old one"
        logger.info(f"Reinstallation successful. Old job {old_job_id} cleaned up, new job {new_job_id} created")

        # Step 3: Uninstall and verify
        logger.info("Step 3: Testing uninstallation")
        self._cleanup_switch_completely(workspace_client)

        assert self._get_switch_job_id_from_config() is None, "Switch config should be removed"
        assert not self._verify_job_exists(workspace_client, new_job_id), f"Job {new_job_id} should be deleted"
        logger.info("Uninstallation successful")

    def test_sql_conversion_async(self, workspace_client, pypi_source):
        """SQL to notebook conversion workflow (async execution, ~30 seconds)

        This test verifies:
        - Install Switch (if needed)
        - Execute async SQL conversion
        - Verify job submission and run ID

        Note: Respects KEEP_AFTER for debugging
        """
        test_config = {
            "source_dialect": "snowflake",
            "file_content": "SELECT * FROM customers WHERE age > 21",
            "wait_for_completion": False
        }
        
        result = self._execute_test_with_setup(
            workspace_client, pypi_source, "async", test_config
        )
        
        self._verify_switch_result(result, expected_async=True)

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC") != "true",
        reason="Sync test skipped (takes ~10 min). Set LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true to run"
    )
    def test_sql_conversion_sync(self, workspace_client, pypi_source):
        """SQL to notebook conversion workflow (sync execution, ~10 minutes)

        This test verifies:
        - Install Switch (if needed)
        - Execute sync SQL conversion
        - Verify completion and output files

        Note: Respects KEEP_AFTER for debugging
        Why separate: Sync mode takes ~10 minutes vs ~30 seconds for async.
        """
        test_config = {
            "source_dialect": "snowflake",
            "file_content": "SELECT 1 AS test_column",
            "wait_for_completion": True,
            "verify_output": True
        }
        
        start_time = time.time()
        result = self._execute_test_with_setup(
            workspace_client, pypi_source, "sync", test_config
        )
        elapsed_time = time.time() - start_time
        logger.info(f"Sync SQL conversion completed in {elapsed_time:.1f} seconds")
        
        self._verify_switch_result(result, expected_async=False, verify_output=True, 
                                 workspace_client=workspace_client)

    def test_generic_conversion(self, workspace_client, pypi_source):
        """Generic file to notebook conversion (Python/Scala/Airflow)

        This test verifies:
        - Install Switch (if needed)
        - Execute generic file conversion with source_format=generic
        - Verify conversion of non-SQL files

        Note: Tests new source_format parameter functionality
        """
        test_config = {
            "source_dialect": "python",
            "file_content": "# Sample Python script\ndef process_data():\n    return 'converted'\n\nif __name__ == '__main__':\n    result = process_data()\n    print(result)",
            "wait_for_completion": False,
            "source_format": "generic"
        }
        
        result = self._execute_test_with_setup(
            workspace_client, pypi_source, "generic", test_config
        )
        
        self._verify_switch_result(result, expected_async=True)

    def test_file_output_conversion(self, workspace_client, pypi_source):
        """SQL to file conversion (target_type=file, output_extension)

        This test verifies:
        - Install Switch (if needed)
        - Execute SQL conversion with file output instead of notebook
        - Verify target_type and output_extension parameters

        Note: Tests new target_type and output_extension parameters
        """
        test_config = {
            "source_dialect": "snowflake",
            "file_content": "SELECT COUNT(*) as total_records FROM sample_table;",
            "wait_for_completion": False,
            "target_type": "file",
            "output_extension": "sql"
        }
        
        result = self._execute_test_with_setup(
            workspace_client, pypi_source, "file_output", test_config
        )
        
        self._verify_switch_result(result, expected_async=True)

    def test_advanced_parameters(self, workspace_client, pypi_source):
        """Conversion with advanced parameters (request_params, custom settings)

        This test verifies:
        - Install Switch (if needed)
        - Execute conversion with custom request_params
        - Verify advanced parameter functionality

        Note: Tests new request_params and advanced configuration options
        """
        test_config = {
            "source_dialect": "teradata",
            "file_content": "SELECT customer_id, SUM(order_amount) as total FROM orders GROUP BY customer_id HAVING total > 1000;",
            "wait_for_completion": False,
            "comment_lang": "Japanese",
            "token_count_threshold": 15000,
            "request_params": {
                "custom_optimization": True,
                "preserve_comments": True,
                "dialect_specific_hints": ["performance", "compatibility"]
            }
        }
        
        result = self._execute_test_with_setup(
            workspace_client, pypi_source, "advanced", test_config
        )
        
        self._verify_switch_result(result, expected_async=True)

    # ==================== Test Execution Helpers ====================

    def _execute_test_with_setup(self, workspace_client, pypi_source, test_type, test_config):
        """Execute test with common setup pattern: install → setup workspace → execute → cleanup
        
        Args:
            workspace_client: Databricks workspace client
            pypi_source: PyPI source ("testpypi" or "pypi")
            test_type: Test type identifier
            test_config: Test configuration dict containing:
                - source_dialect: SQL source dialect
                - file_content: Test file content
                - wait_for_completion: If True, run synchronously
                - Additional conversion parameters (optional)
        
        Returns:
            Switch execution result
        """
        logger.info(f"Starting {test_type} test with PyPI source: {pypi_source}")
        
        # Step 1: Install Switch
        logger.info("Step 1: Installing Switch transpiler")
        job_id = self._setup_switch_for_test(workspace_client, pypi_source)
        logger.info(f"Switch installed successfully with job ID: {job_id}")
        
        # Step 2: Setup test workspace
        logger.info(f"Step 2: Setting up test workspace for {test_type}")
        input_dir, output_dir = self._setup_test_workspace(
            workspace_client, test_type, test_config["file_content"]
        )
        
        try:
            # Step 3: Execute conversion with parameters
            logger.info(f"Step 3: Executing {test_type} conversion")
            conversion_config = {
                "source_dialect": test_config["source_dialect"],
                "input_source": input_dir,
                "output_folder": output_dir,
                "wait_for_completion": test_config.get("wait_for_completion", False)
            }
            
            # Add optional parameters
            optional_params = ["source_format", "target_type", "output_extension", 
                             "comment_lang", "token_count_threshold", "request_params"]
            for param in optional_params:
                if param in test_config:
                    conversion_config[param] = test_config[param]
            
            result = self._run_switch_transpilation(workspace_client, conversion_config)
            logger.info(f"{test_type.capitalize()} test execution completed")
            
            # Store output_dir for sync verification if needed
            if test_config.get("verify_output"):
                result[0]["_output_dir"] = output_dir
            
            return result
            
        finally:
            # Step 4: Cleanup test workspace
            self._cleanup_test_workspace(workspace_client, input_dir)
    
    def _verify_switch_result(self, result, expected_async=True, verify_output=False, workspace_client=None):
        """Verify Switch execution result with standardized checks
        
        Args:
            result: Switch execution result
            expected_async: If True, verify async result fields; if False, verify sync fields
            verify_output: If True, check that output files were created
            workspace_client: Required if verify_output=True
        """
        # Basic result structure verification
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Result should contain exactly one item"
        
        result_item = result[0]
        assert result_item["transpiler"] == "switch", "Transpiler should be 'switch'"
        assert "job_id" in result_item, "Result should contain job_id"
        
        job_id = result_item["job_id"]
        
        if expected_async:
            # Async execution verification
            assert "run_id" in result_item, "Async result should contain run_id"
            assert "run_url" in result_item, "Async result should contain run_url"
            logger.info(f"Async execution verified: job_id={job_id}, run_id={result_item['run_id']}")
        else:
            # Sync execution verification
            assert "state" in result_item, "Sync result should contain state"
            assert "result_state" in result_item, "Sync result should contain result_state"
            assert result_item["state"] in ["TERMINATED", "SKIPPED"], f"Invalid state: {result_item['state']}"
            logger.info(f"Sync execution verified: job_id={job_id}, state={result_item['state']}")
        
        if verify_output:
            # Output file verification for sync tests
            if "_output_dir" not in result_item:
                logger.warning("Output verification requested but output_dir not available")
                return
                
            output_dir = result_item["_output_dir"]
            if workspace_client:
                try:
                    output_items = workspace_client.workspace.list(output_dir)
                    output_count = sum(1 for _ in output_items)
                    assert output_count > 0, "No output files created"
                    logger.info(f"Output verification passed: {output_count} files created")
                except Exception as e:
                    logger.warning(f"Output verification failed: {e}")
            
            # Clean up the temporary field
            del result_item["_output_dir"]

    # ==================== Installation & Configuration ====================

    def _setup_switch_for_test(self, workspace_client, pypi_source):
        """Set up Switch for testing (install + verify in one operation)

        Args:
            workspace_client: Databricks workspace client
            pypi_source: "testpypi" or "pypi"

        Returns:
            str: Switch job ID

        Raises:
            ValueError: If installation or verification fails
        """
        logger.info(f"Setting up Switch for test with PyPI source: {pypi_source}")

        try:
            # Install Switch based on PyPI source
            if pypi_source == "pypi":
                output = self._run_cli_command("install-transpile")
                logger.info(f"PyPI install output: {output}")
            else:
                self._install_switch_from_testpypi()
                logger.info("TestPyPI installation completed via direct API")
        except Exception as e:
            logger.error(f"Switch installation failed: {e}")
            # Debug: Check if Switch package is available
            try:
                import switch
                logger.info(f"Switch package found at: {switch.__file__}")
            except ImportError:
                logger.error("Switch package not available after installation")
            raise ValueError(f"Failed to install Switch: {e}") from e

        # Verify installation success
        job_id = self._get_switch_job_id_from_config()
        if job_id is None:
            raise ValueError("Switch job ID not found after installation")

        if not self._verify_job_exists(workspace_client, job_id):
            raise ValueError(f"Switch job {job_id} not found in workspace")

        logger.info(f"Switch setup completed successfully with job ID: {job_id}")
        return job_id

    def _install_switch_from_testpypi(self):
        """Install Switch from TestPyPI with proper setup

        Why special handling: TestPyPI packages aren't on PyPI.org, causing
        TranspilerInstaller.install_from_pypi() to fail with 404. We install
        directly from TestPyPI then mock the PyPI step.
        """
        # First install the package from TestPyPI
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-i", "https://test.pypi.org/simple/",
            "databricks-switch-plugin",
            "--force-reinstall",
            "--no-deps"
        ], check=True)

        # Set up directory structure and config
        self._setup_switch_config_for_testpypi()

        # Install via direct API with mocked PyPI
        self._install_switch_via_api_with_mock()

    def _setup_switch_config_for_testpypi(self):
        """Set up Switch config file in expected location"""
        import switch
        import shutil

        switch_package_dir = Path(switch.__file__).parent
        source_config = switch_package_dir.parent / "lsp" / "config.yml"

        transpilers_path = Path.home() / ".databricks" / "labs" / "remorph-transpilers"
        switch_path = transpilers_path / "switch" / "lib"
        switch_path.mkdir(parents=True, exist_ok=True)
        target_config = switch_path / "config.yml"

        shutil.copy2(source_config, target_config)
        logger.info(f"Copied Switch config from {source_config} to {target_config}")

    def _install_switch_via_api_with_mock(self):
        """Install Switch using API with mocked PyPI install"""
        from unittest.mock import patch
        from databricks.labs.lakebridge.install import TranspilerInstaller, WorkspaceInstaller
        from databricks.labs.lakebridge.contexts.application import ApplicationContext
        from databricks.sdk import WorkspaceClient

        # Create workspace client and installer
        ws = WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"),
            token=os.getenv("DATABRICKS_TOKEN")
        )
        app_context = ApplicationContext(ws)
        installer = WorkspaceInstaller(
            app_context.workspace_client,
            app_context.prompts,
            app_context.installation,
            app_context.install_state,
            app_context.product_info,
            app_context.resource_configurator,
            app_context.workspace_installation,
        )

        # Mock install_from_pypi to succeed
        switch_path = Path.home() / ".databricks" / "labs" / "remorph-transpilers" / "switch" / "lib"
        with patch.object(TranspilerInstaller, 'install_from_pypi', return_value=switch_path):
            installer.install_switch()

    def _get_switch_job_id_from_config(self):
        """Get Switch job ID from config using new implementation"""
        switch_config = TranspilerInstaller.read_switch_config()
        if switch_config:
            return switch_config.get('custom', {}).get('job_id')
        return None

    # ==================== Conversion Execution Methods ====================

    def _run_switch_transpilation(self, workspace_client, test_config):
        """Execute Switch conversion with unified interface for test scenarios

        Args:
            workspace_client: Databricks workspace client
            test_config: Dict containing conversion parameters:
                - source_dialect: SQL source dialect
                - input_source: Input directory path
                - output_folder: Output directory path
                - wait_for_completion: If True, run synchronously (default: False)
                - source_format: Source file format (sql/generic, optional)
                - target_type: Target output type (notebook/file, optional)
                - output_extension: Output file extension (optional)
                - comment_lang: Comment language (optional)
                - token_count_threshold: Token count threshold (optional)
                - request_params: Advanced request parameters (optional)

        Returns:
            dict: Switch execution result

        Why bypass CLI: The CLI's path validation assumes local filesystem paths,
        but Switch requires Databricks workspace paths like /Workspace/Users/...
        Direct API calls avoid this validation mismatch.
        """
        from databricks.labs.lakebridge.config import TranspileConfig
        from databricks.labs.lakebridge.contexts.application import ApplicationContext
        from databricks.labs.lakebridge.cli import _execute_switch_directly

        # Extract parameters with defaults
        source_dialect = test_config["source_dialect"]
        input_source = test_config["input_source"]
        output_folder = test_config["output_folder"]
        wait_for_completion = test_config.get("wait_for_completion", False)

        # Get catalog and schema from environment variables or test config
        catalog_name = (test_config.get("catalog_name") or 
                       os.getenv("LAKEBRIDGE_SWITCH_E2E_CATALOG", "remorph"))
        schema_name = (test_config.get("schema_name") or 
                      os.getenv("LAKEBRIDGE_SWITCH_E2E_SCHEMA", "transpiler"))

        # Prepare transpiler options for sync/async control and new features
        transpiler_options = {}
        if wait_for_completion:
            transpiler_options["wait_for_completion"] = "true"
        
        # Add new feature parameters if provided
        if "source_format" in test_config:
            transpiler_options["source_format"] = test_config["source_format"]
        if "target_type" in test_config:
            transpiler_options["target_type"] = test_config["target_type"]
        if "output_extension" in test_config:
            transpiler_options["output_extension"] = test_config["output_extension"]
        if "comment_lang" in test_config:
            transpiler_options["comment_lang"] = test_config["comment_lang"]
        if "token_count_threshold" in test_config:
            transpiler_options["token_count_threshold"] = str(test_config["token_count_threshold"])
        if "request_params" in test_config:
            import json
            transpiler_options["request_params"] = json.dumps(test_config["request_params"])

        # Create config with proper paths and transpiler options
        config = TranspileConfig(
            transpiler_config_path=str(self._get_switch_config_path()),
            source_dialect=source_dialect,
            input_source=input_source,
            output_folder=output_folder,
            catalog_name=catalog_name,
            schema_name=schema_name,
            skip_validation=True,  # Skip path validation for workspace paths
            transpiler_options=transpiler_options
        )

        # Execute Switch directly
        ctx = ApplicationContext(workspace_client)
        execution_mode = "synchronous" if wait_for_completion else "asynchronous"
        logger.info(f"Executing Switch transpilation ({execution_mode} mode)")
        logger.debug(f"Test config: {test_config}")

        result = _execute_switch_directly(ctx, config)
        logger.info(f"Switch execution result: {result}")

        return result

    def _run_transpile(self, source_dialect, input_source, output_folder):
        """Run transpile command and parse JSON result"""
        output = self._run_cli_command(
            "transpile",
            "--transpiler-config-path", str(self._get_switch_config_path()),
            "--source-dialect", source_dialect,
            "--input-source", input_source,
            "--output-folder", output_folder
        )

        return self._parse_json_from_output(output)

    def _parse_json_from_output(self, output):
        """Extract JSON from command output"""
        lines = output.strip().split('\n')
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise ValueError("No JSON output found in command output")

    # ==================== Cleanup Methods ====================

    def _cleanup_switch_completely(self, workspace_client):
        """Complete Switch cleanup (unified interface for all cleanup operations)

        This method provides a single entry point for all Switch cleanup operations,
        combining job cleanup, local files cleanup, and configuration cleanup.

        Args:
            workspace_client: Databricks workspace client
        """
        logger.info("Starting complete Switch cleanup")

        try:
            # Primary cleanup using SwitchInstaller if available
            job_id = self._get_switch_job_id_from_config()
            if job_id:
                logger.info(f"Found Switch job {job_id}, initiating cleanup")
                self._uninstall_switch(workspace_client)
            else:
                logger.info("No Switch job ID found, skipping job cleanup")

            # Ensure local files are cleaned up
            self._cleanup_local_switch_files()

            logger.info("Complete Switch cleanup finished successfully")

        except Exception as e:
            logger.error(f"Error during complete Switch cleanup: {e}")
            # Continue with manual cleanup as fallback
            logger.info("Attempting manual cleanup as fallback")
            try:
                if job_id:
                    self._cleanup_databricks_job(workspace_client, job_id)
                self._cleanup_local_switch_files()
                logger.info("Manual cleanup completed")
            except Exception as fallback_error:
                logger.error(f"Manual cleanup also failed: {fallback_error}")
                raise

    def _cleanup_all_switch_jobs(self, workspace_client):
        """Clean up all Switch jobs for current user

        This method scans all jobs in the workspace and deletes Switch jobs
        belonging to the current user. Useful for cleaning up orphaned jobs
        from previous test runs.
        """
        logger.info("=== Starting comprehensive Switch job cleanup ===")

        try:
            # Get current user for tag-based ownership check
            current_user = workspace_client.current_user.me().user_name
            job_name = "lakebridge-switch"  # Fixed job name (no user suffix)
            created_by_tag = "created_by"    # Tag key for ownership

            logger.info(f"Current user: {current_user}")
            logger.info(f"Looking for Switch jobs with name: '{job_name}' and created_by tag: '{current_user}'")

            # Find all Switch jobs for current user
            logger.info("Fetching all jobs from workspace...")
            switch_jobs = []
            all_jobs = []

            try:
                job_count = 0
                for job in workspace_client.jobs.list():
                    job_count += 1
                    all_jobs.append(job)

                    if job.settings and job.settings.name:
                        logger.debug(f"Job {job.job_id}: '{job.settings.name}' - checking name and tags")

                        # Check if job name matches Switch job name
                        if job.settings.name == job_name:
                            # Check if created_by tag matches current user
                            if (job.settings.tags and 
                                created_by_tag in job.settings.tags and 
                                job.settings.tags[created_by_tag] == current_user):
                                logger.info(f"MATCH: Job {job.job_id}: '{job.settings.name}' (created_by: {job.settings.tags[created_by_tag]})")
                                switch_jobs.append(job)
                            else:
                                created_by_value = job.settings.tags.get(created_by_tag, "<no tag>") if job.settings.tags else "<no tags>"
                                logger.debug(f"NO MATCH (owner): Job {job.job_id}: '{job.settings.name}' (created_by: {created_by_value})")
                        else:
                            logger.debug(f"NO MATCH (name): Job {job.job_id}: '{job.settings.name}'")
                    else:
                        logger.debug(f"Job {job.job_id}: No name or settings")

                logger.info(f"Total jobs scanned: {job_count}")
                logger.info(f"All job names (first 10):")
                for i, job in enumerate(all_jobs[:10]):
                    name = job.settings.name if (job.settings and job.settings.name) else "<no name>"
                    logger.info(f"  {i+1}. Job {job.job_id}: '{name}'")

            except Exception as e:
                logger.error(f"Error listing jobs: {e}")
                return

            if not switch_jobs:
                logger.info("No Switch jobs found to clean up")
                logger.info("=== Comprehensive cleanup completed (no jobs found) ===")
                return

            logger.info(f"Found {len(switch_jobs)} Switch job(s) to clean up:")
            for job in switch_jobs:
                logger.info(f"  - Job ID: {job.job_id}, Name: '{job.settings.name}'")

            # Delete each Switch job safely
            deleted_count = 0
            for job in switch_jobs:
                try:
                    logger.info(f"Processing job {job.job_id}: '{job.settings.name}'")

                    # Check if job is currently running
                    try:
                        runs = list(workspace_client.jobs.list_runs(job_id=job.job_id, active_only=True))
                        if runs:
                            logger.warning(f"Skipping job {job.job_id} - has {len(runs)} active runs")
                            continue
                        else:
                            logger.info(f"Job {job.job_id} has no active runs - safe to delete")
                    except Exception as e:
                        logger.warning(f"Could not check active runs for job {job.job_id}: {e}")
                        logger.info(f"Proceeding with deletion of job {job.job_id}")

                    # Delete the job
                    logger.info(f"Deleting job {job.job_id}...")
                    workspace_client.jobs.delete(job.job_id)
                    logger.info(f"✓ Successfully deleted Switch job {job.job_id}: '{job.settings.name}'")
                    deleted_count += 1

                except Exception as e:
                    logger.error(f"✗ Failed to delete job {job.job_id}: {e}")

            logger.info(f"=== Comprehensive cleanup completed: {deleted_count}/{len(switch_jobs)} jobs deleted ===")

        except Exception as e:
            logger.error(f"Error during comprehensive Switch job cleanup: {e}")
            logger.error(f"=== Comprehensive cleanup failed ===", exc_info=True)

    def _uninstall_switch(self, workspace_client):
        """Uninstall Switch completely using SwitchInstaller"""
        try:
            from switch.api.installer import SwitchInstaller

            job_id = self._get_switch_job_id_from_config()

            # Use SwitchInstaller to uninstall
            installer = SwitchInstaller(workspace_client)
            result = installer.uninstall(job_id=job_id)

            if result.success:
                logger.info(result.message)
            else:
                logger.error(result.message)

            # Additionally clean up local transpiler files
            # (SwitchInstaller only cleans workspace files, not local transpiler installation)
            self._cleanup_local_switch_files()

        except ImportError:
            # Fallback if Switch package not available
            logger.warning("Switch package not available, using manual cleanup")
            job_id = self._get_switch_job_id_from_config()
            if job_id:
                self._cleanup_databricks_job(workspace_client, job_id)
            self._cleanup_local_switch_files()

    def _cleanup_databricks_job(self, workspace_client, job_id):
        """Delete Databricks job"""
        try:
            workspace_client.jobs.delete(job_id)
            logger.info(f"Deleted job {job_id}")
        except NotFound:
            logger.info(f"Job {job_id} already deleted")
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")

    def _cleanup_local_switch_files(self):
        """Remove local Switch installation files"""
        transpilers_path = Path.home() / ".databricks" / "labs" / "remorph-transpilers"
        switch_path = transpilers_path / "switch"

        if switch_path.exists():
            import shutil
            shutil.rmtree(switch_path)
            logger.info("Removed local Switch files")

    # ==================== Workspace Operations ====================

    def _setup_test_workspace(self, workspace_client, test_type, file_content):
        """Set up test workspace environment (create directories + upload test file)

        Args:
            workspace_client: Databricks workspace client
            test_type: Test type identifier ("async", "sync", "generic", etc.)
            file_content: File content to upload for testing

        Returns:
            tuple: (input_dir, output_dir) paths
        """
        from databricks.sdk.service.workspace import ImportFormat

        # Create test directories
        current_user = workspace_client.current_user.me().user_name
        test_base_dir = f"/Workspace/Users/{current_user}/.lakebridge-switch-e2e-{test_type}"
        input_dir = f"{test_base_dir}/input_{int(time.time())}"
        output_dir = f"{test_base_dir}/output_{int(time.time())}"

        # Create directories
        workspace_client.workspace.mkdirs(input_dir)
        workspace_client.workspace.mkdirs(output_dir)
        logger.debug(f"Created test directories: {input_dir}, {output_dir}")

        # Determine file extension based on test type
        file_extensions = {
            "generic": "py",      # Python file for generic conversion
            "file_output": "sql", # SQL file for file output test
            "advanced": "sql",    # SQL file for advanced parameters
            "async": "sql",      # Default SQL for async test
            "sync": "sql"        # Default SQL for sync test
        }
        file_ext = file_extensions.get(test_type, "sql")
        
        # Upload test file with appropriate extension
        file_path = f"{input_dir}/test.{file_ext}"
        workspace_client.workspace.upload(
            path=file_path,
            content=file_content.encode('utf-8'),
            format=ImportFormat.AUTO
        )
        logger.debug(f"Uploaded {file_ext.upper()} file to {file_path} with content: {file_content[:50]}...")

        return input_dir, output_dir

    def _cleanup_test_workspace(self, workspace_client, *paths):
        """Clean up test workspace environment (delete multiple paths safely)

        Args:
            workspace_client: Databricks workspace client
            *paths: List of paths to clean up
        """
        for path in paths:
            if not path:
                continue
            try:
                # Get parent directory and delete recursively
                parent_dir = str(Path(path).parent)
                workspace_client.workspace.delete(parent_dir, recursive=True)
                logger.debug(f"Cleaned up workspace directory: {parent_dir}")
            except Exception as e:
                logger.debug(f"Failed to clean up workspace directory {path}: {e}")

    # ==================== Utility Methods ====================

    def _run_cli_command(self, *args):
        """Execute lakebridge CLI command and return output"""
        cmd = ["databricks", "labs", "lakebridge"] + list(args)
        logger.debug(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"), "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN")}
        )

        if result.returncode != 0:
            logger.error(f"Command failed with code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"CLI command failed: {' '.join(args)}")

        return result.stdout

    def _verify_job_exists(self, workspace_client, job_id):
        """Check if Databricks job exists"""
        try:
            job = workspace_client.jobs.get(job_id)
            return job is not None
        except NotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking job {job_id}: {e}")
            return False

    def _update_switch_config_wait_option(self, wait: bool):
        """Update wait_for_completion in Switch config"""
        # First, update workspace config if it exists
        workspace_config_path = Path.home() / ".databricks" / "labs" / "lakebridge" / "config.yml"
        if workspace_config_path.exists():
            try:
                with open(workspace_config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                if "transpiler_options" not in config:
                    config["transpiler_options"] = {}

                config["transpiler_options"]["wait_for_completion"] = str(wait).lower()

                with open(workspace_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                logger.debug(f"Updated workspace config with wait_for_completion={wait}")
            except Exception as e:
                logger.warning(f"Failed to update workspace config: {e}")

    def _get_switch_config_path(self):
        """Get the Switch config path"""
        transpilers_path = Path.home() / ".databricks" / "labs" / "remorph-transpilers"
        return transpilers_path / "switch" / "lib" / "config.yml"
