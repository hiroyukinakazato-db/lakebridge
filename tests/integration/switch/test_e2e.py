"""End-to-end tests for Switch transpiler integration with Lakebridge

Switch is a Databricks-native SQL transpiler that uses LLMs to convert SQL between dialects.
Unlike traditional transpilers (BladeRunner, Morpheus) that use LSP, Switch runs as a 
Databricks job, making it scalable and cloud-native.

This module contains three types of tests:
1. Installation Lifecycle: Test install, reinstall, and uninstall operations
2. Async Transpilation: Test SQL conversion with asynchronous job execution
3. Sync Transpilation: Test SQL conversion with synchronous job execution (wait for completion)

Key Challenges Addressed:
- Workspace vs Local Paths: CLI validates paths as local filesystem paths, but Switch 
  requires Databricks workspace paths. We bypass CLI validation by calling APIs directly.
- TestPyPI Limitations: Packages on TestPyPI aren't available on PyPI.org, causing
  installation failures. We mock the PyPI installation step while still testing the real
  workspace deployment.
- Job Cleanup Timing: Databricks job deletion isn't always immediate. Tests handle
  this gracefully without failing on timing issues.

Environment Variables:
- LAKEBRIDGE_SWITCH_E2E=true: Enable E2E tests (disabled by default)
- LAKEBRIDGE_SWITCH_E2E_PYPI_SOURCE: "testpypi" (default) or "pypi"
- LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true: Include slow synchronous tests
- LAKEBRIDGE_SWITCH_E2E_CLEAN_ALL_BEFORE=true: Clean up all Switch jobs before tests
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

from databricks.labs.lakebridge.cli import _get_switch_job_id


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
            job_id = _get_switch_job_id()
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

    def test_install_lifecycle(self, workspace_client, pypi_source):
        """Test Switch installation lifecycle: Install → Reinstall → Uninstall
        
        This test verifies:
        - Fresh installation works correctly
        - Reinstallation properly cleans up previous installation
        - Uninstallation removes all resources
        """
        logger.info(f"Starting Switch installation lifecycle test with PyPI source: {pypi_source}")

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

        assert _get_switch_job_id() is None, "Switch config should be removed"
        assert not self._verify_job_exists(workspace_client, new_job_id), f"Job {new_job_id} should be deleted"
        logger.info("Uninstallation successful")

    def test_transpile_async(self, workspace_client, pypi_source):
        """Test asynchronous transpilation with Switch

        This test verifies:
        - Install Switch (if needed)
        - Execute async transpilation (~30 seconds)
        - Verify job submission and run ID

        Note: Respects KEEP_AFTER for debugging
        """
        logger.info(f"Starting async transpilation test with PyPI source: {pypi_source}")

        # Step 1: Install Switch
        logger.info("Step 1: Installing Switch transpiler")
        job_id = self._setup_switch_for_test(workspace_client, pypi_source)
        logger.info(f"Switch installed successfully with job ID: {job_id}")

        # Step 2: Execute asynchronous transpilation
        logger.info("Step 2: Testing asynchronous transpilation")
        input_dir, output_dir = self._setup_test_workspace(
            workspace_client, "async", "SELECT * FROM customers WHERE age > 21"
        )

        try:
            result = self._run_switch_transpilation(workspace_client, {
                "source_dialect": "snowflake",
                "input_source": input_dir,
                "output_folder": output_dir,
                "wait_for_completion": False
            })

            # Verify async execution result
            assert result["transpiler"] == "switch"
            assert result["job_id"] == job_id
            assert "run_id" in result
            assert "run_url" in result
            logger.info(f"Async transpilation started with run ID: {result['run_id']}")

        finally:
            self._cleanup_test_workspace(workspace_client, input_dir)

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC") != "true",
        reason="Sync test skipped (takes ~10 min). Set LAKEBRIDGE_SWITCH_E2E_INCLUDE_SYNC=true to run"
    )
    def test_transpile_sync(self, workspace_client, pypi_source):
        """Test synchronous transpilation with Switch

        This test verifies:
        - Install Switch (if needed)
        - Execute sync transpilation (~10 minutes)
        - Verify completion and output files

        Note: Respects KEEP_AFTER for debugging
        Why separate: Sync mode takes ~10 minutes vs ~30 seconds for async.
        """
        logger.info(f"Starting sync transpilation test with PyPI source: {pypi_source}")

        # Step 1: Install Switch
        logger.info("Step 1: Installing Switch transpiler")
        job_id = self._setup_switch_for_test(workspace_client, pypi_source)
        logger.info(f"Switch installed successfully with job ID: {job_id}")

        # Step 2: Execute synchronous transpilation
        logger.info("Step 2: Testing synchronous transpilation")
        input_dir, output_dir = self._setup_test_workspace(
            workspace_client, "sync", "SELECT 1 AS test_column"
        )

        try:
            start_time = time.time()

            result = self._run_switch_transpilation(workspace_client, {
                "source_dialect": "snowflake",
                "input_source": input_dir,
                "output_folder": output_dir,
                "wait_for_completion": True
            })

            elapsed_time = time.time() - start_time
            logger.info(f"Sync transpilation completed in {elapsed_time:.1f} seconds")

            # Verify sync execution result
            assert result["transpiler"] == "switch"
            assert result["job_id"] == job_id
            assert "state" in result
            assert "result_state" in result
            assert result["state"] in ["TERMINATED", "SKIPPED"]

            # Check output was created
            output_items = workspace_client.workspace.list(output_dir)
            output_count = sum(1 for _ in output_items)
            assert output_count > 0, "No output files created"
            logger.info(f"Sync transpilation successful with state: {result['state']}")

        finally:
            self._cleanup_test_workspace(workspace_client, input_dir)

    # ==================== Installation Methods ====================

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
        job_id = _get_switch_job_id()
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
        source_config = switch_package_dir / "lsp" / "config.yml"

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

    # ==================== Execution Methods ====================

    def _run_switch_transpilation(self, workspace_client, test_config):
        """Execute Switch transpilation with unified interface for test scenarios

        Args:
            workspace_client: Databricks workspace client
            test_config: Dict containing transpilation parameters:
                - source_dialect: SQL source dialect
                - input_source: Input directory path
                - output_folder: Output directory path
                - wait_for_completion: If True, run synchronously (default: False)

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

        # Prepare transpiler options for sync/async control
        transpiler_options = {}
        if wait_for_completion:
            transpiler_options["wait_for_completion"] = "true"

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
            job_id = _get_switch_job_id()
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
            # Get current user and create job name pattern
            current_user = workspace_client.current_user.me().user_name
            # Extract username part (before @) and replace dots with underscores
            username_local = current_user.split('@')[0]
            username_pattern = username_local.replace('.', '_')
            job_name_prefix = f"lakebridge-switch {username_pattern}"

            logger.info(f"Current user: {current_user}")
            logger.info(f"Username pattern: {username_pattern}")
            logger.info(f"Job name prefix to match: '{job_name_prefix}'")

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
                        logger.debug(f"Job {job.job_id}: '{job.settings.name}' - checking against '{job_name_prefix}'")
                        if job.settings.name.startswith(job_name_prefix):
                            logger.info(f"MATCH: Job {job.job_id}: '{job.settings.name}'")
                            switch_jobs.append(job)
                        else:
                            logger.debug(f"NO MATCH: Job {job.job_id}: '{job.settings.name}'")
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

            job_id = _get_switch_job_id()

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
            job_id = _get_switch_job_id()
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

    def _setup_test_workspace(self, workspace_client, test_type, sql_content):
        """Set up test workspace environment (create directories + upload SQL)

        Args:
            workspace_client: Databricks workspace client
            test_type: Test type identifier ("async", "sync", etc.)
            sql_content: SQL content to upload for testing

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

        # Upload SQL file
        sql_path = f"{input_dir}/test.sql"
        workspace_client.workspace.upload(
            path=sql_path,
            content=sql_content.encode('utf-8'),
            format=ImportFormat.AUTO
        )
        logger.debug(f"Uploaded SQL file to {sql_path} with content: {sql_content[:50]}...")

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
