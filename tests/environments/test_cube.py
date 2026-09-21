"""Integration tests for CubeEnvironment.

They need a reachable CubeSandbox service and the ``cubesandbox`` SDK
(``uv sync --extra cube``). Set these before running:
  export CUBE_API_URL=http://<cubesandbox-api-host>
  export CUBE_SANDBOX_DOMAIN=<sandbox-proxy-domain>
  export CUBE_TEMPLATE_ID=<template id>
"""

import os
import tempfile

import pytest

try:
    import cubesandbox  # noqa: F401

    CUBESANDBOX_AVAILABLE = True
except ImportError:
    CUBESANDBOX_AVAILABLE = False

from mimoagent.environments.cube import CubeEnvironment, CubeEnvironmentConfig


def is_cube_available():
    """Check if CubeSandbox service is reachable."""
    return CUBESANDBOX_AVAILABLE and bool(os.getenv("CUBE_API_URL")) and bool(os.getenv("CUBE_TEMPLATE_ID"))


skip_no_cube = pytest.mark.skipif(
    not is_cube_available(),
    reason="CubeSandbox not available (cubesandbox not installed or CUBE_API_URL / CUBE_TEMPLATE_ID not set)",
)

TEMPLATE_ID = os.getenv("CUBE_TEMPLATE_ID", "tpl-example")


@skip_no_cube
class TestCubeEnvironmentConfig:
    def test_config_defaults(self):
        config = CubeEnvironmentConfig(template_id=TEMPLATE_ID)
        assert config.template_id == TEMPLATE_ID
        assert config.cwd == "/testbed"
        assert config.timeout == 30
        assert config.env == {}
        assert config.forward_env == []
        assert config.host_mounts == []

    def test_config_custom(self):
        config = CubeEnvironmentConfig(
            template_id=TEMPLATE_ID,
            cwd="/workspace",
            timeout=60,
            env={"FOO": "bar"},
        )
        assert config.cwd == "/workspace"
        assert config.timeout == 60
        assert config.env == {"FOO": "bar"}


@skip_no_cube
@pytest.mark.integration
class TestCubeEnvironment:
    def test_create_and_execute(self):
        """Create a sandbox, run echo, check output and returncode."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            result = env.execute("echo 'hello world'")
            assert result["returncode"] == 0
            assert "hello world" in result["output"]
            assert result["reason"] == "ok"
        finally:
            env.cleanup()

    def test_execute_env_vars(self):
        """Configured env vars are visible to commands."""
        env = CubeEnvironment(
            template_id=TEMPLATE_ID,
            cwd="/tmp",
            env={"TEST_VAR": "cube_value"},
        )
        env.start()
        try:
            result = env.execute("echo $TEST_VAR")
            assert result["returncode"] == 0
            assert "cube_value" in result["output"]
        finally:
            env.cleanup()

    def test_execute_timeout(self):
        """A command that overruns its timeout reports reason='timeout'."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            result = env.execute("sleep 30", timeout=2)
            assert result["reason"] == "timeout"
            assert result["returncode"] == 124
        finally:
            env.cleanup()

    def test_execute_nonzero_exit(self):
        """A failing command reports a non-zero returncode."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            result = env.execute("exit 42")
            assert result["returncode"] == 42
            assert result["reason"] == "ok"
        finally:
            env.cleanup()

    def test_copy_to_file(self):
        """copy_to of a single file lands its content in the sandbox."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("hello from host")
                tmp_path = f.name
            try:
                env.copy_to(tmp_path, "/tmp/test_file.txt")
                result = env.execute("cat /tmp/test_file.txt")
                assert result["returncode"] == 0
                assert "hello from host" in result["output"]
            finally:
                os.unlink(tmp_path)
        finally:
            env.cleanup()

    def test_copy_to_directory(self):
        """copy_to of a directory preserves the tree."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.makedirs(os.path.join(tmpdir, "subdir"))
                with open(os.path.join(tmpdir, "root.txt"), "w") as f:
                    f.write("root file")
                with open(os.path.join(tmpdir, "subdir", "nested.txt"), "w") as f:
                    f.write("nested file")

                env.copy_to(tmpdir, "/tmp/test_dir")

            result = env.execute("cat /tmp/test_dir/root.txt")
            assert "root file" in result["output"]

            result = env.execute("cat /tmp/test_dir/subdir/nested.txt")
            assert "nested file" in result["output"]
        finally:
            env.cleanup()

    def test_copy_out_file(self):
        """copy_out of a file created in the sandbox matches locally."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            env.execute("echo 'from sandbox' > /tmp/out_file.txt")
            with tempfile.TemporaryDirectory() as tmpdir:
                dest = os.path.join(tmpdir, "out_file.txt")
                env.copy_out("/tmp/out_file.txt", dest)
                with open(dest) as f:
                    assert "from sandbox" in f.read()
        finally:
            env.cleanup()

    def test_copy_out_directory(self):
        """copy_out of a directory created in the sandbox preserves the tree."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        try:
            env.execute("mkdir -p /tmp/outdir/sub && echo a > /tmp/outdir/a.txt && echo b > /tmp/outdir/sub/b.txt")
            with tempfile.TemporaryDirectory() as tmpdir:
                dest = os.path.join(tmpdir, "outdir")
                env.copy_out("/tmp/outdir", dest)
                assert os.path.isdir(dest)
                with open(os.path.join(dest, "a.txt")) as f:
                    assert "a" in f.read()
                with open(os.path.join(dest, "sub", "b.txt")) as f:
                    assert "b" in f.read()
        finally:
            env.cleanup()

    def test_cleanup(self):
        """cleanup drops the sandbox reference."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/tmp")
        env.start()
        assert env.sandbox is not None
        env.cleanup()
        assert env.sandbox is None
        assert env.sandbox_id is None

    def test_get_template_vars(self):
        """get_template_vars returns the config as a dict."""
        env = CubeEnvironment(template_id=TEMPLATE_ID, cwd="/workspace")
        vars = env.get_template_vars()
        assert vars["template_id"] == TEMPLATE_ID
        assert vars["cwd"] == "/workspace"
