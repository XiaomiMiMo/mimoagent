"""Test that KubernetesEnvironment uses a per-instance ApiClient.

The per-instance ApiClient eliminates cross-instance pool contention and
per-call TLS handshake overhead. execute() and copy_to() share the same
ApiClient (via self.v1_api), while cleanup() uses a fresh dedicated
ApiClient since the instance client's pool may hold half-closed
WebSocket connections.
"""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from mimoagent.environments.kubernetes import KubernetesEnvironment


@pytest.fixture
def k8s_env():
    """KubernetesEnvironment with kube config stubbed."""
    with patch("mimoagent.environments.kubernetes.config.load_kube_config"):
        with patch("mimoagent.environments.kubernetes.client.ApiClient"):
            with patch("mimoagent.environments.kubernetes.client.CoreV1Api"):
                env = KubernetesEnvironment(image="test:latest")
                env.pod_name = "mimoagent-test1234"
                env.pod_started = True
                yield env


class TestExecutePerInstanceApiClient:
    """execute() must use self.v1_api (per-instance), not create per-call clients."""

    def test_execute_uses_instance_v1_api(self, k8s_env):
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            mock_stream.return_value = resp

            result = k8s_env.execute("echo hello")

        assert result["reason"] == "ok"
        # stream() was called with self.v1_api's exec method
        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == k8s_env.v1_api.connect_get_namespaced_pod_exec

    def test_execute_does_not_create_new_api_client(self, k8s_env):
        with patch("mimoagent.environments.kubernetes.client.ApiClient") as mock_api_client:
            with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
                resp = MagicMock()
                resp.is_open.return_value = False
                resp.returncode = 0
                resp.peek_stdout.return_value = False
                resp.peek_stderr.return_value = False
                mock_stream.return_value = resp

                k8s_env.execute("echo a")
                k8s_env.execute("echo b")

        # No new ApiClient created during execute calls
        mock_api_client.assert_not_called()

    def test_execute_returns_transport_error_on_stream_failure(self, k8s_env):
        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ):
            result = k8s_env.execute("echo hello")

        assert result == {
            "output": "Error opening exec stream: socket is already closed.",
            "returncode": None,
            "reason": "transport_error",
        }

    def test_execute_does_not_mutate_v1_api(self, k8s_env):
        original_v1_api = k8s_env.v1_api

        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ):
            k8s_env.execute("echo hello")

        assert k8s_env.v1_api is original_v1_api


class TestCopyToPerInstanceApiClient:
    """copy_to() must use self.v1_api, with resp.close() in finally."""

    @staticmethod
    def _successful_response():
        resp = MagicMock()
        resp.is_open.return_value = False
        resp.returncode = 0
        resp.peek_stdout.return_value = False
        resp.peek_stderr.return_value = False
        return resp

    def test_copy_to_uses_instance_v1_api(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_file), "/dest/test.txt")

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == k8s_env.v1_api.connect_get_namespaced_pod_exec

    def test_copy_to_bounds_untar_read_with_head(self, k8s_env, tmp_path):
        """untar command must pipe through ``head -c <archive_len>`` so the
        pod-side reader hits a real EOF.

        ``write_stdin("")`` does not close the exec stdin channel, so BusyBox
        tar (Alpine images) would otherwise block reading stdin forever and the
        exec returns rc=None. ``head -c N`` closes the pipe at exactly the
        archive size, giving tar EOF on both GNU and BusyBox.
        """
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        written = []
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_file), "/dest/test.txt")

        cmd = mock_stream.call_args.kwargs["command"]
        cmd_str = cmd[-1]
        archive_len = len(written[0])
        assert f"head -c {archive_len} | tar xmf -" in cmd_str, cmd_str
        # the byte count handed to head must match the tar archive actually sent
        assert archive_len > 0

    def test_copy_to_closes_resp_in_finally(self, k8s_env, tmp_path):
        """If write_stdin raises after stream() opens, resp must still be closed."""
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        resp = MagicMock()
        resp.write_stdin.side_effect = Exception("broken pipe")
        resp.is_open.return_value = False
        resp.returncode = 0

        with patch("mimoagent.environments.kubernetes.stream", return_value=resp):
            with patch("time.sleep"):
                with pytest.raises(Exception, match="broken pipe"):
                    k8s_env.copy_to(str(src_file), "/dest/test.txt", max_retries=2)

        # 2 attempts, resp.close() called in finally each time
        assert resp.close.call_count == 2

    def test_copy_to_packs_tar_once_across_retries(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        written = []

        def stream_side_effect(*args, **kwargs):
            stream_side_effect.calls += 1
            if stream_side_effect.calls <= 2:
                raise Exception("socket is already closed.")
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            return resp

        stream_side_effect.calls = 0

        with patch("mimoagent.environments.kubernetes.stream", side_effect=stream_side_effect):
            with patch("time.sleep"):
                k8s_env.copy_to(str(src_file), "/dest/test.txt")

        # tar bytes only written on the successful (3rd) attempt
        assert len(written) >= 1
        assert isinstance(written[0], bytes) and len(written[0]) > 0

    def test_copy_to_exhausts_retries_on_persistent_error(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ) as mock_stream:
            with patch("time.sleep"):
                with pytest.raises(Exception, match="socket is already closed"):
                    k8s_env.copy_to(str(src_file), "/dest/test.txt", max_retries=3)

        assert mock_stream.call_count == 3

    def test_copy_to_automatically_chunks_large_regular_file(self, k8s_env, tmp_path, monkeypatch):
        src_file = tmp_path / "large.patch"
        src_file.write_bytes(b"large payload")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", src_file.stat().st_size)

        chunk_info = {
            "bytes": src_file.stat().st_size,
            "chunks": 1,
            "seconds": 0.1,
            "sha256": "abc",
        }
        with patch.object(k8s_env, "copy_file_chunked", return_value=chunk_info) as chunked:
            with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
                k8s_env.copy_to(
                    str(src_file),
                    "/tmp/recalc_model.patch",
                    timeout=123,
                    max_retries=7,
                )

        chunked.assert_called_once_with(
            str(src_file),
            "/tmp/recalc_model.patch",
            timeout=123,
            max_retries=7,
        )
        mock_stream.assert_not_called()

    def test_copy_to_keeps_small_file_on_tar_stream(self, k8s_env, tmp_path, monkeypatch):
        src_file = tmp_path / "small.patch"
        src_file.write_bytes(b"small")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", src_file.stat().st_size + 1)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch(
                "mimoagent.environments.kubernetes.stream",
                return_value=self._successful_response(),
            ) as mock_stream:
                k8s_env.copy_to(str(src_file), "/tmp/recalc_model.patch")

        chunked.assert_not_called()
        mock_stream.assert_called_once()

    def test_copy_to_keeps_directories_on_tar_stream(self, k8s_env, tmp_path, monkeypatch):
        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch(
                "mimoagent.environments.kubernetes.stream",
                return_value=self._successful_response(),
            ):
                k8s_env.copy_to(str(src_dir), "/tmp/payload")

        chunked.assert_not_called()

    def test_copy_to_keeps_symlinks_on_tar_stream(self, k8s_env, tmp_path, monkeypatch):
        target = tmp_path / "target.patch"
        target.write_bytes(b"large")
        src_link = tmp_path / "large.patch"
        src_link.symlink_to(target)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch(
                "mimoagent.environments.kubernetes.stream",
                return_value=self._successful_response(),
            ):
                k8s_env.copy_to(str(src_link), "/tmp/large.patch")

        chunked.assert_not_called()

    def test_copy_to_keeps_directory_destination_on_tar_stream(self, k8s_env, tmp_path, monkeypatch):
        src_file = tmp_path / "large.patch"
        src_file.write_bytes(b"large")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch(
                "mimoagent.environments.kubernetes.stream",
                return_value=self._successful_response(),
            ):
                k8s_env.copy_to(str(src_file), "/tmp/")

        chunked.assert_not_called()

    def test_copy_file_chunked_supports_relative_destination(self, k8s_env, tmp_path):
        src_file = tmp_path / "patch.diff"
        src_file.write_bytes(b"patch")
        src_file.chmod(0o751)
        digest = hashlib.sha256(b"patch").hexdigest()
        k8s_env.execute = MagicMock(
            side_effect=[
                {"returncode": 0, "output": ""},
                {"returncode": 0, "output": digest},
                {"returncode": 0, "output": ""},
            ]
        )

        with patch.object(k8s_env, "_write_chunk"):
            k8s_env.copy_file_chunked(str(src_file), "recalc_model.patch")

        first_command = k8s_env.execute.call_args_list[0].args[0]
        mode_command = k8s_env.execute.call_args_list[2].args[0]
        assert first_command.startswith("mkdir -p . && : > recalc_model.patch")
        assert mode_command == "chmod 751 recalc_model.patch"


class TestCleanupDedicatedApiClient:
    """cleanup() uses a fresh ApiClient and closes the instance client first."""

    def test_cleanup_closes_instance_api_client(self, k8s_env):
        mock_instance_client = MagicMock()
        k8s_env._api_client = mock_instance_client

        with patch("mimoagent.environments.kubernetes.client.Configuration.get_default_copy"):
            with patch("mimoagent.environments.kubernetes.client.ApiClient"):
                mock_v1 = MagicMock()
                with patch("mimoagent.environments.kubernetes.client.CoreV1Api", return_value=mock_v1):
                    from kubernetes.client import ApiException

                    mock_v1.delete_namespaced_pod.side_effect = ApiException(status=404)
                    k8s_env.cleanup()

        # Instance client closed before delete cycle
        mock_instance_client.close.assert_called_once()
        assert k8s_env._api_client is None
