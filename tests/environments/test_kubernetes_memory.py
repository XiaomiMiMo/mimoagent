"""Tests for k8s exec memory-leak guards.

Two defenses against unbounded memory growth on long exec sessions:

1. ``_drain_ws_all_buffer`` clears the kubernetes ``WSClient._all`` StringIO
   accumulator, which otherwise grows unbounded (tens of GB over long agent
   sessions) because we only ever read per-channel.
2. command output is capped at ``_MAX_OUTPUT_BYTES`` with head-shedding so a
   command spewing huge output can't exhaust memory; the tail is preserved.
"""

import io
from unittest.mock import MagicMock, patch

import pytest

from mimoagent.environments.kubernetes import KubernetesEnvironment


@pytest.fixture
def k8s_env():
    with patch("mimoagent.environments.kubernetes.config.load_kube_config"):
        with patch("mimoagent.environments.kubernetes.client.ApiClient"):
            with patch("mimoagent.environments.kubernetes.client.CoreV1Api"):
                env = KubernetesEnvironment(image="test:latest")
                env.pod_name = "mimoagent-test1234"
                env.pod_started = True
                yield env


class TestDrainWsAllBuffer:
    def test_drains_all_buffer(self):
        resp = MagicMock()
        resp._all = io.StringIO()
        resp._all.write("x" * 1000)
        KubernetesEnvironment._drain_ws_all_buffer(resp)
        assert resp._all.getvalue() == ""
        assert resp._all.tell() == 0

    def test_no_all_attribute_is_noop(self):
        # getattr default path: a resp without _all must not raise.
        resp = MagicMock(spec=[])
        KubernetesEnvironment._drain_ws_all_buffer(resp)

    def test_swallows_truncate_errors(self):
        resp = MagicMock()
        resp._all.truncate.side_effect = ValueError("closed")
        # Must not propagate — draining is best-effort.
        KubernetesEnvironment._drain_ws_all_buffer(resp)


class TestTrimChunks:
    def test_trims_oldest_to_cap(self):
        chunks = ["aaaa", "bbbb", "cccc"]  # 12 bytes
        trimmed, total = KubernetesEnvironment._trim_chunks(chunks, 12, 8)
        assert "".join(trimmed) == "bbbbcccc"
        assert total == 8

    def test_no_trim_under_cap(self):
        chunks = ["aa", "bb"]
        trimmed, total = KubernetesEnvironment._trim_chunks(chunks, 4, 100)
        assert trimmed == ["aa", "bb"]
        assert total == 4

    def test_empty(self):
        trimmed, total = KubernetesEnvironment._trim_chunks([], 0, 8)
        assert trimmed == []
        assert total == 0


class TestExecuteDrainsAndCaps:
    def test_execute_drains_all_buffer_each_iteration(self, k8s_env):
        """The reader loop must drain resp._all while streaming output."""
        resp = MagicMock()
        resp._all = io.StringIO()

        # Two iterations of output, then the stream closes.
        open_states = iter([True, True, False, False, False])
        resp.is_open.side_effect = lambda: next(open_states, False)
        stdout_peeks = iter([True, True])
        resp.peek_stdout.side_effect = lambda: next(stdout_peeks, False)
        resp.peek_stderr.return_value = False

        def read_stdout():
            # Simulate the client mirroring the chunk into _all too.
            resp._all.write("hello")
            return "hello"

        resp.read_stdout.side_effect = read_stdout
        resp.returncode = 0

        with patch("mimoagent.environments.kubernetes.stream", return_value=resp):
            result = k8s_env.execute("echo hello")

        assert result["reason"] == "ok"
        assert "hello" in result["output"]
        # _all was truncated rather than left to accumulate.
        assert resp._all.getvalue() == ""

    def test_execute_caps_output_at_max_bytes(self, k8s_env, monkeypatch):
        """Output larger than the cap is shed from the head, tail preserved."""
        monkeypatch.setattr(KubernetesEnvironment, "_MAX_OUTPUT_BYTES", 100)
        resp = MagicMock()
        resp._all = io.StringIO()

        # Emit 20 chunks of 50 bytes = 1000 bytes through the loop.
        labels = [chr(ord("A") + i) * 50 for i in range(20)]
        emit = iter(labels)
        n_iters = len(labels)
        open_states = iter([True] * n_iters + [False])
        resp.is_open.side_effect = lambda: next(open_states, False)
        peeks = iter([True] * n_iters)
        resp.peek_stdout.side_effect = lambda: next(peeks, False)
        resp.peek_stderr.return_value = False
        resp.read_stdout.side_effect = lambda: next(emit, "")
        resp.returncode = 0

        with patch("mimoagent.environments.kubernetes.stream", return_value=resp):
            result = k8s_env.execute("flood")

        out = result["output"]
        # Bounded to roughly the cap (allow one chunk of slack before trim).
        assert len(out) <= 150
        # The tail (last chunk) survives; the head (first chunk) is gone.
        assert labels[-1] in out
        assert labels[0] not in out
