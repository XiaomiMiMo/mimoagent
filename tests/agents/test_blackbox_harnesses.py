"""Focused host-side tests for the payload-based blackbox harnesses.

One file for grok / kimi-code / kimi-cli / openclaw / opencode / omp / hermes /
dsh / kilocode / mini-swe-agent: they share
:class:`HarnessAgent`, so the config/command/parse contracts are worth asserting
side by side. Nothing here touches the network or a pod — the fake env records
commands and replays a canned event log.
"""

import json
from pathlib import Path

import pytest

from mimoagent.agents.blackbox.dsh import DshAgent, DshAgentConfig
from mimoagent.agents.blackbox.grok import GrokAgent, GrokAgentConfig
from mimoagent.agents.blackbox.hermes import HermesAgent, HermesAgentConfig
from mimoagent.agents.blackbox.kilocode import KilocodeAgent, KilocodeAgentConfig
from mimoagent.agents.blackbox.kimi_cli import KimiCliAgent, KimiCliAgentConfig
from mimoagent.agents.blackbox.kimi_code import KimiCodeAgent, KimiCodeAgentConfig
from mimoagent.agents.blackbox.mini_swe_agent import MiniSweAgent, MiniSweAgentConfig
from mimoagent.agents.blackbox.omp import OmpAgent, OmpAgentConfig
from mimoagent.agents.blackbox.openclaw import OpenclawAgent, OpenclawAgentConfig
from mimoagent.agents.blackbox.opencode import OpencodeAgent, OpencodeAgentConfig
from mimoagent.agents.factory import get_agent_class, list_agent_types, make_agent

HARNESSES = {
    "grok": (GrokAgent, GrokAgentConfig),
    "kimi-code": (KimiCodeAgent, KimiCodeAgentConfig),
    "kimi-cli": (KimiCliAgent, KimiCliAgentConfig),
    "openclaw": (OpenclawAgent, OpenclawAgentConfig),
    "opencode": (OpencodeAgent, OpencodeAgentConfig),
    "omp": (OmpAgent, OmpAgentConfig),
    "hermes": (HermesAgent, HermesAgentConfig),
    "dsh": (DshAgent, DshAgentConfig),
    "kilocode": (KilocodeAgent, KilocodeAgentConfig),
    "mini-swe-agent": (MiniSweAgent, MiniSweAgentConfig),
}
NAMES = sorted(HARNESSES)
# The builder's shell variable is derived from the harness name except where
# upstream's own short name won.
_BUILD_VERSION_VAR = {"mini-swe-agent": "MSWEA_VERSION"}
_KEY = "test-gateway-key"


class _Stats:
    def __init__(self):
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_creation_tokens = 0


class _ModelConfig:
    model_name = "claude-opus-5"
    model_kwargs = {"base_url": "http://gw.internal/v1", "api_key": _KEY}


class _Model:
    def __init__(self):
        self.config = _ModelConfig()
        self.config.model_kwargs = dict(_ModelConfig.model_kwargs)
        self.n_calls = 0
        self.token_stats = _Stats()

    def query(self, *args, **kwargs):
        return {}

    def get_template_vars(self):
        return {}


class FakeEnv:
    """Records pod commands; answers reads from ``files`` (path substring →
    content) and the harness run itself with ``rc``."""

    class _Config:
        cwd = "/testbed"

    def __init__(self, *, raw="", rc=0, reason="ok", files=None):
        self.config = self._Config()
        self.raw = raw
        self.rc = rc
        self.reason = reason
        self.files = files or {}
        self.commands = []
        self.copies = []
        self.staged = {}
        self.detached = []

    def copy_to(self, src, dst, **kwargs):
        self.copies.append((src, dst))
        try:
            self.staged[dst] = Path(src).read_text(encoding="utf-8", errors="replace")
        except OSError:  # the install script tarball etc.
            self.staged[dst] = ""

    def execute_detached(self, command, cwd="", timeout=None, *, idle_files=(), idle_timeout=0):
        # the harness run itself goes through here; record how it was asked for
        self.detached.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
                "idle_files": list(idle_files),
                "idle_timeout": idle_timeout,
            }
        )
        return self.execute(command, cwd=cwd, timeout=timeout)

    def execute(self, command, cwd="", timeout=None):
        self.commands.append(command)
        if "bash /tmp/mimo-install-" in command:
            # each harness's own script prints its own sentinel
            name = command.split("bash /tmp/mimo-install-")[1].split(".sh")[0]
            prefix = name.upper().replace("-", "_")
            return {"output": f"{prefix}_INSTALL_OK\n", "returncode": 0}
        if "mkconfig.py" in command:  # mini-swe-agent derives its staged config
            return {"output": "MKCONFIG_OK\n", "returncode": 0}
        if command.startswith("export PATH=/opt/mimo-"):  # the harness run
            return {"output": "", "returncode": self.rc, "reason": self.reason}
        if command.startswith("tail -c 2000000"):
            return {"output": self.raw, "returncode": 0}
        for needle, content in self.files.items():
            if needle in command:
                return {"output": content, "returncode": 0}
        return {"output": "", "returncode": 0}

    # -- helpers for assertions -------------------------------------------------

    @property
    def run_command(self):
        return next(c for c in self.commands if c.startswith("export PATH=/opt/mimo-"))

    @property
    def run_commands(self):
        return [c for c in self.commands if c.startswith("export PATH=/opt/mimo-")]

    def staged_text(self, needle):
        return next(text for dst, text in self.staged.items() if needle in dst)


_DETACHED_IMPL = FakeEnv.execute_detached


def _agent(name, env=None, **cfg):
    cls = HARNESSES[name][0]
    return cls(_Model(), env or FakeEnv(), skip_install=True, **cfg)


# -- registry / pinning ---------------------------------------------------------


def _script_text(relpath: str) -> str:
    """Text of a payload-build script, or skip the test.

    The payload builder and its patch helpers run on a dev box and are not part
    of this feature's runtime surface, so the assertions that guard their
    version pins and build-time patches only run where they are checked out.
    """
    path = Path(__file__).parents[2] / relpath
    if not path.is_file():
        pytest.skip(f"{relpath} not checked out")
    return path.read_text()


@pytest.mark.parametrize("name", NAMES)
def test_factory_lists_and_resolves_every_harness(name):
    assert name in list_agent_types()
    assert get_agent_class(name) is HARNESSES[name][0]


@pytest.mark.parametrize("name", NAMES)
def test_version_pin_is_consistent_across_adapter_config_and_builder(name):
    # payload upgrades touch three places; a half-done bump ships a config
    # pointing at a tarball nobody built. The literal version is deliberately
    # NOT asserted here — it moves — only that the three agree.
    version = HARNESSES[name][1]().version
    assert version, f"{name}: agent.version must be pinned"
    builder = _script_text("scripts/harness_payloads/build_harness_payloads.sh")
    assert f"build_{name.replace('-', '_')}()" in builder
    var = _BUILD_VERSION_VAR.get(name, f"{name.replace('-', '_').upper()}_VERSION")
    assert f"{var}={version}" in builder
    config = (Path(__file__).parents[2] / f"example_configs/{name}.yaml").read_text()
    assert f"version: {version}" in config
    # The installer, rather than the public config, owns the artifact name.


# -- no valid output must never read as success ---------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_empty_event_stream_is_an_error_even_at_rc_zero(name):
    agent = _agent(name)
    text, status = agent._parse_result(0, [])
    assert status == agent.ERROR_STATUS
    assert "no agent message" in text


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "events",
    [
        [{"unrelated": "shape"}],
        [{"type": "text"}],  # no payload at all
        [{"role": "assistant", "content": []}],
        [{"role": "assistant", "content": [{"type": "think", "think": "x"}]}],
    ],
)
def test_malformed_events_are_an_error(name, events):
    agent = _agent(name)
    _text, status = agent._parse_result(0, events)
    assert status == agent.ERROR_STATUS


def test_grok_requires_a_terminating_end_event():
    agent = _agent("grok")
    text, status = agent._parse_result(0, [{"type": "text", "data": "partial"}])
    assert status == agent.ERROR_STATUS  # stream died before `end`
    assert text == "partial"


def test_grok_error_event_fails_even_with_partial_text():
    agent = _agent("grok")
    events = [
        {"type": "text", "data": "here is my plan"},
        {"type": "error", "message": "upstream 429"},
        {"type": "end", "sessionId": "s1", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    text, status = agent._parse_result(0, events)
    assert status == agent.ERROR_STATUS
    assert text == "here is my plan"  # partial text is kept for debugging


def test_opencode_error_event_fails_even_with_partial_text():
    agent = _agent("opencode")
    events = [
        {"type": "text", "part": {"text": "half an answer"}},
        {"type": "error", "part": {"error": "ProviderAuthError"}},
    ]
    text, status = agent._parse_result(0, events)
    assert status == agent.ERROR_STATUS
    assert text == "half an answer"


@pytest.mark.parametrize("stop_reason", ["timeout", "error", "aborted", "abort", "cancelled", "canceled"])
def test_openclaw_failure_stop_reasons_are_errors(stop_reason):
    agent = _agent("openclaw")
    envelope = {
        "payloads": [{"text": "did some of it"}],
        "meta": {"finalAssistantVisibleText": "did some of it", "stopReason": stop_reason},
    }
    text, status = agent._parse_result(0, [envelope])
    assert status == agent.ERROR_STATUS
    assert text == "did some of it"


def test_openclaw_aborted_flag_is_an_error_and_clean_stop_is_not():
    agent = _agent("openclaw")
    base = {"payloads": [{"text": "done"}], "meta": {"stopReason": "end_turn"}}
    assert agent._parse_result(0, [base])[1] == agent.IDLE_STATUS
    aborted = {"payloads": [{"text": "done"}], "meta": {"stopReason": "end_turn", "aborted": True}}
    assert agent._parse_result(0, [aborted])[1] == agent.ERROR_STATUS


def test_omp_error_stop_reason_and_failed_auto_retry_are_errors():
    agent = _agent("omp")
    message = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}], "stopReason": "error"},
    }
    assert agent._parse_result(0, [message])[1] == agent.ERROR_STATUS
    ok = json.loads(json.dumps(message))
    ok["message"]["stopReason"] = "stop"
    assert agent._parse_result(0, [ok])[1] == agent.IDLE_STATUS
    assert agent._parse_result(0, [ok, {"type": "auto_retry_end", "success": False}])[1] == agent.ERROR_STATUS


@pytest.mark.parametrize("name", ["kimi-code", "kimi-cli"])
def test_kimi_assistant_content_accepts_string_and_block_list(name):
    agent = _agent(name)
    text, status = agent._parse_result(0, [{"role": "assistant", "content": "plain string"}])
    assert (text, status) == ("plain string", agent.IDLE_STATUS)
    blocks = [
        {
            "role": "assistant",
            "content": [
                {"type": "think", "think": "hidden", "encrypted": "SIG=="},
                {"type": "text", "text": "block list"},
            ],
        }
    ]
    assert agent._parse_result(0, blocks) == ("block list", agent.IDLE_STATUS)


def test_hermes_nonzero_rc_reports_the_stderr_tail():
    agent = _agent("hermes", FakeEnv(files={"hermes.stderr.txt": "boom: provider auth failed\n"}))
    text, status = agent._parse_result(1, [{"type": "raw", "text": ""}])
    assert status == agent.ERROR_STATUS
    assert "boom: provider auth failed" in text


# -- yolo, tool whitelists, off-trajectory behaviours ---------------------------


def test_grok_command_is_yolo_and_passes_the_disallowed_tool_list():
    # nothing is dropped by default: grok answers ask_user_question itself in
    # headless mode (measured 0.004s to "no user is available, continue with your
    # best judgment"), so the flag is only an escape hatch. What this pins is the
    # unattended-run invariant plus that a configured list reaches argv verbatim.
    env = FakeEnv(raw=json.dumps({"type": "end", "sessionId": "s"}) + "\n")
    agent = _agent("grok", env)
    agent._parse_result(0, [{"type": "text", "data": "hi"}, {"type": "end"}])
    command = " ".join(agent._command_parts("t"))
    assert "--yolo" in command
    assert GrokAgentConfig().disallowed_tools is None
    assert "--disallowed-tools" not in command
    opinionated = " ".join(_agent("grok", env, disallowed_tools="web_search")._command_parts("t"))
    assert "--disallowed-tools web_search" in opinionated.replace("'", "")


def test_grok_auxiliary_model_roles_go_to_a_dead_port():
    agent = _agent("grok")
    toml = agent._config_toml()
    for role in ("session_summary", "web_search", "image_description", "prompt_suggestion"):
        assert f'{role} = "title-blackhole"' in toml
    # the blackhole model must not reach the gateway and must not retry
    assert 'base_url = "http://127.0.0.1:1/v1"' in toml
    assert "max_retries = 0" in toml
    assert _KEY not in toml.split("[model.title-blackhole]")[1]


def test_grok_off_trajectory_features_are_disabled():
    # 1.0.4 asks the main model for a per-turn dashboard summary line (not one
    # of the [models] summary roles, so the blackhole entry does not catch it),
    # ships a `workflow` tool in the default set, and scans the Claude/Cursor/
    # Codex vendor trees for skills/rules/agents to inject into the prompt.
    agent = _agent("grok")
    env = agent._harness_env()
    # the dashboard summary is an extra call on the MAIN model, so the
    # blackhole routes above cannot catch it — it has to be off
    assert env["GROK_TURN_SUMMARY"] == "0"
    toml = agent._config_toml()
    for vendor in ("claude", "cursor", "codex"):
        assert f"[compat.{vendor}]" in toml
    # root `auto_update` is an unrecognized key in 1.0.4 (config warning)
    assert "auto_update" not in toml
    # native skill roots have no on/off switch, only ignore paths
    ignored = _agent("grok", skills_ignore=("~/.agents/skills",))._config_toml()
    assert "~/.agents/skills" in ignored


def test_opencode_disables_todowrite_per_agent():
    agent = _agent("opencode")
    config = json.loads(agent._provider_config())
    # the root `tools` key is a dead legacy field: the switch has to be
    # per-agent, on every built-in agent
    assert set(config["agent"]) == {"build", "plan", "general"}
    for spec in config["agent"].values():
        # the exact lowercase name "todowrite" is a hard 400 on the
        # subscription channel, so it must be off whatever else is allowed
        assert spec["tools"]["todowrite"] is False
    assert config["permission"] == "allow"
    command = " ".join(agent._command_parts("t"))
    assert "--dangerously-skip-permissions" in command
    assert "--title mimo-rollout" in command  # suppresses the title LLM call


def test_opencode_todo_tool_can_be_re_enabled():
    config = json.loads(_agent("opencode", todo_tool=True)._provider_config())
    assert "todowrite" not in config["agent"]["build"]["tools"]


def test_openclaw_denies_meta_tools_and_runs_with_full_exec():
    agent = _agent("openclaw")
    config = json.loads(agent._openclaw_config())
    from mimoagent.agents.blackbox.openclaw import _DENY_TOOLS

    deny = config["tools"]["deny"]
    assert deny == _DENY_TOOLS
    # only the sessions_* family goes, and for a wire reason: it trips the
    # subscription channel's third-party fingerprint. `subagents` stays.
    assert all(name.startswith("sessions_") for name in deny)
    assert "subagents" not in deny
    assert config["agents"]["defaults"]["memorySearch"] == {"enabled": False}
    assert config["tools"]["profile"] == "coding"
    assert config["tools"]["exec"] == {"mode": "full"}
    # the schema rejects timeoutSeconds=0; --timeout 0 is the unlimited knob
    assert "timeoutSeconds" not in config["agents"]["defaults"]
    assert "--timeout 0" in " ".join(agent._command_parts("t"))


def test_omp_is_yolo_and_passes_its_tool_whitelist_verbatim():
    agent = _agent("omp")
    command = " ".join(agent._command_parts("t"))
    assert "--approval-mode yolo" in command
    assert "--no-title" in command  # no title-generation LLM call
    # an unknown tool name makes omp exit 2, so the list must arrive as written
    tools = OmpAgentConfig().tools
    assert f"--tools {tools}" in command
    # the full documented set, subagents (task) included
    assert "task" in tools.split(",") and "web_search" in tools.split(",")


def test_omp_drops_the_request_fields_bedrock_routes_reject():
    # omp attaches the context-management beta to every thinking request and
    # `strict: true` to qualifying tool defs, with no upstream opt-out. A
    # Bedrock-backed route 400s the whole request over either field, so the
    # payload build gates both on env vars and the adapter leaves them unset.
    cfg = OmpAgentConfig()
    assert cfg.context_management is False
    assert cfg.tool_strict is False
    env = _agent("omp")._harness_env()
    assert "OMP_CONTEXT_MANAGEMENT" not in env
    assert "OMP_TOOL_STRICT" not in env
    on = _agent("omp", context_management=True, tool_strict=True)._harness_env()
    assert on["OMP_CONTEXT_MANAGEMENT"] == "1"
    assert on["OMP_TOOL_STRICT"] == "1"
    # the gates only exist if the build patched them in
    builder = _script_text("scripts/harness_payloads/build_harness_payloads.sh")
    section = builder.split("build_omp()")[1].split("build_kimi_cli()")[0]
    assert "process.env.OMP_CONTEXT_MANAGEMENT" in section
    assert "process.env.OMP_TOOL_STRICT" in section


def test_kimi_cli_stages_an_agent_file_with_fully_qualified_tools():
    env = FakeEnv()
    agent = _agent("kimi-cli", env)
    agent._stage_files()
    spec = env.staged_text("agent.yaml")
    # the spec replaces the built-in profile, so tools travel as
    # module:Class and the system prompt has to be pointed at explicitly
    assert "kimi_cli.tools.shell:Shell" in spec
    assert "kimi_cli.tools.file:WriteFile" in spec
    assert "system_prompt_path: /opt/mimo-kimi-cli/" in spec
    # AskUserQuestion stays: --print sets afk, and afk auto-dismisses it in
    # 0.026s with "make your own decision" rather than waiting for anyone
    assert "kimi_cli.tools.ask_user:AskUserQuestion" in spec
    assert f"--agent-file {agent._agent_file}" in " ".join(agent._command_parts("t"))
    assert "default_yolo = true" in agent._config_toml()


def test_kimi_code_tool_trim_hook_is_guarded_at_build_time():
    # 0.36.1 replaced the compiled-in profile literal with a supported hook: a
    # discovered agent file with `override: true` takes over the default
    # profile. The build self-check must fail if that hook's invariants move.
    builder = _script_text("scripts/harness_payloads/build_harness_payloads.sh")
    section = builder.split("build_kimi_code()")[1].split("build_openclaw()")[0]
    assert 'DEFAULT_AGENT_PROFILE_NAME$1 = "agent"' in section
    assert "definition.override" in section
    assert "base_prompt" in section
    adapter = (Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/kimi_code.py").read_text()
    assert "override" in adapter and "tools:" in adapter


def test_hermes_toolset_whitelist_and_yolo():
    agent = _agent("hermes")
    command = " ".join(agent._command_parts("t"))
    assert "--yolo" in command
    # -t is an exclusive whitelist; it must arrive exactly as configured (some
    # gateway channels fingerprint tool-set combinations)
    assert f"-t {HermesAgentConfig().toolsets}" in command
    config = json.loads(agent._config_yaml())
    assert config["approvals"] == {"mode": "off"}
    # clarify is the one ask tool here that yolo does NOT cover: --yolo only maps
    # to approvals, while -Q wires the interactive callback. The timeout is the
    # only bound on the stall, and it MUST be written as clarify.timeout —
    # resolve_clarify_timeout() reads that key first and hermes ships its own
    # default of 120 there, so agent.clarify_timeout never wins (measured).
    assert "clarify" in (HermesAgentConfig().toolsets or "").split(",")
    assert config["clarify"] == {"timeout": HermesAgentConfig().clarify_timeout}
    assert HermesAgentConfig().clarify_timeout <= 10
    assert "clarify_timeout" not in config.get("agent", {})
    # title generation is the one extra LLM call per session
    assert config["auxiliary"] == {"title_generation": {"enabled": False}}
    assert config["model_catalog"] == {"enabled": False}
    # -Q reserves stdout for the reply: no reasoning box, no tirith banner, and
    # (2026.8.13) no inline write_file/patch diff preview ahead of the reply
    assert config["display"] == {"show_reasoning": False, "inline_diffs": False}
    assert config["security"]["tirith_enabled"] is False


def test_dsh_overlay_repoints_the_catalog_route_and_kills_offtrajectory_rows():
    agent = _agent("dsh", effort="max")
    rows = {row["id"]: row for row in json.loads(agent._overlay()) if isinstance(row, dict) and "id" in row}
    # the route key must stay "anthropic": that is pi-ai's catalog entry for
    # claude-opus-5, and its compat.forceAdaptiveThinking is what makes
    # xhigh/max adaptive. A hand-declared route cannot set it.
    route = rows["llm-pi-ai"]["config"]["providers"]["anthropic"]
    assert route["baseURL"] == "http://gw.internal"
    assert route["apiKeyEnv"] == "DSH_GW_API_KEY"
    assert route["reasoning"] == "max"
    assert route["modelOverrides"] == {"claude-opus-5": {"contextWindow": 1000000, "maxTokens": 65536}}
    assert "models" not in route  # a models list would replace the catalog
    assert rows["agent-default-model"]["config"] == {
        "provider": "anthropic",
        "model": "claude-opus-5",
    }
    # usage is read host-side; zstd would need an extra dependency
    assert rows["session-persistence-jsonl"]["config"]["compression"] == "none"
    assert rows["credentials"]["config"] == {"watch": False}
    from mimoagent.agents.blackbox.dsh import _DISABLED_ROWS

    for off in _DISABLED_ROWS:
        assert rows[off]["disabled"] is True
    # the one invariant left: no extra LLM call per session. `user-questions` is
    # deliberately NOT here — the model-facing `tool-ask-user` row is absent from
    # the headless profile anyway, so disabling the service changed nothing on the
    # wire (measured: same 25 tools either way, no ask tool among them).
    assert "session-title-llm" in _DISABLED_ROWS
    assert "user-questions" not in _DISABLED_ROWS
    # no effort configured = no reasoning field at all
    assert "reasoning" not in json.loads(_agent("dsh")._overlay())[0]["config"]["providers"]["anthropic"]


def test_dsh_extra_rows_are_appended_last():
    agent = _agent("dsh", extra_rows=[{"id": "tool-fs-search", "config": {"x": 1}}])
    rows = json.loads(agent._overlay())
    assert rows[-1] == {"id": "tool-fs-search", "config": {"x": 1}}


def test_dsh_command_is_headless_yolo_and_resumes_on_later_turns():
    env = FakeEnv(raw="the answer\n")
    agent = DshAgent(_Model(), env, skip_install=True)
    assert agent.run("first") == ("Completed", "the answer")
    first = env.run_command
    assert "--profile headless" in first
    assert "--patch /tmp/mimo-dsh-logs/overlay.yml" in first
    assert '"$(cat /tmp/mimo-dsh-instruction.txt)"' in first
    first_env = env.staged_text("dsh.env")
    assert "DSH_PERMISSION_MODE=danger-full-access" in first_env
    assert "DSH_SESSION_ID=session-mimo-" in first_env
    assert "DSH_RESUME_SESSION" not in first_env

    assert agent.run("second") == ("Completed", "the answer")
    # the env file is rewritten per turn; turn 2 must resume the session
    assert "DSH_RESUME_SESSION=1" in env.staged_text("dsh.env")


def test_dsh_session_ids_are_unique_per_agent():
    # dsh refuses to resume a session persisted under a different cwd, so a
    # constant id breaks as soon as one pod serves a second task
    first = DshAgent(_Model(), FakeEnv(), skip_install=True)._session_id
    second = DshAgent(_Model(), FakeEnv(), skip_install=True)._session_id
    assert first != second
    assert first.startswith("session-mimo-")


def test_dsh_folds_session_usage_chunks_once_per_turn():
    def usage_chunk(out):
        return json.dumps(
            {
                "type": "assistant/chunk",
                "data": {
                    "chunk": {
                        "type": "usage",
                        "usage": {
                            "inputTokens": 2,
                            "outputTokens": out,
                            "cacheReadTokens": 30,
                            "cacheWriteTokens": 4,
                        },
                    }
                },
            }
        )

    session = "\n".join(
        [json.dumps({"type": "assistant/chunk", "data": {"chunk": {"type": "text"}}}), usage_chunk(5), usage_chunk(6)]
    )
    env = FakeEnv(raw="done\n", files={"session.jsonl": session})
    agent = DshAgent(_Model(), env, skip_install=True)
    assert agent.run("t") == ("Completed", "done")
    assert agent.model.n_calls == 2
    assert agent.model.token_stats.output_tokens == 11
    assert agent.model.token_stats.cache_read_tokens == 60
    # the log is cumulative across the resumed session; no double counting
    assert agent.run("t2") == ("Completed", "done")
    assert agent.model.n_calls == 2
    assert agent.model.token_stats.output_tokens == 11


def test_dsh_payload_patches_are_asserted_at_build_time():
    builder = _script_text("scripts/harness_payloads/build_harness_payloads.sh")
    section = builder.split("build_dsh()")[1].split("\nALL=(")[0]
    # patch 1: the launcher's HMR/patch-file watchers (they kill the run when
    # the host inotify budget is gone)
    assert "if (false) try {" in section
    assert "dsh: watcher block anchor moved" in section
    # patch 2: resumable headless sessions
    assert "DSH_RESUME_SESSION" in section
    assert "resumeSessionId" in section
    assert "dsh: headless agent-creation anchor moved" in section


# -- effort / thinking mapping ---------------------------------------------------


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_grok_folds_max_to_xhigh_its_real_ceiling(effort):
    # 1.0.4's enum stops at xhigh and sends it verbatim (0.2.106 rewrote xhigh
    # into wire "max"; both captured on the real channel). max folds so an
    # all-arms max sweep still runs, with a warning that this arm is a level low
    agent = _agent("grok", effort=effort)
    assert agent._effort == "xhigh"
    assert 'default_reasoning_effort = "xhigh"' in agent._config_toml()
    assert "--reasoning-effort xhigh" in " ".join(agent._command_parts("t"))


def test_grok_without_effort_sends_no_thinking_config():
    agent = _agent("grok")
    assert agent._effort is None
    assert "default_reasoning_effort" not in agent._config_toml()
    assert "--reasoning-effort" not in " ".join(agent._command_parts("t"))


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_opencode_effort_lands_as_a_model_variant(effort):
    agent = _agent("opencode", thinking_effort=effort)
    entry = json.loads(agent._provider_config())["provider"]["gateway"]["models"]["claude-opus-5"]
    assert entry["variants"] == {effort: {"thinking": {"type": "adaptive"}, "effort": effort}}
    assert f"--variant {effort}" in " ".join(agent._command_parts("t"))


def test_opencode_rejects_budget_and_effort_together():
    agent = _agent("opencode", thinking_effort="max", thinking_budget=32000)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        agent._provider_config()


def test_opencode_raises_the_output_token_ceiling_when_needed():
    assert "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in _agent("opencode")._harness_env()
    env = _agent("opencode", output_limit=64000)._harness_env()
    assert env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "64000"


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_kimi_code_effort_survives_the_max_to_high_migration(effort):
    env = FakeEnv()
    agent = _agent("kimi-code", env, effort=effort)
    agent._stage_files()
    assert f'effort = "{effort}"' in agent._config_toml()
    # upstream rewrites a persisted "max" to "high" once; the marker file makes
    # it treat that migration as already applied
    marker = env.staged_text("migrations-effort.json")
    assert json.loads(marker) == {"thinking-effort-max-to-high": "2026-01-01T00:00:00.000Z"}


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_kimi_cli_effort_travels_via_the_patched_env_var(effort):
    assert _agent("kimi-cli", effort=effort)._harness_env()["KIMI_CLI_THINKING_EFFORT"] == effort
    assert "KIMI_CLI_THINKING_EFFORT" not in _agent("kimi-cli")._harness_env()
    assert "KIMI_CLI_THINKING_EFFORT" not in _agent("kimi-cli", effort="max", thinking=False)._harness_env()


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_omp_thinking_is_adaptive_with_the_full_ladder(effort):
    agent = _agent("omp", thinking=effort)
    entry = json.loads(agent._models_yml())["providers"]["gateway"]["models"][0]
    assert entry["thinking"]["mode"] == "anthropic-adaptive"
    assert entry["thinking"]["efforts"][-2:] == ["xhigh", "max"]
    assert f"--thinking {effort}" in " ".join(agent._command_parts("t"))
    # a custom anthropic-messages provider must declare apiKey auth, or omp
    # impersonates Claude Code (oauth betas, clamped max_tokens)
    assert json.loads(agent._models_yml())["providers"]["gateway"]["auth"] == "apiKey"


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_openclaw_thinking_default_carries_the_effort(effort):
    config = json.loads(_agent("openclaw", thinking=effort)._openclaw_config())
    assert config["agents"]["defaults"]["thinkingDefault"] == effort
    assert "thinkingDefault" not in json.loads(_agent("openclaw")._openclaw_config())["agents"]["defaults"]


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_hermes_reasoning_effort_and_native_anthropic_escape_hatch(effort):
    agent = _agent("hermes", effort=effort)
    assert json.loads(agent._config_yaml())["agent"]["reasoning_effort"] == effort
    # without the patched escape hatch upstream strips signed thinking blocks
    # from any non-anthropic.com base_url
    assert agent._harness_env()["HERMES_FORCE_NATIVE_ANTHROPIC"] == "1"
    assert "agent" not in json.loads(_agent("hermes")._config_yaml())


# -- base_url conventions & key hygiene -----------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("grok", "http://gw.internal/v1"),  # appends /messages
        ("opencode", "http://gw.internal/v1"),  # @ai-sdk/anthropic
        ("kimi-code", "http://gw.internal"),  # official SDK appends /v1/messages
        ("kimi-cli", "http://gw.internal"),
        ("openclaw", "http://gw.internal"),
        ("omp", "http://gw.internal"),
        ("hermes", "http://gw.internal"),
        ("dsh", "http://gw.internal"),  # pi-ai appends /v1/messages
        ("kilocode", "http://gw.internal/v1"),  # @ai-sdk/anthropic, like opencode
    ],
)
def test_base_url_v1_convention_per_driver(name, expected):
    agent = _agent(name)
    text = {
        "grok": lambda a: a._config_toml(),
        "opencode": lambda a: a._provider_config(),
        "kimi-code": lambda a: a._config_toml(),
        "kimi-cli": lambda a: a._config_toml(),
        "openclaw": lambda a: a._openclaw_config(),
        "omp": lambda a: a._models_yml(),
        "hermes": lambda a: a._config_yaml(),
        "dsh": lambda a: a._overlay(),
        "kilocode": lambda a: a._provider_config(),
    }[name](agent)
    assert f'"{expected}"' in text
    assert "http://gw.internal/v1/v1" not in text
    if not expected.endswith("/v1"):
        # a driver that appends /v1/messages itself must not get a /v1 base
        assert '"http://gw.internal/v1"' not in text


@pytest.mark.parametrize("name", NAMES)
def test_api_key_never_lands_in_the_run_command(name):
    # keys go through a sourced env file: argv is world-readable via ps
    env = FakeEnv()
    agent = HARNESSES[name][0](_Model(), env, skip_install=True)
    agent._stage_files()
    agent._run_harness("task")
    assert _KEY not in env.run_command
    env_file = env.staged_text(f"{name}.env")
    assert f". /tmp/mimo-{name}-logs/{name}.env" in env.run_command
    assert "no_proxy" in env_file


# -- multi-turn resume ----------------------------------------------------------


def _run_twice(name, raw, files=None):
    env = FakeEnv(raw=raw, files=files or {})
    agent = HARNESSES[name][0](_Model(), env, skip_install=True)
    first = agent.run("first")
    second = agent.run("second")
    return env, agent, first, second


def test_grok_resumes_with_the_session_id_from_the_end_event():
    raw = "\n".join(
        [
            json.dumps({"type": "text", "data": "done"}),
            json.dumps({"type": "end", "sessionId": "sess-42", "num_turns": 1}),
        ]
    )
    env, _agent_obj, first, second = _run_twice("grok", raw)
    assert first[0] == "Completed" and second[0] == "Completed"
    assert "-r " not in env.run_commands[0]
    assert "-r sess-42" in env.run_commands[1]


@pytest.mark.parametrize(
    ("name", "raw", "flag"),
    [
        ("kimi-code", json.dumps({"role": "assistant", "content": "done"}), "--continue"),
        ("kimi-cli", json.dumps({"role": "assistant", "content": "done"}), "--continue"),
        (
            "opencode",
            json.dumps({"type": "text", "part": {"text": "done"}}),
            "--continue",
        ),
        (
            "omp",
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                        "stopReason": "stop",
                    },
                }
            ),
            "--continue",
        ),
    ],
)
def test_second_turn_continues_the_session(name, raw, flag):
    env, _agent_obj, first, second = _run_twice(name, raw)
    assert (first[0], second[0]) == ("Completed", "Completed")
    assert flag not in env.run_commands[0]
    assert flag in env.run_commands[1]


def test_openclaw_reuses_one_session_key_across_turns():
    raw = json.dumps({"payloads": [{"text": "done"}], "meta": {"stopReason": "end_turn"}})
    env, _agent_obj, first, second = _run_twice("openclaw", raw)
    assert (first[0], second[0]) == ("Completed", "Completed")
    assert all("--session-key mimo-task" in c for c in env.run_commands)


def test_hermes_resumes_by_session_id_without_restoring_cwd():
    raw = "the answer\n\nsession_id: 20260815_120000_abc\n"
    files = {
        "hermes.stderr.txt": "\nsession_id: 20260815_120000_abc\n",
        "sqlite3": "[10, 4, 3, 2, 2]\n",
    }
    env, _agent_obj, first, second = _run_twice("hermes", raw, files)
    assert first == ("Completed", "the answer")
    assert "--resume" not in env.run_commands[0]
    assert "--resume 20260815_120000_abc --no-restore-cwd" in env.run_commands[1]


# -- usage / n_calls folding ----------------------------------------------------


def test_grok_folds_end_event_usage_and_counts_model_calls():
    agent = _agent("grok")
    model = agent.model
    events = [
        {"type": "text", "data": "done"},
        {
            "type": "end",
            "sessionId": "s",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 12,
                "cache_read_input_tokens": 900,
            },
            "modelUsage": {"claude-opus-5": {"modelCalls": 3}},
        },
    ]
    assert agent._parse_result(0, events) == ("done", agent.IDLE_STATUS)
    assert (model.token_stats.input_tokens, model.token_stats.output_tokens) == (30, 12)
    assert model.token_stats.cache_read_tokens == 900
    assert model.n_calls == 3  # one folded call + 2 extra from modelUsage


def test_opencode_folds_every_step_finish_block():
    agent = _agent("opencode")
    model = agent.model
    events = [
        {"type": "step_finish", "part": {"tokens": {"input": 5, "output": 7, "cache": {"read": 100, "write": 20}}}},
        {"type": "text", "part": {"text": "done"}},
        {"type": "step_finish", "part": {"tokens": {"input": 6, "output": 8, "cache": {"read": 200, "write": 0}}}},
    ]
    assert agent._parse_result(0, events) == ("done", agent.IDLE_STATUS)
    assert model.n_calls == 2
    assert model.token_stats.input_tokens == 11
    assert model.token_stats.output_tokens == 15
    assert model.token_stats.cache_read_tokens == 300
    assert model.token_stats.cache_creation_tokens == 20


def test_openclaw_folds_usage_and_approximates_calls_from_tool_summary():
    agent = _agent("openclaw")
    model = agent.model
    envelope = {
        "payloads": [{"text": "done"}],
        "meta": {
            "stopReason": "end_turn",
            "toolSummary": {"calls": 4},
            "agentMeta": {"usage": {"input": 20, "output": 9, "cacheRead": 500, "cacheWrite": 30}},
        },
    }
    assert agent._parse_result(0, [envelope]) == ("done", agent.IDLE_STATUS)
    assert model.token_stats.input_tokens == 20
    assert model.token_stats.cache_creation_tokens == 30
    # the envelope has no exact call count: 1 folded call + 4 tool calls
    assert model.n_calls == 5


def test_kimi_code_folds_wire_usage_only_once_per_turn():
    wire = "\n".join(
        json.dumps({"usage": {"inputOther": 2, "output": 5, "inputCacheRead": 40, "inputCacheCreation": 10}})
        for _ in range(2)
    )
    events = "\n".join(
        [
            json.dumps({"role": "assistant", "content": "done"}),
            json.dumps({"role": "meta", "type": "session.resume_hint", "session_id": "sid1"}),
        ]
    )
    env = FakeEnv(raw=events, files={"wire.jsonl": wire})
    agent = KimiCodeAgent(_Model(), env, skip_install=True)
    assert agent.run("first") == ("Completed", "done")
    assert agent.model.n_calls == 2
    assert agent.model.token_stats.output_tokens == 10
    # wire.jsonl is cumulative across turns; the second turn must not re-count
    assert agent.run("second") == ("Completed", "done")
    assert agent.model.n_calls == 2
    assert agent.model.token_stats.output_tokens == 10


def test_kimi_cli_folds_status_update_usage_only_once_per_turn():
    wire = "\n".join(
        json.dumps(
            {
                "message": {
                    "type": "StatusUpdate",
                    "payload": {
                        "token_usage": {
                            "input_other": 3,
                            "output": 6,
                            "input_cache_read": 70,
                            "input_cache_creation": 5,
                        }
                    },
                }
            }
        )
        for _ in range(2)
    )
    env = FakeEnv(raw=json.dumps({"role": "assistant", "content": "done"}), files={"wire.jsonl": wire})
    agent = KimiCliAgent(_Model(), env, skip_install=True)
    assert agent.run("first") == ("Completed", "done")
    assert (agent.model.n_calls, agent.model.token_stats.output_tokens) == (2, 12)
    assert agent.run("second") == ("Completed", "done")
    assert (agent.model.n_calls, agent.model.token_stats.output_tokens) == (2, 12)


def test_hermes_folds_session_tree_usage_from_state_db():
    raw = "answer\n\nsession_id: sid-1\n"
    files = {"hermes.stderr.txt": "session_id: sid-1\n", "sqlite3": "[40, 12, 800, 60, 3]\n"}
    env = FakeEnv(raw=raw, files=files)
    agent = HermesAgent(_Model(), env, skip_install=True)
    assert agent.run("t") == ("Completed", "answer")
    stats = agent.model.token_stats
    assert (stats.input_tokens, stats.output_tokens) == (40, 12)
    assert (stats.cache_read_tokens, stats.cache_creation_tokens) == (800, 60)
    assert agent.model.n_calls == 3  # 1 folded + (api_call_count - 1)


def test_omp_folds_one_call_per_assistant_message():
    def message(text, tokens):
        return {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "stopReason": "stop",
                "usage": {"input": tokens, "output": tokens, "cacheRead": 10, "cacheWrite": 1},
            },
        }

    env = FakeEnv(raw="\n".join(json.dumps(m) for m in (message("", 4), message("done", 6))))
    agent = make_agent("omp", _Model(), env, skip_install=True)
    assert agent.run("t") == ("Completed", "done")
    assert agent.model.n_calls == 2
    assert agent.model.token_stats.input_tokens == 10
    assert agent.model.token_stats.cache_read_tokens == 20


@pytest.mark.parametrize("name", NAMES)
def test_transport_failure_is_infra_not_a_result(name):
    from mimoagent.agents.blackbox.harness_base import InfraError

    env = FakeEnv(rc=0, reason="transport_error")
    agent = HARNESSES[name][0](_Model(), env, skip_install=True)
    with pytest.raises(InfraError):
        agent.run("t")


@pytest.mark.parametrize("name", sorted(HARNESSES))
def test_run_goes_through_the_detached_exec(name):
    """The harness session is the one exec that can outlive the k8s websocket
    lifetime cap, so it must go through execute_detached; everything else
    (install, staging, log reads) stays on plain execute."""
    env = FakeEnv()
    agent = HARNESSES[name][0](_Model(), env, skip_install=True, run_timeout=7200)
    agent.run("t")

    (call,) = env.detached
    assert call["command"] == env.run_command
    assert call["cwd"] == "/testbed"
    assert call["timeout"] == 7200
    assert call["idle_timeout"] == 0  # watchdog off unless configured
    assert call["idle_files"][0] == agent._output_file
    if agent.SPLIT_STDERR:
        assert call["idle_files"] == [agent._output_file, f"{agent._logs_dir}/{agent.NAME}.stderr.txt"]
    else:
        assert call["idle_files"] == [agent._output_file]


def test_stall_timeout_reaches_the_watchdog():
    env = FakeEnv()
    _agent("grok", env=env, stall_timeout=1800).run("t")
    assert env.detached[0]["idle_timeout"] == 1800


def test_stall_or_client_timeout_is_a_harness_error_not_completed():
    env = FakeEnv(rc=None, reason="stall")
    status, _ = _agent("grok", env=env).run("t")
    assert status == "GrokError"


def test_env_without_detached_exec_falls_back_to_attached_run(capsys):
    raw = json.dumps({"type": "text", "data": "done"}) + "\n" + json.dumps({"type": "end", "num_turns": 1})
    env = FakeEnv(raw=raw)
    del FakeEnv.execute_detached  # simulate a foreign Environment implementation
    try:
        status, _ = _agent("grok", env=env).run("t")
    finally:
        FakeEnv.execute_detached = _DETACHED_IMPL
    assert status == "Completed"
    assert env.run_command  # still ran, attached
    # the mimoagent logger does not propagate; its handler prints to stdout
    assert "no execute_detached" in capsys.readouterr().out


# -- kilocode -------------------------------------------------------------------


def test_kilocode_denies_todowrite_and_offtrajectory_tools():
    agent = _agent("kilocode")
    config = json.loads(agent._provider_config())
    from mimoagent.agents.blackbox.kilocode import _DENY_TOOLS

    perm = config["permission"]
    # last-match-wins over globs: the catch-all must come first
    assert list(perm)[0] == "*" and perm["*"] == "allow"
    for tool in list(_DENY_TOOLS) + ["todowrite", "todoread"]:
        assert perm[tool] == "deny"
    # the only real deny is the 400-triggering name; question/suggest are not
    # registered in `kilo run` at all, so denying them was dead config
    assert perm["todowrite"] == "deny"
    assert _DENY_TOOLS == []
    assert "question" not in perm
    # the small model must stay on our gateway or the title call leaves it
    assert config["small_model"] == "gateway/claude-opus-5"
    assert config["disabled_providers"] == ["kilo", "apertis"]
    # auto-compaction is how a long task survives its window
    assert config["compaction"] == {"auto": True}
    assert config["instructions"] == [] and config["mcp"] == {}
    assert json.loads(_agent("kilocode", todo_tool=True)._provider_config())["permission"].get("todowrite") is None


def test_kilocode_command_is_yolo_and_suppresses_the_title_call():
    command = " ".join(_agent("kilocode")._command_parts("t"))
    assert "--auto" in command  # without it the ask handler auto-*rejects*
    assert "--title mimo-rollout" in command
    assert "--format json" in command


def test_kilocode_env_kills_the_runtime_npm_install_and_repo_config():
    env = _agent("kilocode")._harness_env()
    # a fresh HOME otherwise npm-installs @kilocode/plugin from the network
    assert env["KILO_DISABLE_DEFAULT_PLUGINS"] == "1"
    # repo-planted kilo.json / .kilo/ / AGENTS.md must not reach the prompt
    assert env["KILO_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["KILO_TELEMETRY_LEVEL"] == "off"
    assert "KILO_DISABLE_PROJECT_CONFIG" not in _agent("kilocode", project_config=True)._harness_env()
    assert "KILO_EXPERIMENTAL_OUTPUT_TOKEN_MAX" not in env
    assert _agent("kilocode", output_limit=64000)._harness_env()["KILO_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "64000"


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_kilocode_effort_lands_as_a_model_variant(effort):
    agent = _agent("kilocode", thinking_effort=effort)
    entry = json.loads(agent._provider_config())["provider"]["gateway"]["models"]["claude-opus-5"]
    assert entry["variants"] == {effort: {"thinking": {"type": "adaptive"}, "effort": effort}}
    assert f"--variant {effort}" in " ".join(agent._command_parts("t"))
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _agent("kilocode", thinking_effort=effort, thinking_budget=32000)._provider_config()


def test_kilocode_error_event_fails_even_with_partial_text():
    agent = _agent("kilocode")
    events = [
        {"type": "text", "part": {"text": "half an answer"}},
        {"type": "error", "error": {"data": {"message": "ProviderAuthError"}}},
    ]
    assert agent._parse_result(0, events) == ("half an answer", agent.ERROR_STATUS)


def test_kilocode_folds_step_finish_usage_and_resumes():
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "step_finish",
                    "part": {"tokens": {"input": 5, "output": 7, "cache": {"read": 100, "write": 20}}},
                }
            ),
            json.dumps({"type": "text", "part": {"text": "done"}}),
        ]
    )
    env, agent, first, second = _run_twice("kilocode", raw)
    assert (first[0], second[0]) == ("Completed", "Completed")
    assert "--continue" not in env.run_commands[0]
    assert "--continue" in env.run_commands[1]
    assert agent.model.n_calls == 2  # one step per turn
    assert agent.model.token_stats.cache_read_tokens == 200


# -- mini-swe-agent -------------------------------------------------------------


def test_mini_swe_agent_command_is_structurally_non_interactive():
    command = " ".join(_agent("mini-swe-agent")._command_parts("t"))
    # -y alone leaves InteractiveAgent's confirm_exit, which EOFErrors on a pipe
    assert "--agent-class default" in command
    assert "-y" in command and "--exit-immediately" in command
    assert "-c /tmp/mimo-mini-swe-agent-logs/mini-harness.yaml" in command
    assert "environment.environment_class=local" in command
    assert "environment.timeout=300" in command  # upstream's 30s kills builds
    assert "-o /tmp/mimo-mini-swe-agent-logs/traj-0.json" in command


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_mini_swe_agent_effort_passes_through_litellm(effort):
    command = " ".join(_agent("mini-swe-agent", reasoning_effort=effort)._command_parts("t"))
    assert f"model.model_kwargs.reasoning_effort={effort}" in command
    assert "reasoning_effort" not in " ".join(_agent("mini-swe-agent", reasoning_effort=None)._command_parts("t"))


def test_mini_swe_agent_env_is_offline_safe_and_anthropic_native():
    env = _agent("mini-swe-agent")._harness_env()
    # litellm's anthropic transport appends /v1/messages itself
    assert env["ANTHROPIC_API_BASE"] == "http://gw.internal"
    # otherwise litellm blocks on raw.githubusercontent.com at import
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert env["MSWEA_CONFIGURED"] == "1"  # no first-run wizard
    assert env["MSWEA_GLOBAL_CALL_LIMIT"] == "0"


def test_mini_swe_agent_stages_a_config_derived_from_the_installed_mini_yaml():
    env = FakeEnv()
    agent = _agent("mini-swe-agent", env)
    agent._stage_files()
    command = next(c for c in env.commands if "mkconfig.py" in c)
    # the staged config is derived from the *installed* upstream mini.yaml so a
    # version bump picks up upstream's prompt instead of a stale copy
    assert "site-packages/minisweagent/config/mini.yaml" in command
    assert command.startswith("/opt/mimo-mini-swe-agent/venv/bin/python ")
    script = env.staged_text("mkconfig.py")
    assert 'agent.pop("mode", None)' in script  # InteractiveAgent-only knob
    assert 'agent["max_consecutive_format_errors"] = max_fmt_errors' in script


def test_mini_swe_agent_status_comes_from_exit_status_not_the_exit_code():
    def traj(exit_status, usage=None):
        return json.dumps(
            {
                "info": {"exit_status": exit_status},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "the answer",
                        "extra": {"response": {"usage": usage or {}}},
                    }
                ],
            }
        )

    submitted = FakeEnv(files={"traj-0.json": traj("Submitted")})
    agent = _agent("mini-swe-agent", submitted)
    assert agent._parse_result(0, []) == ("the answer", agent.IDLE_STATUS)
    # a step/cost limit stop also exits 0 — it must not read as Completed
    limited = FakeEnv(files={"traj-0.json": traj("LimitsExceeded")})
    agent = _agent("mini-swe-agent", limited)
    assert agent._parse_result(0, []) == ("the answer", agent.ERROR_STATUS)


def test_mini_swe_agent_usage_subtracts_cache_from_litellm_prompt_tokens():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 20,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 50,
    }
    rows = MiniSweAgent._message_usage([{"role": "assistant", "extra": {"response": {"usage": usage}}}])
    # litellm's prompt_tokens is all-in; Anthropic's input_tokens is the fresh
    # part only, so input would inflate ~20x without the subtraction
    assert rows == [{"input": 50, "output": 20, "cache_read": 900, "cache_write": 50}]


# -- payload distribution & the hermes thinking-retention patch ------------------


@pytest.mark.parametrize("name", NAMES)
def test_example_config_pins_a_version_only(name):
    # Public configs select a pinned version only. The per-harness installer
    # maps that version to the upstream public source; URL/path fields remain
    # offline-mirror overrides on HarnessAgentConfig and must not leak into
    # normal examples.
    config = (Path(__file__).parents[2] / f"example_configs/{name}.yaml").read_text()
    assert "\n  payload_url:" not in config
    assert "\n  payload_path:" not in config


def test_hermes_replays_the_full_thinking_history():
    # upstream keeps thinking blocks only on the latest assistant message, which
    # rewrites the cached prefix every turn (measured: one full cache miss) and
    # loses the earlier reasoning from the trajectory
    assert _agent("hermes")._harness_env()["HERMES_KEEP_ALL_THINKING"] == "1"
    installer = _install_script("hermes")
    assert "hermes: thinking-retention anchor moved" in installer
    assert 'not os.environ.get("HERMES_KEEP_ALL_THINKING")' in installer
    # the self-check must exercise the real function, both ways
    assert "upstream default changed" in installer
    assert "keep-all patch ineffective" in installer


@pytest.mark.parametrize("name", NAMES)
def test_context_and_output_limits_match_the_channel(name):
    # measured off a gateway's own 400s: context 1000000, output ceiling
    # 128000 for claude-opus-5. A stale 262144 here
    # makes the harness compact at a quarter of the real window.
    config = HARNESSES[name][1]()
    for field, expected in (("context_limit", 1_000_000), ("context_window", 1_000_000)):
        value = getattr(config, field, None)
        if value is not None:
            assert value == expected, f"{name}.{field}={value}"
    for field in ("output_limit", "max_tokens"):
        value = getattr(config, field, None)
        if value is not None:
            assert 0 < value <= 128_000, f"{name}.{field}={value} exceeds the 128000 ceiling"


def test_openclaw_and_kimi_harnesses_unpin_their_output_ceilings():
    # both upstreams cap the wire max_tokens below what the channel serves:
    # openclaw clamps a non-Sonnet-5 model to 32000, kimi-cli hardcodes 50000
    assert _agent("openclaw")._harness_env()["OPENCLAW_USE_MODEL_MAX_TOKENS"] == "1"
    assert _agent("kimi-cli")._harness_env()["KIMI_CLI_MAX_TOKENS"] == "65536"
    assert _agent("kimi-cli", output_limit=32000)._harness_env()["KIMI_CLI_MAX_TOKENS"] == "32000"
    oc = _install_script("openclaw")
    assert "openclaw: max_tokens clamp anchor moved" in oc
    assert "OPENCLAW_USE_MODEL_MAX_TOKENS" in oc
    kc = _install_script("kimi-cli")
    assert "KIMI_CLI_MAX_TOKENS" in kc
    assert "default_max_tokens=50000" in kc  # the anchor the patch replaces


def test_openclaw_preserves_every_turns_thinking():
    # shouldPreserveThinkingBlocks() matches new Claude families as
    # `claude-[5-9]`, which "claude-opus-5" misses -> dropThinkingBlocks strips
    # the signed thinking of every assistant turn but the latest
    installer = _install_script("openclaw")
    assert "openclaw: thinking-preservation allowlist anchor moved" in installer
    assert 'id.includes("fable-5") || id.includes("opus-5") || id.includes("opus-4")' in installer


def test_kimi_code_survives_an_exhausted_inotify_budget():
    # a failed chokidar watch escapes as an unhandled rejection and kills the
    # turn with rc=1; polling two small dirs cannot ENOSPC
    env = _agent("kimi-code")._harness_env()
    assert env["CHOKIDAR_USEPOLLING"] == "1"
    assert env["CHOKIDAR_INTERVAL"] == "1000"


# -- per-harness install scripts -------------------------------------------------

_RESOURCES_DIR = Path(__file__).parents[2] / "src/mimoagent/agents/blackbox/resources"
# Hosts, paths and mechanisms that must never appear in an installer: internal
# mirrors, the retired static-busybox downloader, and the self-deleting trap
# that only existed to hide a private CDN location from the rollout.
_FORBIDDEN_INSTALLER_STRINGS = (
    "busybox",
    "trap 'rm -f \"$0\"'",
)


def _install_script(name):
    return (_RESOURCES_DIR / f"install-{name}.sh").read_text()


def _all_installer_paths():
    return sorted(_RESOURCES_DIR.glob("install-*.sh"))


@pytest.mark.parametrize("name", NAMES)
def test_each_harness_owns_an_install_script(name):
    # same shape for every harness: source the shared library, pin the
    # adapter's version, install from the upstream public source unless a
    # payload override is given, verify, write the marker, print the sentinel
    script = _install_script(name)
    prefix = name.upper().replace("-", "_")
    version = HARNESSES[name][1]().version
    assert '. "${MIMO_INSTALL_COMMON:-$(dirname "$0")/install-common.sh}"' in script
    assert f'VERSION="${{{prefix}_VERSION:-{version}}}"' in script
    assert f'DEST="/opt/mimo-{name}"' in script
    assert 'MARKER="$DEST/.mimo-payload-version"' in script
    # libc detection feeds the musl/glibc source selection everywhere
    assert "mimo_detect_libc" in script
    # payload override first, upstream public source otherwise
    assert f'payload_override {prefix} "$DEST" || install_public' in script
    assert "install_public()" in script
    # idempotency needs both halves: marker + a probe of the unpacked tree
    assert "has_wanted()" in script and "probe()" in script
    assert 'mimo_marker_write "$MARKER" "$VERSION"' in script
    # the sentinel the adapter greps for
    assert f'echo "{prefix}_INSTALL_OK"' in script


@pytest.mark.parametrize("path", _all_installer_paths(), ids=lambda p: p.name)
def test_install_script_parses_as_bash(path):
    import subprocess

    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


@pytest.mark.parametrize("path", _all_installer_paths(), ids=lambda p: p.name)
def test_install_scripts_have_no_internal_hosts_or_retired_mechanisms(path):
    text = path.read_text()
    for needle in _FORBIDDEN_INSTALLER_STRINGS:
        assert needle not in text, f"{path.name} contains {needle!r}"


def test_install_scripts_download_only_through_the_shared_helpers():
    # every network access goes through mimo_fetch / fetch_* so mirror
    # overrides (NPM_REGISTRY, PIP_INDEX_URL, GITHUB_BASE, ...) apply uniformly
    for path in _all_installer_paths():
        if path.name == "install-common.sh":
            continue
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("curl ") and not stripped.startswith("wget "), (
                f"{path.name}: raw downloader call: {stripped}"
            )


def test_install_common_defines_the_shared_contract():
    common = (_RESOURCES_DIR / "install-common.sh").read_text()
    for fn in (
        "mimo_detect_libc()",
        "ensure_downloader()",
        "mimo_fetch()",
        "fetch_npm_tarball()",
        "fetch_node()",
        "fetch_cpython()",
        "fetch_github_release()",
        "fetch_rg()",
        "mimo_pip_install()",
        "payload_override()",
        "mimo_marker_matches()",
        "mimo_marker_write()",
        "mimo_wrapper()",
        "mimo_verify_version()",
    ):
        assert fn in common, fn
    # public defaults, each env-overridable
    assert "${NPM_REGISTRY:-https://registry.npmjs.org}" in common
    assert "${NODE_DIST:-https://nodejs.org/dist}" in common
    assert "${PBS_BASE:-https://github.com/astral-sh/python-build-standalone/releases/download}" in common
    assert "${GITHUB_BASE:-https://github.com}" in common
    assert "PIP_INDEX_URL" in common


@pytest.mark.parametrize("name", ["mini-swe-agent", "kimi-cli", "hermes"])
def test_python_harnesses_install_a_standalone_cpython(name):
    script = _install_script(name)
    assert 'fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$DEST/python"' in script
    assert "-m venv" in script
    assert "mimo_pip_install" in script


@pytest.mark.parametrize("name", ["kimi-code", "openclaw", "dsh"])
def test_node_harnesses_install_a_pinned_node(name):
    script = _install_script(name)
    assert 'fetch_node "$NODE_VERSION" "$DEST/node"' in script


@pytest.mark.parametrize("name", ["opencode", "kilocode"])
def test_bun_binary_harnesses_pick_the_musl_package_on_musl(name):
    script = _install_script(name)
    assert '[ "$MIMO_LIBC" = "musl" ]' in script
    assert "-musl" in script


def test_claude_code_install_picks_musl_cpython_and_platform():
    script = (_RESOURCES_DIR / "install-claude-code.sh").read_text()
    assert 'fetch_cpython "$CPYTHON_VERSION" "$CPYTHON_TAG" "$PYTHON_INSTALL_DIR/python"' in script
    assert 'mimo_pip_install "$PYTHON_BIN" "$CLAUDE_SDK_SPEC"' in script
    # the CLI asset is selected per libc from the official release bucket
    assert "$CLAUDE_RELEASES_BASE/$version/$MIMO_PLATFORM/claude" in script
    assert "claude-code-releases" in script


def test_mimocode_install_falls_back_to_a_static_rg():
    # mimo needs a working `rg` sibling; the release tarball carries only the
    # binary, so the install fetches a static ripgrep instead of aborting.
    script = (_RESOURCES_DIR / "install-mimocode.sh").read_text()
    assert 'fetch_rg "$RG_BIN"' in script
    assert 'fetch_github_release "XiaomiMiMo/MiMo-Code" "v${MIMOCODE_VERSION}"' in script
    assert "Re-upload the asset with rg included" not in script


def test_codex_install_ships_the_code_mode_host():
    # codex >= 0.153 spawns `codex-code-mode-host` from its own bin dir for
    # models whose tool_mode is code_mode_only and fails closed (zero tools)
    # when it is missing. The installer must therefore unpack the upstream
    # codex-package bundle, not a bare binary, and refuse to call a host-less
    # tree "installed" so stale single-binary pods get upgraded.
    script = (_RESOURCES_DIR / "install-codex.sh").read_text()
    assert 'PACKAGE_ASSET="codex-package-x86_64-unknown-linux-musl.tar.gz"' in script
    assert 'HOST_BIN="$CODEX_BIN_DIR/codex-code-mode-host"' in script
    assert "bundle_complete" in script
    assert 'fetch_github_release "openai/codex" "$tag" "$PACKAGE_ASSET"' in script
    # the agent shell sees both the bin dir (codex + host) and the bundled rg
    from mimoagent.agents.blackbox.codex import _CODEX_BIN_DIR, _CODEX_PATH_DIR

    assert _CODEX_BIN_DIR == "/opt/mimo-codex/bin"
    assert _CODEX_PATH_DIR == "/opt/mimo-codex/codex-path"


@pytest.mark.parametrize("name", NAMES)
def test_install_runs_the_harness_own_script_with_its_env_knobs(name):
    env = FakeEnv()
    agent = HARNESSES[name][0](_Model(), env, version="9.9.9")
    agent._install()
    prefix = name.upper().replace("-", "_")
    uploaded = [dst for _src, dst in env.copies]
    assert f"/tmp/mimo-install-{name}.sh" in uploaded
    command = next(c for c in env.commands if "bash /tmp/mimo-install-" in c)
    assert f"{prefix}_VERSION=9.9.9" in command
    assert f"bash /tmp/mimo-install-{name}.sh" in command
    # no payload_path / payload_url = the script's upstream public source
    assert f"{prefix}_PAYLOAD=" not in command
    assert f"{prefix}_PAYLOAD_URL=" not in command


def test_install_passes_an_explicit_payload_url_through():
    env = FakeEnv()
    agent = HARNESSES["grok"][0](_Model(), env, version="1.0.4", payload_url="http://mirror/grok.tar.gz")
    agent._install()
    command = next(c for c in env.commands if "bash /tmp/mimo-install-" in c)
    assert "GROK_PAYLOAD_URL=http://mirror/grok.tar.gz" in command


def test_install_rejects_payload_path_and_url_together():
    agent = HARNESSES["grok"][0](
        _Model(), FakeEnv(), version="1.0.4", payload_url="http://mirror/x", payload_path="/tmp/x"
    )
    with pytest.raises(ValueError, match="at most one"):
        agent._install()


def test_install_requires_a_pinned_version():
    agent = HARNESSES["grok"][0](_Model(), FakeEnv(), version="")
    with pytest.raises(ValueError, match="version must be pinned"):
        agent._install()


def test_install_env_is_forwarded_to_the_installer():
    env = FakeEnv()
    agent = HARNESSES["grok"][0](
        _Model(), env, version="1.0.4", install_env={"NPM_REGISTRY": "https://npm.example.com", "X": "a b"}
    )
    agent._install()
    command = next(c for c in env.commands if "bash /tmp/mimo-install-" in c)
    assert "NPM_REGISTRY=https://npm.example.com " in command
    assert "X='a b' " in command


# -- every install stages the shared helper library ---------------------------------
# install-common.sh carries the download / libc / upstream-source helpers every
# installer sources. Both adapter families must stage it: the HarnessAgent base
# _install and each of pi/codex/mimocode/claude-code, which override _install.


def test_stage_install_common_uploads_the_library_once_per_env():
    from mimoagent.agents.blackbox.install_common import POD_INSTALL_COMMON, stage_install_common

    env = FakeEnv()
    prefix = stage_install_common(env)
    assert prefix == f"MIMO_INSTALL_COMMON={POD_INSTALL_COMMON} "
    uploaded = [dst for _src, dst in env.copies]
    assert POD_INSTALL_COMMON in uploaded
    # Idempotent per env: a second call re-uses the staged library.
    before = len(env.copies)
    stage_install_common(env)
    assert len(env.copies) == before


@pytest.mark.parametrize("name", ["dsh", "mini-swe-agent"])
def test_harness_family_install_stages_the_library(name):
    env = FakeEnv()
    HARNESSES[name][0](_Model(), env, version="9.9.9")._install()
    uploaded = [dst for _src, dst in env.copies]
    assert "/tmp/mimo-install-common.sh" in uploaded
    command = next(c for c in env.commands if "bash /tmp/mimo-install-" in c)
    assert command.startswith("MIMO_INSTALL_COMMON=/tmp/mimo-install-common.sh ")


@pytest.mark.parametrize("name", ["pi", "codex", "mimocode", "claude-code"])
def test_custom_install_family_stages_the_library(name):
    env = FakeEnv()
    make_agent(name, _Model(), env, version="9.9.9")._install()
    uploaded = [dst for _src, dst in env.copies]
    assert "/tmp/mimo-install-common.sh" in uploaded
    command = next(c for c in env.commands if "bash /tmp/mimo-install-" in c)
    assert "MIMO_INSTALL_COMMON=/tmp/mimo-install-common.sh " in command


# -- the ask-the-user policy, in one place --------------------------------------
# This got reversed twice, so the rule and its evidence live together: an
# ask-class tool is kept whenever the harness resolves it itself under
# yolo/headless, and dropped only when calling it costs the turn. Gaps below are
# from /tmp/bbh/ask_probe.sh against a local mock (zero gateway traffic): the
# mock answers request #1 with a tool_use for the ask tool and the gap is how
# long the agent loop stalled.
def test_ask_class_tools_stay_on_the_wire_where_they_resolve_themselves():
    # grok: 0.004s → "no user is available, continue with your best judgment"
    assert GrokAgentConfig().disallowed_tools is None
    # kimi-code: 0.053s → "disabled while auto permission mode is active"
    assert "AskUserQuestion" in KimiCodeAgentConfig().tools
    # kimi-cli: 0.024s → afk auto-dismiss, {"answers": {}}
    assert "kimi_cli.tools.ask_user:AskUserQuestion" in _agent("kimi-cli")._agent_yaml()
    # hermes: 5.3s, bounded only by clarify.timeout (yolo does not reach clarify)
    assert "clarify" in (HermesAgentConfig().toolsets or "")
    # kilocode / dsh: their ask tools are not registered in headless at all, so
    # nothing is denied on their behalf
    from mimoagent.agents.blackbox.dsh import _DISABLED_ROWS
    from mimoagent.agents.blackbox.kilocode import _DENY_TOOLS

    assert _DENY_TOOLS == [] and "user-questions" not in _DISABLED_ROWS


def test_off_trajectory_tools_stay_closed_everywhere():
    from mimoagent.agents.blackbox.dsh import _DISABLED_ROWS

    # The two classes closed on every harness: an extra LLM call per session,
    # and the planning tools that some gateways fingerprint as third-party clients.
    assert "session-title-llm" in _DISABLED_ROWS
    assert json.loads(_agent("kilocode")._provider_config())["permission"]["todowrite"] == "deny"
