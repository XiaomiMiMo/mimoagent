# Repository Guide

## What This Project Does

`mimoagent` is an agent framework for software-engineering tasks such as SWE-bench, repository repair, code review, and rollout/evaluation data collection. It separates three concerns: an agent decides what to do, a model produces text or tool calls, and an environment executes commands and owns the testbed. The same batch runner can combine different agent types, model protocols, datasets, and environment backends through YAML configuration.

The main user-facing command is the `mimoagent` entry point, implemented by `mimoagent.run.extra.batch:app`. A typical run loads a dataset, creates a dataset-specific environment around a base backend, constructs the selected model and agent, runs the task loop, records messages and tool results, then calculates the dataset reward from the resulting repository state.

The project is a fork of mini-swe-agent v1.9.0 (MIT); see `NOTICE`. The package, CLI and most of the code have since been rewritten.

## Architecture

### Run and Configuration

- `src/mimoagent/run/extra/batch.py` is the main batch CLI. It handles dataset slicing, shuffling, workers, retries/resume behavior, logging, trajectory saving, model statistics, and reward calculation.
- `src/mimoagent/run/extra/ray_batch.py` provides Ray-based distributed batch execution (`uv sync --extra ray`).
- `src/mimoagent/config/__init__.py` resolves config paths and expands `${VAR}` / `${VAR:-default}` references in string values when a config is loaded (`expand_env_vars`).
- `example_configs/` contains runnable profiles. The `agent.type`, `model.protocol`, `model.model_name`, and `environment.environment_class` fields select the major runtime components. Example configs read credentials and endpoints from environment variables; never write them into the files.

### Agents

- `src/mimoagent/agents/base.py` owns the common lifecycle: render system and task templates, query the model, dispatch a step, enforce limits, record messages, and return statuses such as `Idle`, `LimitsExceeded`, `ModelQueryError`, or `InfraError`.
- `src/mimoagent/agents/default.py` is the standard native tool-calling loop. It parses model tool calls, executes one call synchronously or multiple calls in parallel, emits tool results, and supports the lowercase `bash`, `read`, `write`, `edit`, and `agent` tools. Pre-dispatch policies such as the anti-hack guard register through `add_action_interceptor` instead of overriding `execute_action`; agents that decorate real tool results override `_execute_tool`.
- `src/mimoagent/agents/bashonly/` is the native `DefaultAgent` variant for the `bashonly-agent` profile. It keeps the default lifecycle while exposing only the isolated bash-only tool profile.
- `src/mimoagent/agents/cc/cc_agent.py` is the native Claude-Code-aligned loop. It uses the same `BaseAgent` lifecycle and model interface as `DefaultAgent`, but uses the capitalized Claude-style tool catalogue (`Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob`, `Agent`, `Compact`) and Claude-oriented schemas. Compaction and the optional antihack guard are integrated at this layer.
- `src/mimoagent/agents/codex/` is the native Codex-aligned loop. With `ptc: true` it exposes the codex-rs `exec` + `wait` surface and runs JavaScript in the standalone `codex-code-mode-host` (resolved by `tools/codex/code_mode_host.py`: the agent's `code_mode_host_path`, `MIMOAGENT_CODE_MODE_HOST_PATH`, or a cached download of the pinned `openai/codex` release asset); Codex-style `exec_command` and `apply_patch` remain nested tools on the JavaScript `tools` object. `ptc: false` exposes those tools directly. `CodexAgent` is responses-only.
- `src/mimoagent/agents/mimocode/` is the native loop with the MiMo-Code tool catalogue (`tools/mimocode/`).
- `src/mimoagent/agents/blackbox/` contains adapters for external coding CLIs: Claude Code, Codex CLI, MiMo-Code, OpenCode, Pi, Grok, Kimi Code, Kimi CLI, Kilo Code, OpenClaw, Oh My Pi, Hermes, DeepSeek Harness and mini-swe-agent. These adapters share the agent constructor contract but run the third-party scaffold inside the environment rather than using the native Python tool loop. Each harness owns an `install-<name>.sh` in `agents/blackbox/resources/` that installs a pinned version from the upstream public source (npm registry, GitHub releases, PyPI) and sources the shared `install-common.sh`; `payload_path` / `payload_url` install a prebuilt tarball instead (`scripts/harness_payloads/`), and `install_env` forwards mirror overrides. The hours-long harness session goes through `Environment.execute_detached` (see `agents/blackbox/detached.py`): on Kubernetes that launches the harness in its own process group and polls an rc marker with short execs, because exec websockets have a limited connection lifetime and cannot be re-attached.

### Claude-like and Codex-like Modes

There are two separate meanings of "Claude-like" and "Codex-like" in this repository:

1. Native modes: `agent.type: cc-agent` selects `CCAgent`, and `agent.type: codex-agent` selects the native `CodexAgent` (requires `protocol: responses`). Mimoagent owns the conversation, calls the configured model through its model adapter, dispatches tools, stores the trajectory, and controls termination. Only the tool catalogue and protocol-specific behavior differ.
2. Blackbox modes: `agent.type: claude-code` starts the Claude Agent SDK/Claude Code scaffold in the test environment, while `agent.type: codex` installs and starts the upstream OpenAI Codex CLI in the test environment. The subprocess edits the repository in place and writes its own raw session log; mimoagent supplies the task/environment, captures logs, and grades the resulting repository. See `example_configs/claude-code.yaml` and `example_configs/codex.yaml`.

For native Claude-style tool calling, use `example_configs/swe_cc_agent.yaml`. For the external Claude Code or Codex CLI behavior, use the corresponding blackbox config. Do not infer the mode from the model name alone; `agent.type` is the selector.

### Models and Protocols

- `src/mimoagent/models/__init__.py` selects the model adapter from `model.protocol`.
- `protocol: chat` uses the OpenAI Chat Completions adapter and is the default.
- `protocol: anthropic` uses the native Anthropic Messages adapter, preserving thinking blocks, tool use/results, multimodal image content, streaming, and prompt-cache metadata.
- `protocol: responses` uses the OpenAI Responses adapter, translating chat-style messages and tool calls to Responses input/output items while preserving raw response items for replay.
- Blackbox agents do not query through the model object for their agent loop; they use the shared model configuration for gateway, model name, and credentials.

### Environments and Datasets

- `src/mimoagent/environments/__init__.py` maps `environment_class` names to base backends: `local`, `docker`, `kubernetes`, `cube`, and `modal`.
- `src/mimoagent/environments/kubernetes.py` is the primary remote backend and executes in Kubernetes pods. Its config carries only generic knobs (`kubeconfig`, `namespace`, `node_selector`, `tolerations`, `image_pull_secrets`, resources, `labels`, `annotations`, `host_network`, `answer_leak_blocklist`); cluster-specific policy belongs in the user's config, not in code defaults.
- `src/mimoagent/environments/local.py` and `docker.py` provide local/container execution for development and controlled runs.
- `src/mimoagent/environments/cube.py` communicates with CubeSandbox through the `cubesandbox` SDK (the `cube` extra); endpoints come from `CUBE_API_URL` / `CUBE_SANDBOX_DOMAIN`.
- `src/mimoagent/environments/modal.py` runs each task in a [Modal](https://modal.com) Sandbox through the `modal` SDK (the `modal` extra). The sandbox is created from the task's registry image under a Modal App (`app_name`), with credentials from the SDK's own configuration (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` or `~/.modal.toml`); `sandbox_timeout`, `cpu`, `memory`, `gpu`, `region`, `block_network`, `secrets` and `registry_secret` are the main knobs.
- `src/mimoagent/environments/detached.py` is the `DetachedExecMixin` behind `execute_detached` on the Kubernetes and Modal backends: it launches the command in its own session with stdio on remote files and polls an rc marker with short execs, so an hours-long harness session never depends on one long-lived exec connection, and an optional stall watchdog kills the process group when its output files stop changing.
- `src/mimoagent/environments/datasets/` contains dataset-specific setup and reward implementations for DeepSWE, opensource-code, ARVO, and generic. Instances declare `dataset_type`.
- `src/mimoagent/environments/rubric_judge.py` is the rubric judge, a reward stage rather than a dataset. Any dataset's instance may carry `rubric.rubrics`; when the yaml `environment.judge_agent` block is also set, `DatasetEnvironment.calculate_reward` runs an in-pod judge agent before the dataset's own verifier and reports both scores. The yaml `reward_mode` key (`auto`, `programmatic`, `rubric`, `both`) selects the stages; without rubrics or a judge config the dataset's programmatic reward is unchanged.
- `src/mimoagent/environments/utils.py` selects the base backend, resolves the task image (instance `docker_image`, dataset default, optional `image_prefix` mirror), filters environment config fields, and wraps the base backend in the dataset-specific environment.

### Tools

- `src/mimoagent/tools/` holds the default lowercase catalogue plus the Claude-style `tools/cc/`, Codex-style `tools/codex/` and MiMo-Code-style `tools/mimocode/` catalogues, each registered through its own registry.
- `src/mimoagent/tools/ripgrep.py` resolves the static `rg` binary the `Grep`/`Glob` tools stage into environments that lack ripgrep (`MIMOAGENT_RG_PATH`, the user cache, or a download of the pinned release).

## Repository Layout

- `src/mimoagent/agents/`: native agent loops, agent factory, subagents, antihack logic, and blackbox adapters.
- `src/mimoagent/models/`: model protocol adapters, token accounting, and content conversion helpers.
- `src/mimoagent/tools/`: tool catalogues and the ripgrep resolver.
- `src/mimoagent/environments/`: execution backends, dataset wrappers, reward calculation, and environment utilities.
- `src/mimoagent/run/`: CLI runners, distributed runners, trajectory persistence, and logging helpers.
- `src/mimoagent/compaction/`: context compaction prompts and message surgery.
- `src/mimoagent/config/`: config path resolution and `${VAR}` expansion.
- `example_configs/`: profiles for native agents, blackbox agents, model protocols, and environment backends.
- `tests/`: unit tests for agents, models, tools, environments and batch execution; tests marked `integration` need Docker or an external service and are skipped otherwise.
- `scripts/`: trajectory collection / conversion utilities, dataset converters, viewers, and the harness payload builder (see `scripts/README.md`).
- `docs/`: design notes and benchmark notes.

## Python Toolchain

- Use `uv` for project dependencies and command execution.
- The project is pinned to Python `==3.12.*`; use uv's Python 3.12 environment for development and execution.
- Do not install project dependencies with `pip`, create a separate virtual environment manually, or activate `.venv` by hand for normal development.
- Keep dependency changes in `pyproject.toml` and regenerate `uv.lock` with `uv lock`.

### Common Commands

```bash
uv sync                         # project plus the default dev group
uv sync --extra cube            # CubeSandbox backend
uv sync --extra modal           # Modal Sandbox backend
uv sync --extra ray             # Ray-distributed runner
uv run mimoagent --help         # run the batch CLI
uv run pytest                   # run tests
uv run ruff check               # lint
uv run ruff format              # format
```

## Dependency Policy

- Runtime dependencies belong in `[project].dependencies`; use the current lock version as the minimum for foundational libraries and an audited exact/minor pin for behaviorally important SDKs and framework packages.
- Developer-only tools belong in `[dependency-groups].dev`.
- User-facing optional runtime features belong in `[project.optional-dependencies]` so both `pip install -e ".[cube]"` and `uv sync --extra cube` work.
- Pin the Kubernetes client to the audited release; upgrade it deliberately with the Kubernetes environment tests.
- Pin the OpenAI and Anthropic SDKs exactly in `pyproject.toml`; update those pins deliberately after running the Chat, Responses, and Anthropic model tests.
- Do not hand-edit `uv.lock`; run `uv lock` after changing dependency metadata.

## Development Workflow

1. Read the relevant module, its factory/registry, and its tests before changing behavior.
2. For a new agent type, update the lazy registry in `agents/factory.py`, add or update an example config, and add focused tests.
3. For a new tool catalogue, keep the schema and execution contract aligned with the target scaffold and register it through the corresponding tool registry.
4. For a new blackbox harness, add the adapter under `agents/blackbox/`, an `install-<name>.sh` that sources `install-common.sh` and installs from the upstream public source, an example config pinning the version, and a builder function in `scripts/harness_payloads/build_harness_payloads.sh`; the tests in `tests/agents/test_blackbox_harnesses.py` check that the three agree.
5. For model protocol changes, preserve message replay and token accounting across both streaming and non-streaming paths.
6. After Python edits, run `uv run ruff check --fix` and `uv run ruff format` on changed files, then run focused tests.
7. Run `uv lock --check` after dependency changes.

## Runtime Notes

Batch runs require a model configuration, a dataset, and an environment backend. Kubernetes examples need access to a cluster (`KUBECONFIG`); the Docker example runs anywhere a Docker daemon is available. Blackbox agents install their CLI from public sources inside the task environment, so it needs outbound network access at setup time (or a prebuilt payload). Dataset reward code may mutate or sanitize the testbed, so use the environment and dataset wrappers rather than bypassing them in a runner.
