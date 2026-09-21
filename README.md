# MiMo Agent (`mimoagent`)

`mimoagent` is an engineering-first agentic rollout framework. It connects
agents, tools, environments, datasets and graders, and manages rollouts at scale.
Four concerns stay separate:

| Layer | What it covers |
|---|---|
| **Agent** | custom white-box agents and 10+ black-box coding CLIs (Claude Code, Codex, OpenCode, ...); agent loops run in or out of the pod, with different levels of intrusiveness into the environment |
| **Model** | three native protocols — OpenAI Chat, OpenAI Responses, Anthropic Messages — each through its official SDK |
| **Environment** | pluggable backends: local, Docker, Kubernetes pods, CubeSandbox, Modal Sandboxes |
| **Dataset** | adapters for datasets and benchmarks together with their graders: DeepSWE, opensource-code, ARVO, and generic |

`mimoagent` is a library: any agent, model protocol, dataset and backend can be
composed programmatically. It faithfully records trajectories in a training
schema and grades the final environment in place, ready for downstream
consumption.

`mimoagent` started as a fork of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) v1.9.0 (commit
`b3d50788`, August 2025) and has since been largely rewritten; see [NOTICE](NOTICE).

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/XiaomiMiMo/mimoagent.git
cd mimoagent
uv sync                      # project + dev tools
# uv sync --extra cube       # CubeSandbox backend
# uv sync --extra modal      # Modal Sandbox backend
# uv sync --extra ray        # Ray-distributed batch runner
```

`pip install -e .` works too.

## Quickstart (Docker)

The fastest way to see the whole loop is one
[DeepSWE](https://github.com/datacurve-ai/deep-swe) task on a local Docker
daemon. DeepSWE publishes its 113 tasks in Harbor layout with prebuilt images
on a public registry; convert them once into a mimoagent JSONL:

```bash
git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git
uv run python scripts/convert_deepswe.py --src deep-swe/tasks --out deepswe.jsonl
```

Check the grading path first — apply the reference solution, expect
`Recalculated - Pass`:

```bash
uv run mimoagent -c example_configs/swe_docker.yaml \
    --dataset deepswe.jsonl --slice 0:1 \
    --recalc-input gt -o outputs/smoke-gt
```

Then let a model solve it:

```bash
export OPENAI_API_KEY=...                  # and OPENAI_BASE_URL for a gateway
export MIMOAGENT_MODEL_NAME=gpt-5          # any model your endpoint serves
uv run mimoagent -c example_configs/swe_docker.yaml \
    --dataset deepswe.jsonl --slice 0:1 \
    -o outputs/smoke
```

Each instance gets a directory under `--output` with `instance.log`, the
trajectory (`traj.json`, `agent_msgs/`), and `reward_extra_info.json`;
`results.jsonl` collects one record per instance. `--workers N` runs instances
in parallel, `--num-rollouts K` samples several trajectories per instance,
`--filter` / `--slice` / `--shuffle` select instances, and `--redo-existing`
reruns finished ones.

## Configuration

A config has four blocks. Values may reference environment variables as
`${VAR}` or `${VAR:-default}`, so credentials and endpoints never need to be
written into files:

```yaml
use_dataset_env: true

agent:
  type: default                 # see "Agents" below
  tools: [{tool: bash, config: {timeout: 300}}, {tool: read}, {tool: write}, {tool: edit}]
  system_template: |
    You are an agent, your current working directory is {{cwd}}.
  instance_template: |
    Fix the following issue:

    {{task}}
  step_limit: 500

environment:
  environment_class: docker     # local | docker | kubernetes | cube | modal
  cwd: /testbed
  timeout: 300

model:
  model_name: "${MIMOAGENT_MODEL_NAME:-gpt-5}"
  protocol: chat                # chat | responses | anthropic
  model_kwargs:
    base_url: "${OPENAI_BASE_URL:-https://api.openai.com/v1}"
    api_key: "${OPENAI_API_KEY:-}"
    temperature: 1.0
```

`example_configs/` has a runnable profile for every agent type and backend.
`--override-config '{"agent": {"step_limit": 50}}'` patches any field from the
command line; passing several `-c` files assigns one config per instance at
random (for A/B comparisons).

## Agents

| `agent.type` | What runs | Notes |
|---|---|---|
| `default` | native tool-calling loop with `bash`, `read`, `write`, `edit`, `agent` (subagents) | parallel tool calls, optional anti-reward-hacking guard |
| `bashonly-agent` | the default loop with only `bash` | |
| `cc-agent` | native loop with the Claude Code tool catalogue (`Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob`, `Agent`, `Compact`) | model-decided context compaction |
| `codex-agent` | native loop with the Codex catalogue; `ptc: true` exposes `exec` + `wait` and runs JavaScript in the Codex code-mode host | requires `protocol: responses` |
| `mimocode-agent` | native loop with the [MiMo-Code](https://github.com/XiaomiMiMo/MiMo-Code) catalogue | |
| blackbox: `claude-code`, `codex`, `mimocode`, `opencode`, `pi`, `grok`, `kimi-code`, `kimi-cli`, `kilocode`, `openclaw`, `omp`, `hermes`, `dsh`, `mini-swe-agent` | the upstream coding CLI itself, installed and run inside the task environment | mimoagent supplies the task and grades the repository |

Blackbox adapters install a pinned version of the CLI from its public source
(npm registry, GitHub releases, PyPI) when the environment starts; the task
container therefore needs outbound network access at setup time. For offline
or mirrored setups, build a payload once with
`scripts/harness_payloads/build_harness_payloads.sh` and point the agent at it
with `payload_path` / `payload_url`; mirror URLs can be passed through
`install_env` (`NPM_REGISTRY`, `PIP_INDEX_URL`, `GITHUB_BASE`, ...).

## Environments

| `environment_class` | Backend | Notes |
|---|---|---|
| `local` | subprocesses on the host | development only |
| `docker` | one container per instance | `forward_env` passes proxies through |
| `kubernetes` | one pod per instance | `kubeconfig` (default `$KUBECONFIG` / `~/.kube/config`, in-cluster when running as a pod), `namespace` (`$K8S_NAMESPACE`), `node_selector`, `tolerations`, `image_pull_secrets`, resource requests/limits, `labels` / `annotations`; long harness sessions run detached because exec websockets have a limited lifetime |
| `cube` | [CubeSandbox](https://pypi.org/project/cubesandbox/) micro-VMs with snapshot / clone / rollback | `uv sync --extra cube`; `CUBE_API_URL`, `CUBE_SANDBOX_DOMAIN` |
| `modal` | one [Modal](https://modal.com/docs/guide/sandbox) Sandbox per instance, created from the task's registry image | `uv sync --extra modal`; credentials from `modal token new` or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`; `app_name`, `sandbox_timeout`, `cpu` / `memory` / `gpu`, `region`, `block_network`, `secrets`, `registry_secret` for private images; long harness sessions run detached like on Kubernetes |

`environment.image_prefix` prepends a registry mirror to every task image.

## Datasets

Rows can come from a Hugging Face dataset (`--dataset princeton-nlp/SWE-bench_Verified --split test`),
a parquet file, or a JSONL file. The dataset type is taken from the row's
`dataset_type` field or inferred from its image:

`deepswe`, `opensource-code`, `arvo`, and `generic`.

Any dataset's rows may additionally carry `rubric.rubrics`; with an
`environment.judge_agent` block, an in-environment judge agent scores the
result before the dataset's own verifier runs (`example_configs/rubric-judge.yaml`).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [AGENTS.md](AGENTS.md)
for a map of the codebase.

## License

MIT — see [LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
