# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

Initial public release.

mimoagent began as a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
v1.9.0 (commit `b3d50788`, ) and was developed internally before this
release. Compared with the upstream project it adds:

- **Agents**: a native tool-calling loop with parallel tool execution and
  subagents (`default`, `bashonly-agent`), Claude-Code-style (`cc-agent`),
  Codex-style (`codex-agent`, with the code-mode host) and MiMo-Code-style
  (`mimocode-agent`) tool catalogues, model-decided context compaction, a
  simulated-user sidecar, and an anti-reward-hacking interceptor.
- **Blackbox harnesses**: adapters that install and run third-party coding
  CLIs inside the task environment — Claude Code, Codex CLI, MiMo-Code,
  OpenCode, Pi, Grok Build, Kimi Code, Kimi CLI, Kilo Code, OpenClaw, Oh My Pi,
  Hermes Agent, DeepSeek Harness and mini-swe-agent — installed from their
  public upstream distributions at pinned versions, with an offline payload
  path.
- **Models**: native OpenAI Chat Completions, OpenAI Responses and Anthropic
  Messages adapters with streaming, thinking-block replay, multimodal content
  and token accounting; no litellm.
- **Environments**: Kubernetes pods (with detached execution for hours-long
  sessions and chunked file transfer), Docker, CubeSandbox micro-VMs, Modal
  Sandboxes, local.
- **Datasets and reward**: SWE-bench, SWE-bench Pro, DeepSWE, Terminal-Bench,
  DeepSWE, opensource-code, ARVO, and a generic
  git dataset; a rubric-judge reward stage that stacks on any dataset; git
  leak prevention and build-artifact cleanup.
- **Batch runner**: parallel workers, multiple rollouts per instance, resume,
  ground-truth replay (`--recalc-input gt`), random config assignment, and a
  Ray-distributed variant.
