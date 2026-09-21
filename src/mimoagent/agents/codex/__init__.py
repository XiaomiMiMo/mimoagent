"""Native Codex agent (``agent.type: codex-agent``).

The agent loop lives in :mod:`mimoagent.agents.codex.codex_agent`; ``prompts/``
holds its system prompt. The ``.txt`` resources are picked up by
``packages.find`` and shipped as package data. Its Codex-aligned tool catalogue
is :mod:`mimoagent.tools.codex`.
"""
