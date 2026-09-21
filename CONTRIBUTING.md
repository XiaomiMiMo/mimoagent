# Contributing

Thanks for your interest in mimoagent. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/XiaomiMiMo/mimoagent.git
cd mimoagent
uv sync --all-extras
uv run pre-commit install
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. `AGENTS.md`
describes the codebase layout and the conventions for adding agents, tools,
harnesses and datasets.

## Before opening a pull request

```bash
uv run ruff check --fix && uv run ruff format
uv run pytest -q                          # integration tests skip without Docker / services
uv lock --check                           # if you touched dependencies
```

- Keep pull requests focused; describe the behaviour change and how you tested it.
- Add or update tests next to the code you change (`tests/` mirrors `src/`).
- Pin new third-party harness versions in the adapter, its installer and its
  example config; `tests/agents/test_blackbox_harnesses.py` checks they agree.
- Write code, comments, docs and commit messages in English.
- Never commit credentials, API keys, private hostnames, datasets or run outputs.

## Reporting security issues

See [SECURITY.md](SECURITY.md).
