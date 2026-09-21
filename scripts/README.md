# Scripts

Standalone utilities around a batch run. They are not part of the `mimoagent`
package and may need extra packages; the simplest way to run one is with
`uv run --with <deps> python scripts/<script>.py ...`.

| Script | Purpose | Extra deps |
|---|---|---|
| `collect.py` | Collect trajectories from `outputs/<prefix>*` run directories into one JSONL (skips errored runs). | `tqdm` |
| `collect_fc_trajs.py` | Collect native tool-calling (function-call) trajectories into a flat SFT-style JSONL. | `typer`, `tqdm` |
| `collect_responses_trajs.py` | Same for OpenAI Responses API trajectories (`protocol: responses` runs), preserving raw response items. | `typer`, `tqdm` |
| `convert_mini_swe_traj_to_sft.py` | Convert `traj.json` files into the SFT `messages` format. | `tqdm` |
| `convert_2_train_json.py` | Extract the `messages` field of a JSONL into a training JSON. | — |
| `exp_stats.py` | Per-experiment statistics over `traj.json` files (rewards, exit statuses, token usage). | `rich` (already a project dependency) |
| `convert_deepswe.py` | Convert a checkout of the public [DeepSWE](https://github.com/datacurve-ai/deep-swe) `tasks/` directory into `batch.py` rows (JSONL), reference solutions included. | — |
| `streamlit_rollout_traj.py` | Web viewer for rollout trajectory JSON files (paging, pass rate, side-by-side trajectories). | `streamlit` |
| `view_jsonl.py` | Web viewer for JSONL message data. | `streamlit` |
| `harness_payloads/` | Build offline payload tarballs for the blackbox harness adapters; see its README. | see README |

Examples:

```bash
uv run --with tqdm python scripts/collect.py --prefix outputs/my-run -o my-run.jsonl
uv run --with streamlit python -m streamlit run scripts/streamlit_rollout_traj.py -- --input_file rollouts.json
```
