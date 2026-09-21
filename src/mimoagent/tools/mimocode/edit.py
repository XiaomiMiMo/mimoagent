"""MiMo-facing Edit wrapper over the copied CC execution implementation."""

import shlex
from typing import Any

from mimoagent.tools.base import ToolOutput
from mimoagent.tools.mimocode.edit_cc_impl import EditTool as CCEditTool
from mimoagent.tools.mimocode.paths import resolve_path, session_cwd
from mimoagent.tools.mimocode.prompts import load_tool_prompt

# The diff travels in metadata, which the agent renders into the model-facing
# observation. Editing a minified single-line file yields a diff far larger than
# the whole observation budget, so cap it at the source too rather than relying
# only on the agent-side metadata bound.
_MAX_DIFF_CHARS = 2000


class EditTool(CCEditTool):
    """Use the CC-safe exact replacement engine with MiMo protocol metadata."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return load_tool_prompt("edit")

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        if isinstance(params, dict) and params.get("file_path"):
            params = {**params, "file_path": resolve_path(str(params["file_path"]), context)}
        if isinstance(params, dict) and params.get("old_string") == "":
            return self._create_file(params, context)
        if isinstance(params, dict) and params.get("old_string") not in (None, ""):
            file_path = params.get("file_path")
            state = (context or {}).get("state")
            if isinstance(file_path, str) and (
                not isinstance(state, dict) or file_path not in state.get("read_files", set())
            ):
                return ToolOutput(
                    output=f"Error: You must read {file_path} before editing it.",
                    success=False,
                )
        result = super().execute(params, context)
        if result.success:
            # Upstream returns the fixed success line and keeps the diff for the
            # UI. Here metadata reaches the model, so bound it.
            diff = result.output or ""
            if len(diff) > _MAX_DIFF_CHARS:
                diff = f"{diff[:_MAX_DIFF_CHARS]}\n[... diff truncated: {len(result.output)} chars total ...]"
            result.metadata = {**result.metadata, "diff": diff}
            result.output = "Edit applied successfully."
        return result

    def _create_file(self, params: dict[str, Any], context: dict[str, Any] | None) -> ToolOutput:
        """Handle the documented ``old_string=""`` shorthand for a brand-new file.

        Only for a file that does not exist yet. The engine underneath refuses an
        empty ``old_string`` outright, and routing past that refusal into a write
        turns a typo into silent truncation of the file being fixed: the read gate
        only asks that the path was read, not that it is new.
        """
        from mimoagent.tools.mimocode.write import WriteTool

        file_path = params.get("file_path")
        env = (context or {}).get("env")
        if isinstance(file_path, str) and env is not None:
            probe = f"[ -e {shlex.quote(file_path)} ] && echo EXISTS || echo NEW"
            if env.execute(probe, cwd=session_cwd(context) or None).get("output", "").strip() == "EXISTS":
                return ToolOutput(
                    output=(
                        f"Error: {file_path} already exists. An empty old_string only creates a new file. "
                        "Pass the exact text you want replaced, or use write to overwrite the file."
                    ),
                    success=False,
                )
        result = WriteTool({}).execute(
            {"file_path": file_path, "content": params.get("new_string", "")},
            context,
        )
        if result.success:
            # Keep edit's own success line: the model called edit, and a
            # write-shaped result would teach it the wrong tool/observation pair.
            result.output = "Edit applied successfully."
        return result


__all__ = ["EditTool"]
