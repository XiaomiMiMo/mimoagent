"""Edit tool: exact-match string replacement in an existing file.

Implementation contract: behaves like ``content.replace(old_str, new_str, 1)``
on the file bytes, but only after verifying ``old_str`` occurs exactly once.

Container-side dependencies are bash + awk + the POSIX basics (tr, wc) only —
no python in the image. The old/new strings travel via ``env.copy_to`` (tar
stream), never on a command line — so there is no ARG_MAX ceiling and no
shell-quoting hazard.

awk reads the strings and the target file as single whole records via
``RS="\\1"`` (the SOH control byte). ``RS="\\0"`` would be the natural choice
but is NOT portable: busybox awk (alpine images) stores strings as C strings,
so ``"\\0"`` collapses to ``""`` which is awk *paragraph mode* — files with
blank lines get split and the edit fails. ``\\1`` is a real byte in every awk
(verified against gawk, mawk, busybox awk). Files containing NUL bytes are
rejected as binary by a bash-level ``tr -d '\\0' | wc -c`` check *before* awk
runs (a C-string awk would silently truncate the body at the first NUL);
files containing ``\\1`` itself are caught by awk's second-record probe.

Total environment round-trips per edit: one ``copy_to`` + one ``execute``.
"""

import os
import shlex
import tempfile
import uuid
from typing import Any

from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

# awk program: read old/new/target as whole records, enforce single
# occurrence, write the replaced content to OUT. Communicates via exit code:
#   0 = replaced   3 = not found   4 = multiple (count on stdout)
#   5 = \1 byte in input (cannot represent)   6 = read failure
_AWK_PROG = r"""
BEGIN {
    RS = "\1"; ORS = ""
    if ((getline old < OLDF) < 0) exit 6
    if ((getline x < OLDF) > 0) exit 5
    close(OLDF)
    if ((getline new < NEWF) <= 0) new = ""
    close(NEWF)
    if ((getline body < TARGET) < 0) exit 6
    if ((getline extra < TARGET) > 0) exit 5
    close(TARGET)

    count = 0; s = body
    while ((i = index(s, old)) > 0) { count++; s = substr(s, i + length(old)) }
    if (count == 0) exit 3
    if (count > 1) { print count; exit 4 }

    i = index(body, old)
    print substr(body, 1, i - 1) new substr(body, i + length(old)) > OUT
    close(OUT)
    exit 0
}
"""


class EditTool(BaseTool):
    """Replace an exact (possibly multi-line) string in an existing file."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return """Performs exact string replacements in files.

Usage:

- You must use your `read` tool at least once in the conversation before editing.
- When editing text from read tool output, ensure you preserve the exact indentation (tabs/spaces). Never include any part of the line number prefix in the old_str or new_str.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Tool calls in the same message run concurrently, so several edit or write calls on one file in the same message can race: the writes silently overwrite each other (each call still reports success), or a call fails with a spurious binary-file error.

Notes:

- `path` must be absolute (start with `/`) and must point to an existing file.
- Omit `new_str` (or pass an empty string) to delete `old_str`.
- Ensure the edit results in idiomatic, correct code; do not leave the file in a broken state.

IMPORTANT:

- EXACT MATCHING: `old_str` must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool fails if it matches multiple locations or does not match exactly.
- UNIQUENESS: `old_str` must uniquely identify a single instance in the file. Include sufficient context before and after the change point (3-5 lines recommended) so the match is unambiguous.
- REPLACEMENT: `new_str` replaces `old_str`. Both strings must be different."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (must match uniquely, including whitespace and newlines).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string. If omitted or empty, old_str is deleted.",
                },
            },
            "required": ["path", "old_str"],
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for edit execution")
        if not isinstance(params, dict):
            raise ToolException(f"Parameters must be a dictionary, got {type(params).__name__}")

        path = params.get("path")
        if not path:
            raise ToolException("Missing required parameter: path")
        if not path.startswith("/"):
            raise ToolException(f"Path must be absolute (start with /), got: {path}")

        old_str = params.get("old_str")
        if old_str is None:
            raise ToolException("Missing required parameter: old_str")
        new_str = params.get("new_str") or ""
        if old_str == "":
            return ToolOutput(
                output="Error: old_str must not be empty. To create or overwrite a file, use the write tool.",
                success=False,
            )
        if old_str == new_str:
            return ToolOutput(
                output="Error: old_str and new_str are identical — nothing to change.",
                success=False,
            )
        if any(c in old_str or c in new_str for c in ("\0", "\x01")):
            return ToolOutput(
                output="Error: old_str/new_str must not contain NUL or \\x01 control bytes.", success=False
            )

        remote_dir = f"/tmp/mimo_edit_{uuid.uuid4().hex[:12]}"
        self._ship_strings(env, remote_dir, old_str, new_str)

        q_target = shlex.quote(path)
        q_dir = shlex.quote(remote_dir)
        # Single script: existence check -> awk count+replace into $d/out ->
        # diff for the observation -> overwrite via cat (preserves the target's
        # permissions/inode, unlike mv) -> cleanup. The trap guarantees no
        # temp residue on any exit path.
        script = f"""
target={q_target}
d={q_dir}
trap 'rm -rf "$d"' EXIT
if [ ! -f "$target" ]; then echo "MIMO_EDIT:NOFILE"; exit 0; fi
if [ ! -r "$target" ] || [ ! -w "$target" ]; then echo "MIMO_EDIT:NOACCESS"; exit 0; fi
# Binary check BEFORE awk: busybox/mawk would silently truncate the body at
# the first NUL instead of failing. tr/wc are POSIX and O(n) in C.
if [ "$(wc -c < "$target")" -ne "$(tr -d '\\0' < "$target" | wc -c)" ]; then
  echo "MIMO_EDIT:BINARY"; exit 0
fi
count=$(awk -v OLDF="$d/old" -v NEWF="$d/new" -v TARGET="$target" -v OUT="$d/out" {shlex.quote(_AWK_PROG)})
rc=$?
case $rc in
  0) ;;
  3) echo "MIMO_EDIT:NOTFOUND"; exit 0 ;;
  4) echo "MIMO_EDIT:MULTIPLE:$count"; exit 0 ;;
  5) echo "MIMO_EDIT:BINARY"; exit 0 ;;
  *) echo "MIMO_EDIT:AWKFAIL:$rc"; exit 0 ;;
esac
echo "Replacement successful. Showing difference:"
# The scratch copy is 0644 from awk; give it the target's mode so the header
# does not announce a mode change that the cat below never makes.
chmod "$(stat -c '%a' "$target" 2>/dev/null || echo 644)" "$d/out" 2>/dev/null
if command -v git >/dev/null 2>&1; then
  git diff --no-index -- "$target" "$d/out" 2>/dev/null | head -50
elif command -v diff >/dev/null 2>&1; then
  diff -u -L "a$target" -L "b$target" "$target" "$d/out" 2>/dev/null | head -50
fi
cat "$d/out" > "$target" || {{ echo "MIMO_EDIT:WRITEFAIL"; exit 0; }}
echo "MIMO_EDIT:OK"
"""
        result = env.execute(script)
        output = result.get("output", "")

        if "MIMO_EDIT:OK" in output:
            # The diff's new side is the scratch copy; name it as the target so
            # the observation carries no per-call random path.
            body = output.replace("MIMO_EDIT:OK", "").replace(f"{remote_dir[1:]}/out", path.lstrip("/")).strip()
            return ToolOutput(output=body or "String replaced successfully.")
        if "MIMO_EDIT:NOFILE" in output:
            return ToolOutput(output=f"Error: File not found: {path}", success=False)
        if "MIMO_EDIT:NOACCESS" in output:
            return ToolOutput(output=f"Error: File not readable/writable: {path}", success=False)
        if "MIMO_EDIT:NOTFOUND" in output:
            return ToolOutput(
                output="Error: The exact string was not found in the file. Make sure the old_str matches exactly including whitespace and newlines.",
                success=False,
            )
        if "MIMO_EDIT:MULTIPLE:" in output:
            count = output.split("MIMO_EDIT:MULTIPLE:")[1].split()[0]
            return ToolOutput(
                output=f"Error: String found {count} times, must be unique. Please include more context to make the string unique.",
                success=False,
            )
        if "MIMO_EDIT:BINARY" in output:
            return ToolOutput(
                output=f"Error: {path} appears to be a binary file (contains NUL/control bytes); edit only supports text files.",
                success=False,
            )
        return ToolOutput(output=f"Error during replacement: {output.strip()[:2000]}", success=False)

    @staticmethod
    def _ship_strings(env, remote_dir: str, old_str: str, new_str: str) -> None:
        """Materialise old/new as files and copy them into the environment in
        one tar-stream transfer (a directory with two entries)."""
        with tempfile.TemporaryDirectory(prefix="mimo_edit_") as local_dir:
            oldp = os.path.join(local_dir, "old")
            newp = os.path.join(local_dir, "new")
            with open(oldp, "w", encoding="utf-8", newline="") as f:
                f.write(old_str)
            with open(newp, "w", encoding="utf-8", newline="") as f:
                f.write(new_str)
            try:
                env.copy_to(local_dir, remote_dir)
            except Exception as e:
                raise ToolException(f"Failed to transfer edit strings to environment: {e}")
