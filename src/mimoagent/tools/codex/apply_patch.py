"""Codex ``apply_patch``: V4A freeform patch (``*** Begin Patch``).

The tool is freeform in Codex (raw string, not JSON). mimoagent's function-
calling path still wraps a JSON object; we accept ``input`` / ``patch``, a JSON
string of the patch, or the raw V4A text (see ``parse_apply_patch_args``).

The environment probes below must not assume they own stdout. ``env.execute``
runs a login shell (``bash -lc``), so ``/etc/profile`` and ``~/.bash_profile``
banners are prepended to every probe's output; reading the whole stdout made
apply_patch fail on every image that prints one. Each probe therefore extracts
only its own machine-readable part.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import shlex
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mimoagent.environments import TransportError
from mimoagent.tools.base import BaseTool, ToolException, ToolOutput

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File:"
_DELETE = "*** Delete File:"
_UPDATE = "*** Update File:"
_MOVE = "*** Move to:"

# The state probe prints exactly one of these; a banner may precede it.
_STATE_MARKER_RE = re.compile(r"MIMO_(FILE|OTHER|MISSING)")


def _last_line(output: str) -> str:
    """The last non-empty line, which is where a probe's own output lands."""

    for line in reversed((output or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


class ApplyPatchError(Exception):
    """A well-formed call that could not be applied (missing file, hunk miss)."""


@dataclass
class _Hunk:
    locators: list[str] = field(default_factory=list)
    lines: list[tuple[str, str]] = field(default_factory=list)  # (tag, text) tag in + - space
    is_end_of_file: bool = False


@dataclass
class _Add:
    path: str
    content: str


@dataclass
class _Delete:
    path: str


@dataclass
class _Update:
    path: str
    move_to: str | None
    hunks: list[_Hunk]


_Op = _Add | _Delete | _Update


@dataclass
class _PreparedPatch:
    writes: list[tuple[str, str]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    originals: dict[str, str] = field(default_factory=dict)
    original_modes: dict[str, int] = field(default_factory=dict)
    write_modes: dict[str, int] = field(default_factory=dict)
    new_paths: set[str] = field(default_factory=set)
    reports: list[str] = field(default_factory=list)


@dataclass
class _VirtualPath:
    state: str
    initial_state: str
    content: str | None = None
    mode: int | None = None
    mutated: bool = False


def parse_apply_patch_args(raw: Any) -> dict[str, Any]:
    """Normalize function-call arguments into ``{"input": <patch text>}``.

    Codex emits a freeform string; function-calling gateways may JSON-encode
    it, wrap it as ``{"input": ...}`` / ``{"patch": ...}``, or pass a dict.
    """
    if isinstance(raw, dict):
        text = raw.get("input")
        if text is None:
            text = raw.get("patch")
        if text is None:
            return raw if raw else {"input": ""}
        return {**raw, "input": text}
    if raw is None:
        return {"input": ""}
    if not isinstance(raw, str):
        return {"input": str(raw)}
    s = raw.strip()
    if not s:
        return {"input": ""}
    if s.startswith("{") or s.startswith('"'):
        import json

        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return {"input": raw}
        if isinstance(parsed, dict):
            return parse_apply_patch_args(parsed)
        if isinstance(parsed, str):
            return {"input": parsed}
    return {"input": raw}


def parse_v4a(text: str) -> list[_Op]:
    """Parse a V4A patch into add/delete/update ops. Raises ApplyPatchError."""
    if not isinstance(text, str) or not text.strip():
        raise ApplyPatchError("empty patch")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # The grammar permits one final LF after End Patch, but nothing else
    # outside the boundary markers.
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    lines = normalized.split("\n")
    if not lines or lines[0] != _BEGIN:
        raise ApplyPatchError(f"the first line must be '{_BEGIN}'")
    if lines[-1] != _END:
        raise ApplyPatchError(f"the last line must be '{_END}'")

    ops: list[_Op] = []
    i = 1
    end = len(lines) - 1

    def operation_header(line: str) -> bool:
        return line.startswith((_ADD + " ", _DELETE + " ", _UPDATE + " "))

    while i < end:
        line = lines[i]
        if line.startswith(_ADD + " "):
            path = line[len(_ADD) :].strip()
            if not path:
                raise ApplyPatchError(f"empty path at line {i + 1}")
            i += 1
            buf: list[str] = []
            while i < end and not operation_header(lines[i]):
                raw_line = lines[i]
                if not raw_line.startswith("+"):
                    raise ApplyPatchError(f"Add File line {i + 1} for {path!r} must start with '+'")
                buf.append(raw_line[1:])
                i += 1
            if not buf:
                raise ApplyPatchError(f"Add File operation for {path!r} has no content lines")
            ops.append(_Add(path=path, content="\n".join(buf) + "\n"))
            continue
        if line.startswith(_DELETE + " "):
            path = line[len(_DELETE) :].strip()
            if not path:
                raise ApplyPatchError(f"empty path at line {i + 1}")
            ops.append(_Delete(path=path))
            i += 1
            if i < end and not operation_header(lines[i]):
                raise ApplyPatchError(f"Delete File operation for {path!r} must not have a body")
            continue
        if line.startswith(_UPDATE + " "):
            path = line[len(_UPDATE) :].strip()
            if not path:
                raise ApplyPatchError(f"empty path at line {i + 1}")
            i += 1
            move_to = None
            if i < end and lines[i].startswith(_MOVE + " "):
                move_to = lines[i][len(_MOVE) :].strip()
                if not move_to:
                    raise ApplyPatchError(f"empty Move to path at line {i + 1}")
                i += 1
            hunks: list[_Hunk] = []
            current = _Hunk()

            def flush_hunk() -> None:
                nonlocal current
                if current.lines or current.is_end_of_file:
                    hunks.append(current)
                elif current.locators:
                    raise ApplyPatchError(f"locator for {path!r} is not followed by change lines")
                current = _Hunk()

            while i < end and not operation_header(lines[i]):
                change_line = lines[i]
                if change_line == "@@" or change_line.startswith("@@ "):
                    locator = change_line[3:] if change_line.startswith("@@ ") else ""
                    if current.lines or current.is_end_of_file:
                        flush_hunk()
                    if locator:
                        current.locators.append(locator)
                    else:
                        current.locators.clear()
                    i += 1
                    continue
                if change_line == "*** End of File":
                    if current.is_end_of_file:
                        raise ApplyPatchError(f"duplicate End of File marker for {path!r}")
                    current.is_end_of_file = True
                elif current.is_end_of_file:
                    # code-rs tolerates blank separator lines after this marker.
                    if change_line:
                        raise ApplyPatchError(f"End of File marker for {path!r} must be the final hunk line")
                elif change_line.startswith(("+", "-", " ")):
                    current.lines.append((change_line[0], change_line[1:]))
                elif change_line == "":
                    # Models commonly emit a bare blank line for empty context.
                    current.lines.append((" ", ""))
                else:
                    raise ApplyPatchError(f"invalid Update File line {i + 1} for {path!r}: {change_line!r}")
                i += 1
            flush_hunk()
            if not hunks and move_to is None:
                raise ApplyPatchError(f"Update File operation for {path!r} is empty")
            if hunks and not any(tag in ("+", "-") for hunk in hunks for tag, _ in hunk.lines):
                raise ApplyPatchError(f"Update File operation for {path!r} makes no changes")
            ops.append(_Update(path=path, move_to=move_to, hunks=hunks))
            continue

        raise ApplyPatchError(f"invalid operation header at line {i + 1}: {line!r}")

    if not ops:
        raise ApplyPatchError("no file operations in patch")
    return ops


def apply_hunks_to_text(original: str, hunks: list[_Hunk], path: str = "the file") -> str:
    """Apply V4A hunks to ``original``. Raises ApplyPatchError on a miss.

    ``path`` only shapes the error text, and it has to: a miss inside a
    multi-file patch is otherwise indistinguishable from a miss in any other
    file. The wording follows upstream's ``compute_replacements`` errors.
    """
    original = original.replace("\r\n", "\n").replace("\r", "\n")
    ended_nl = original.endswith("\n") if original else True
    lines = original.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    cursor = 0
    for hunk in hunks:
        search_start = cursor
        for locator in hunk.locators:
            locator_index = _find_line(lines, locator, search_start)
            if locator_index is None:
                raise ApplyPatchError(f"Failed to find context {locator!r} in {path}")
            search_start = locator_index + 1

        old_seq = [t for tag, t in hunk.lines if tag in (" ", "-")]
        new_seq = [t for tag, t in hunk.lines if tag in (" ", "+")]
        if not old_seq and not new_seq:
            continue

        if not old_seq:
            # code-rs treats a context-free addition as an EOF insertion even
            # when the optional End of File marker is omitted.
            idx = len(lines)
            matched_old = old_seq
            replacement = new_seq
        else:
            matched_old = old_seq
            replacement = new_seq
            matches = _find_subseq_matches(
                lines,
                matched_old,
                start=search_start,
                eof=hunk.is_end_of_file,
            )
            # A final empty context line represents the file's terminating LF,
            # not a separate source line. code-rs retries without that sentinel.
            if not matches and matched_old[-1] == "":
                matched_old = matched_old[:-1]
                if replacement and replacement[-1] == "":
                    replacement = replacement[:-1]
                matches = _find_subseq_matches(
                    lines,
                    matched_old,
                    start=search_start,
                    eof=hunk.is_end_of_file,
                )
            if not matches:
                raise ApplyPatchError("Failed to find expected lines in {}:\n{}".format(path, "\n".join(old_seq)))
            # seek_sequence resolves repeated context to the first match at or
            # after the current chunk cursor. For an EOF hunk the search was
            # already pinned to the final window, so the first match is it.
            idx = matches[0]
        lines = lines[:idx] + replacement + lines[idx + len(matched_old) :]
        cursor = idx + len(replacement)

    result = "\n".join(lines)
    if ended_nl:
        if result and not result.endswith("\n"):
            result += "\n"
        elif not result:
            result = ""
    return result


def _find_line(lines: list[str], expected: str, start: int) -> int | None:
    for normalizer in (lambda value: value, str.rstrip, str.strip):
        wanted = normalizer(expected)
        for index in range(start, len(lines)):
            if normalizer(lines[index]) == wanted:
                return index
    return None


def _find_subseq_matches(
    haystack: list[str],
    needle: list[str],
    *,
    start: int,
    eof: bool,
) -> list[int]:
    """Locate ``needle`` in ``haystack`` at or after ``start``, code-rs style.

    Matching relaxes in three passes (exact, rstrip, strip), returning the first
    pass that yields any position. When ``eof`` is set the hunk carried
    ``*** End of File``, so — like code-rs ``seek_sequence`` under NormalizeToLf —
    the search is pinned to the final window (``last_start``): an EOF hunk that
    does not match at the end of the file is *not* silently relocated earlier.
    """

    if not needle or len(needle) > len(haystack):
        return []
    last_start = len(haystack) - len(needle)
    search_start = last_start if eof else min(max(0, start), last_start + 1)

    for normalizer in (lambda value: value, str.rstrip, str.strip):
        wanted = [normalizer(value) for value in needle]
        matches = [
            index
            for index in range(search_start, last_start + 1)
            if [normalizer(value) for value in haystack[index : index + len(needle)]] == wanted
        ]
        if matches:
            return matches
    return []


class ApplyPatchTool(BaseTool):
    """Apply a Codex V4A patch to files in the environment."""

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return """Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON. Pass the patch document as `input`.

Do not create or edit files with `cat` or other shell write tricks. Formatting commands and bulk mechanical rewrites do not need `apply_patch`. Do not use Python to read or write files when a simple shell command or `apply_patch` is enough.

Patch document:

*** Begin Patch
*** [ACTION] File: [path]
[hunks or file content]
*** End Patch

Actions:
- Add File: create a new file. Body lines are prefixed with `+`.
- Delete File: remove an existing file.
- Update File: change an existing file. Optional `*** Move to: [newpath]` on the next line. Hunks start with `@@` (optional locator) then lines prefixed with ` ` (context), `-` (remove), or `+` (add). Context plus removals are matched from the prior hunk onward. Use `*** End of File` after a contextual hunk to anchor it at EOF; a context-free addition is appended at EOF with or without that marker.

You may include several file operations in one patch. Use absolute paths, or paths relative to the working directory stated in your instructions."""

    def get_function_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The apply_patch payload (`*** Begin Patch` ... `*** End Patch`). Do not wrap the patch in extra JSON.",
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        }

    def execute(self, params: Any, context: dict[str, Any] | None = None) -> ToolOutput:
        env = (context or {}).get("env")
        if not env:
            raise ToolException("No environment provided for apply_patch")
        cwd = getattr(getattr(env, "config", None), "cwd", "") or ""

        if isinstance(params, str):
            text = params
        elif isinstance(params, dict):
            text = params.get("input") or params.get("patch") or ""
        else:
            raise ToolException(f"apply_patch params must be a dict or string, got {type(params).__name__}")
        if not str(text).strip():
            raise ToolException("apply_patch requires a non-empty 'input' patch")

        try:
            ops = parse_v4a(str(text))
        except ApplyPatchError as e:
            return ToolOutput(output=f"Error: {e}", success=False)

        try:
            prepared = self._prepare(env, cwd, ops)
            self._commit(env, prepared)
        except ApplyPatchError as e:
            return ToolOutput(output=f"Error: {e}", success=False)
        except TransportError:
            raise
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to apply patch: {e}") from e
        # Upstream leads the report with this header (apply-patch/src/lib.rs).
        # Without it the model sees a bare "M path" and cannot tell a report from
        # an echo of its own patch.
        report = "\n".join(["Success. Updated the following files:", *prepared.reports])
        return ToolOutput(output=report if prepared.reports else "Patch applied.", success=True)

    def _prepare(self, env: Any, cwd: str, ops: list[_Op]) -> _PreparedPatch:
        """Validate and calculate every operation before touching the filesystem."""
        prepared = _PreparedPatch()
        virtual_paths: dict[str, _VirtualPath] = {}

        def load(path: str) -> _VirtualPath:
            current = virtual_paths.get(path)
            if current is not None:
                return current
            state = self._path_state(env, path)
            if state == "file":
                content = self._read_file(env, path)
                mode = self._file_mode(env, path)
                prepared.originals[path] = content
                prepared.original_modes[path] = mode
            else:
                content = None
                mode = None
                if state == "missing":
                    prepared.new_paths.add(path)
            current = _VirtualPath(state=state, initial_state=state, content=content, mode=mode)
            virtual_paths[path] = current
            return current

        for op in ops:
            path = _resolve_path(op.path, cwd)
            current = load(path)

            if isinstance(op, _Add):
                if current.state == "other":
                    raise ApplyPatchError(f"Add File target is not a regular file: {op.path}")
                current.state = "file"
                current.content = op.content
                current.mutated = True
                prepared.reports.append(f"A {path}")
                continue

            if current.state != "file":
                operation = "Delete File" if isinstance(op, _Delete) else "Update File"
                raise ApplyPatchError(f"{operation} target is not a regular file: {op.path}")

            if isinstance(op, _Delete):
                current.state = "missing"
                current.content = None
                current.mode = None
                current.mutated = True
                prepared.reports.append(f"D {path}")
                continue

            assert current.content is not None
            updated = apply_hunks_to_text(current.content, op.hunks, path)
            destination = _resolve_path(op.move_to, cwd) if op.move_to else path
            if destination != path:
                target = load(destination)
                if target.state == "other":
                    raise ApplyPatchError(f"Move target is not a regular file: {op.move_to}")
                target.state = "file"
                target.content = updated
                # Preserve the overwritten destination's mode. For a new
                # destination, carrying the source mode keeps moved executable
                # files executable.
                target.mode = target.mode if target.mode is not None else current.mode
                target.mutated = True
                current.state = "missing"
                current.content = None
                current.mode = None
                current.mutated = True
                prepared.reports.append(f"M {path} -> {destination}")
            else:
                current.content = updated
                current.mutated = True
                prepared.reports.append(f"M {path}")

        # Commit only the final state of each path. This retains full
        # prevalidation while matching code-rs's sequential in-patch semantics.
        for path, current in virtual_paths.items():
            if not current.mutated:
                continue
            if current.state == "file":
                assert current.content is not None
                prepared.writes.append((path, current.content))
                if current.mode is not None:
                    prepared.write_modes[path] = current.mode
            elif current.initial_state == "file":
                prepared.deletes.append(path)

        return prepared

    def _commit(self, env: Any, prepared: _PreparedPatch) -> None:
        """Stage all writes, then replace targets; best-effort rollback on failure."""
        token = uuid.uuid4().hex
        staged: list[tuple[str, str]] = []
        try:
            for index, (destination, content) in enumerate(prepared.writes):
                staged_path = f"{destination}.mimo-apply-patch-{token}-{index}"
                if self._path_state(env, staged_path) != "missing":
                    raise ApplyPatchError(f"temporary patch path already exists: {staged_path}")
                staged.append((staged_path, destination))
                self._write_file(env, staged_path, content)
        except Exception:
            self._cleanup_staged(env, staged)
            raise

        try:
            for staged_path, destination in staged:
                self._run_checked(
                    env,
                    f"mv -f -- {shlex.quote(staged_path)} {shlex.quote(destination)}",
                    f"replace {destination}",
                )
                mode = prepared.write_modes.get(destination)
                if mode is not None:
                    self._restore_mode(env, destination, mode)
            for path in prepared.deletes:
                self._run_checked(env, f"rm -f -- {shlex.quote(path)}", f"delete {path}")
        except Exception as commit_error:
            rollback_errors = self._rollback(env, prepared)
            detail = f"; rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
            if isinstance(commit_error, TransportError):
                raise TransportError(f"{commit_error}{detail}") from commit_error
            if isinstance(commit_error, ToolException):
                raise ToolException(f"{commit_error}{detail}") from commit_error
            if isinstance(commit_error, ApplyPatchError):
                raise ApplyPatchError(f"{commit_error}{detail}") from commit_error
            raise ToolException(f"patch commit failed: {commit_error}{detail}") from commit_error
        finally:
            self._cleanup_staged(env, staged)

    def _rollback(self, env: Any, prepared: _PreparedPatch) -> list[str]:
        errors: list[str] = []
        for path, content in prepared.originals.items():
            try:
                self._write_file(env, path, content)
                self._restore_mode(env, path, prepared.original_modes[path])
            except Exception as error:
                errors.append(f"restore {path}: {error}")
        for path in prepared.new_paths:
            try:
                self._run_checked(env, f"rm -f -- {shlex.quote(path)}", f"remove {path}")
            except Exception as error:
                errors.append(f"remove {path}: {error}")
        return errors

    def _cleanup_staged(self, env: Any, staged: list[tuple[str, str]]) -> None:
        for staged_path, _ in staged:
            try:
                env.execute(f"rm -f -- {shlex.quote(staged_path)}")
            except Exception:
                pass

    @staticmethod
    def _run_checked(env: Any, command: str, operation: str) -> dict[str, Any]:
        result = env.execute(command)
        if result.get("reason") == "transport_error":
            raise TransportError(f"Transport error while attempting to {operation}")
        if result.get("returncode") != 0:
            output = (result.get("output") or "").strip()
            suffix = f": {output[:500]}" if output else ""
            raise ApplyPatchError(f"failed to {operation}{suffix}")
        return result

    @classmethod
    def _path_state(cls, env: Any, path: str) -> str:
        quoted = shlex.quote(path)
        result = cls._run_checked(
            env,
            f"if [ -f {quoted} ]; then printf MIMO_FILE; "
            f"elif [ -e {quoted} ]; then printf MIMO_OTHER; else printf MIMO_MISSING; fi",
            f"inspect {path}",
        )
        output = result.get("output") or ""
        # The last match is ours: a banner cannot end with one of these markers
        # unless it deliberately prints it, and our printf is always last.
        matches = _STATE_MARKER_RE.findall(output)
        states = {"FILE": "file", "OTHER": "other", "MISSING": "missing"}
        if not matches:
            raise ApplyPatchError(f"could not determine file state for {path}: {output.strip()!r}")
        return states[matches[-1]]

    @classmethod
    def _file_mode(cls, env: Any, path: str) -> int:
        result = cls._run_checked(
            env,
            f"stat -Lc %a -- {shlex.quote(path)}",
            f"inspect mode for {path}",
        )
        value = _last_line(result.get("output") or "")
        try:
            mode = int(value, 8)
        except ValueError as error:
            raise ApplyPatchError(f"could not determine file mode for {path}: {value!r}") from error
        if not 0 <= mode <= 0o7777:
            raise ApplyPatchError(f"invalid file mode for {path}: {value!r}")
        return mode

    @classmethod
    def _restore_mode(cls, env: Any, path: str, mode: int) -> None:
        cls._run_checked(
            env,
            f"chmod -- {mode:o} {shlex.quote(path)}",
            f"restore mode for {path}",
        )

    @classmethod
    def _read_file(cls, env: Any, path: str) -> str:
        q = shlex.quote(path)
        if cls._path_state(env, path) != "file":
            raise ApplyPatchError(f"Not a regular file: {path}")
        raw = cls._read_bytes(env, path, q)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            # A lossy decode here would let apply_patch rewrite a non-UTF-8 file
            # with U+FFFD in place of bytes the patch never touched, and report
            # success. Fail cleanly instead — the episode continues, the file is
            # untouched.
            raise ApplyPatchError(f"cannot patch non-UTF-8 file: {path} ({error})") from error

    @classmethod
    def _read_bytes(cls, env: Any, path: str, quoted: str) -> bytes:
        """Return the file's exact bytes, distinguishing transport from decode.

        ``copy_out`` moves bytes verbatim; only a real transport failure raises.
        Without ``copy_out`` we base64 the file over the exec channel rather than
        ``cat``-ing it, because ``env.execute`` decodes stdout with
        ``errors="replace"`` and would corrupt non-UTF-8 content before we could
        see it.
        """

        if hasattr(env, "copy_out"):
            local = None
            try:
                fd, local = tempfile.mkstemp(prefix="mimo_ap_")
                os.close(fd)
                env.copy_out(path, local)
                return Path(local).read_bytes()
            finally:
                if local:
                    try:
                        os.unlink(local)
                    except OSError:
                        pass
        result = cls._run_checked(env, f"base64 {quoted} | tr -d '\\n'", f"read {path}")
        # `tr -d '\n'` collapses the payload onto one line, so it is the last
        # line even when a login-shell banner precedes it.
        encoded = _last_line(result.get("output") or "")
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ApplyPatchError(f"could not read {path}: {error}") from error

    @classmethod
    def _write_file(cls, env: Any, path: str, content: str) -> None:
        parent = str(Path(path).parent)
        if parent and parent != ".":
            cls._run_checked(env, f"mkdir -p -- {shlex.quote(parent)}", f"create directory {parent}")
        fd, local = tempfile.mkstemp(prefix="mimo_apw_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            env.copy_to(local, path)
        except TransportError:
            raise
        except Exception as e:
            raise ToolException(f"Failed to write {path}: {e}") from e
        finally:
            try:
                os.unlink(local)
            except OSError:
                pass


def _resolve_path(path: str, cwd: str) -> str:
    path = (path or "").strip()
    if not path:
        raise ApplyPatchError("empty path in patch")
    if any(character in path for character in ("\x00", "\n", "\r")):
        raise ApplyPatchError("patch paths must not contain NUL or newline characters")
    if path.startswith("/"):
        return os.path.normpath(path)
    base = cwd or os.getcwd()
    return os.path.normpath(str(Path(base) / path))
