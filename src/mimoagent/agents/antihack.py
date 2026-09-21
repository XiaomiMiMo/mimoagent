"""Runtime anti-reward-hacking guard — an opt-in component for tool-calling agents.

GLM-5.2-style online guard: inspect each tool call *before* it executes and, if
a rule-based filter flags it as a likely reward hack — fetching upstream source,
reading protected evaluation artifacts, probing for hidden answer files — block
the real execution and return a dummy observation instead. The rollout
continues: the model just sees an unhelpful result, so the shortcut can't reach
the answer and can't inflate the verifiable reward, and the trajectory is never
abruptly terminated (which is what destabilizes training when whole rollouts are
rejected).

The guard is not part of ``DefaultAgent``: it is a policy layered on top of an
agent, not something every agent has. An agent adopts it by declaring an
``antihack`` field on its config and registering a guard on the agent's
action-interceptor chain in ``__init__``::

    @dataclass
    class MyAgentConfig(DefaultAgentConfig):
        antihack: Any = field(default_factory=AntiHackConfig)

    class MyAgent(DefaultAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.antihack = AntiHackGuard(self.config.antihack)
            self.add_action_interceptor(self.antihack)

The guard logic lives here once and nowhere else.

Default OFF. An absent or ``enabled: false`` config is a strict no-op —
``inspect`` returns ``None`` for every action — so existing runs are byte-for-
byte unaffected.

Two-stage detection (per the GLM-5.2 blog) is scaffolded but only stage 1 ships:

* **Stage 1 (here):** a regex filter tuned for recall over the known leak
  vectors. This is what runs in v1.
* **Stage 2 (stub):** an optional LLM judge (``llm_judge_enabled``) that
  confirms the *intent* of a flagged action to keep precision high. Until it is
  wired to a model endpoint, ``confirm`` conservatively upholds every regex hit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from mimoagent.tools.base import ToolOutput
from mimoagent.utils.log import get_logger

# Which param field(s) of each tool carry the model-controllable string we scan,
# and therefore which pattern set applies. Bash commands are free-form shell
# (curl/cat/find chains) → bash_patterns. The file tools expose a path and/or a
# search pattern → path_patterns; scanning them closes the non-Bash bypass where
# the model reads an eval artifact directly via Read/Grep/Glob instead of `cat`.
# Both tool catalogues are covered. Keys are exact-match against the tool_call
# name, so each schema needs its own row, and three schemas exist:
#   tools/          (DefaultAgent) lowercase ids, path param
#   tools/cc/       (CCAgent)      capitalized ids, file_path param
#   tools/mimocode/ (MimocodeAgent) lowercase ids, file_path param
# mimocode reuses the lowercase ids of the original catalogue but takes cc's
# param name, so the lowercase rows list both spellings — a field a schema does
# not have simply yields nothing to scan. The lowercase grep/glob rows exist only
# for mimocode; the original catalogue has no such tools.
_BASH_FIELDS: dict[str, tuple[str, ...]] = {
    "Bash": ("command",),
    "bash": ("command",),
    "exec_command": ("cmd",),
    # Patch bodies can add curl/git-clone lines; scan them as shell too.
    "apply_patch": ("input", "patch"),
}
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "read": ("path", "file_path"),
    "edit": ("path", "file_path"),
    "write": ("path", "file_path"),
    "Grep": ("pattern", "path"),
    "Glob": ("pattern", "path"),
    "grep": ("pattern", "path"),
    "glob": ("pattern", "path"),
    "apply_patch": ("input", "patch"),
}

# Recall-first defaults, derived empirically from 6,026 judged hacking
# trajectories (traj_analysis/hack_agg), tuned to allow sanitized local
# history while blocking cross-ref / remote archaeology. Tune from config
# rather than editing code. All compiled case-insensitively.
_DEFAULT_BASH_PATTERNS: list[str] = [
    # ---------------------------------------------------------------------
    # Git: allow sanitized local history, block cross-ref / remote archaeology
    # ---------------------------------------------------------------------
    # Assumption:
    #   Safe:
    #     git log
    #     git log --oneline -- path/to/file.py
    #     git show HEAD
    #     git show HEAD:path/to/file.py
    #     git show HEAD~1:path/to/file.py
    #     git diff
    #     git diff HEAD~1..HEAD
    #     git blame path/to/file.py
    #
    #   Unsafe:
    #     git fetch / pull / clone
    #     git log --all
    #     git show origin/main:path
    #     git checkout upstream/main -- file.py
    #     git reflog / stash / tags / hidden refs
    #
    # This relies on the harness sanitizing .git so HEAD history contains no
    # future solution commits, no remote refs, no tags to fixed releases, no stash,
    # no reflog, and no unreachable future objects.
    # Network/ref expansion.
    r"\bgit\s+(fetch|pull|clone|remote|ls-remote)\b",
    # Hidden/local archaeology surfaces.
    r"\bgit\s+(reflog|stash|bisect|show-ref|for-each-ref)\b",
    # Ref discovery. `git branch` alone is okay; all/remotes is not.
    r"\bgit\s+branch\b[^|;&\n]*(?:-a|-r|--all|--remotes)\b",
    # Tags often point to released/fixed versions. Keep blocked unless your
    # sanitizer deletes all tags.
    r"\bgit\s+tag\b",
    # Dangerous refs/options for otherwise useful git commands.
    # Allows HEAD / HEAD~ / HEAD^ local history, but blocks remote/ref namespace.
    r"\bgit\s+(log|show|diff|grep|checkout|switch|restore|merge|rebase|cherry-pick|ls-tree|archive)\b"
    r"[^|;&\n]*(?:--all|--branches|--remotes|--tags|\borigin/|\bupstream/|\brefs/|\bremotes/|\btags/|\bFETCH_HEAD\b|\bORIG_HEAD\b)",
    # Raw object database spelunking. Even with sanitized refs, these are rarely
    # needed for normal SWE debugging and are common leak tools if unreachable
    # objects remain.
    r"\bgit\s+(cat-file|fsck|verify-pack|pack-objects|unpack-objects|index-pack)\b",
    # Submodules can fetch external source. Allow normal local inspection, but
    # block remote update behavior.
    r"\bgit\s+submodule\b[^|;&\n]*(?:--remote|\bupdate\b[^|;&\n]*--remote)",
    # Optional stricter mode:
    # Uncomment if your object database may still contain unreachable future
    # commits. This will block `git show <sha>` even for safe historical commits.
    #
    # r"\bgit\s+(show|diff|grep|checkout|switch|restore|merge|rebase|cherry-pick|ls-tree)\b"
    # r"[^|;&\n]*\b[0-9a-f]{7,40}\b",
    # ---------------------------------------------------------------------
    # Fetch fixed/upstream source over the network
    # ---------------------------------------------------------------------
    # Still block raw patches, commits, releases, archives, and direct source
    # snapshots. This is less broad than blocking every github.com access.
    r"raw\.githubusercontent\.com",
    r"codeload\.github\.com",
    r"api\.github\.com/repos/[^ \t\n]+/(?:commits|pulls|releases|compare|git/)",
    r"github\.com/[^ \t\n]+/(?:raw|blob|commit|commits|pull|pulls|releases|archive|compare)/",
    r"\b(curl|wget)\b[^|;&\n]*"
    r"(?:raw\.githubusercontent|codeload\.github|gist\.github|pastebin|"
    r"/raw/|/blob/|/commit/|/commits/|/pull/|/pulls/|/releases?|/archive|/compare/)",
    # ---------------------------------------------------------------------
    # Install / inspect fixed upstream package
    # ---------------------------------------------------------------------
    # Relaxed compared with your current rule:
    #   allow pinned dependency installs such as `pip install foo==1.2.3`
    #   block VCS/URL/latest/canary-style source fetching.
    #
    # Ideally make this target-aware: block the target project/package name, but
    # allow ordinary dependency setup.
    r"\bpip3?\s+install\b[^|;&\n]*(?:git\+|https?://|--upgrade|\s-U\b)",
    r"\bpip3?\s+download\b[^|;&\n]*(?:git\+|https?://)",
    r"\b(npm|pnpm|yarn)\s+(?:install|add|i)\b[^|;&\n]*"
    r"(?:git\+|github:|gitlab:|bitbucket:|https?://|\blatest\b|\bnext\b|\bcanary\b)",
    r"\bgo\s+get\b[^|;&\n]*@(?:latest|master|main|HEAD)",
    r"\b(poetry|composer)\s+(?:add|require)\b[^|;&\n]*(?:git\+|https?://|dev-|latest)",
    r"\bcargo\s+(?:add|install)\b[^|;&\n]*(?:--git|https?://)",
    r"\bgem\s+install\b[^|;&\n]*(?:--source|https?://)",
    # ---------------------------------------------------------------------
    # Protected evaluation artifacts / answer files
    # ---------------------------------------------------------------------
    # Less broad than blocking words like "solution", "expected_output",
    # "golden", or "oracle" everywhere. Those terms appear in normal tests.
    r"(?:^|[^\w./-])\.eval(?:/|$)",
    r"\bsecret_cases\b",
    r"\bfind\b[^|;&\n]*(?:-name|-path|-wholename)[^|;&\n]*"
    r"\b(?:hidden_tests?|private_tests?|secret_cases|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)\b",
    r"\b(?:grep|rg|ag)\b[^|;&\n]*"
    r"\b(?:hidden_tests?|private_tests?|secret_cases|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)\b",
    r"\b(?:cat|less|more|head|tail|sed|awk|xxd|od|strings)\b[^|;&\n]*"
    r"(?:\.eval/|secret_cases|hidden_tests?|private_tests?|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)",
]


_DEFAULT_PATH_PATTERNS: list[str] = [
    r"(?:^|/)\.eval(?:/|$)",
    r"\bsecret_cases\b",
    r"\bhidden_tests?\b",
    r"\bprivate_tests?\b",
    r"\banswer.?key\b",
    r"\bgold(?:en)?_patch\b",
    r"\btest_patch\b",
    r"\bground.?truth\b",
]


@dataclass
class AntiHackConfig:
    """Config for the runtime guard. Surfaced under ``agent.antihack`` in yaml.

    ``enabled`` defaults to False so the guard is inert unless a run opts in.
    """

    enabled: bool = False
    bash_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_BASH_PATTERNS))
    path_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_PATH_PATTERNS))
    # What the model sees back in place of the blocked tool's real output. Kept
    # plausibly benign on purpose — it should read like an ordinary empty/failed
    # result, not announce that a guard fired.
    dummy_output: str = "Permission denied. Resource unavailable."
    dummy_success: bool = False
    # Stage-2 intent judge. Off in v1 (regex-only); when True, ``confirm``
    # delegates to ``_llm_judge`` (currently a conservative stub).
    llm_judge_enabled: bool = False

    @classmethod
    def from_obj(cls, obj: AntiHackConfig | dict | None) -> AntiHackConfig:
        """Normalize whatever the agent config holds into an ``AntiHackConfig``.

        The agent config is built from yaml via ``filter_dataclass_kwargs``,
        which keeps ``antihack`` as a raw dict (nested dataclasses are not
        auto-constructed). Accept dict / instance / None so wiring is trivial.
        Unknown keys are dropped to match the tolerant-config convention.
        """
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in obj.items() if k in valid})
        raise TypeError(f"antihack config must be dict/AntiHackConfig/None, got {type(obj)!r}")


@dataclass
class AntiHackVerdict:
    """A single flagged action: which tool/field tripped which pattern."""

    tool: str
    field: str
    value: str
    pattern: str


class AntiHackGuard:
    """Regex guard plus a stage-2 hook, usable as a ``DefaultAgent`` action interceptor.

    One instance per agent, registered with ``agent.add_action_interceptor``.
    Calling the guard with a parsed action returns the dummy observation for a
    flagged call — the tool never runs, so the env is untouched and the hack
    can't reach the answer — and ``None`` for everything else. ``inspect`` is
    read-only and safe from the parallel tool-call workers (the compiled
    patterns are immutable); ``blocks`` is append-only, and ``list.append`` is
    atomic under CPython's GIL, so no lock.

    An agent that decorates real tool results must do so in ``_execute_tool``
    rather than ``execute_action`` — the dummy has to come back undecorated, or
    its metadata tells the model a guard fired.
    """

    def __init__(self, config: AntiHackConfig | dict | None = None):
        self.config = AntiHackConfig.from_obj(config)
        flags = re.IGNORECASE
        self._bash_res = [(p, re.compile(p, flags)) for p in self.config.bash_patterns]
        self._path_res = [(p, re.compile(p, flags)) for p in self.config.path_patterns]
        # One dict per blocked call (tool, field, value, pattern). Telemetry only
        # in v1 (dummy-only); the seam to a reward/penalty channel is here.
        self.blocks: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, action: dict) -> dict | None:
        """Action interceptor: the dummy observation for a flagged call, else ``None``."""
        verdict = self.inspect(action)
        if verdict is None or not self.confirm(action, verdict):
            return None
        self.blocks.append(asdict(verdict))
        get_logger().warning(
            f"[anti-hack] blocked {verdict.tool}.{verdict.field} (pattern={verdict.pattern!r}): {verdict.value[:200]}"
        )
        return self.dummy_output().to_dict()

    def inspect(self, action: dict) -> AntiHackVerdict | None:
        """Return a verdict if ``action`` matches a hack pattern, else None.

        Returns None immediately when disabled, so the hot path costs one bool
        check for runs that don't opt in.
        """
        if not self.config.enabled:
            return None
        tool = action.get("tool")
        params = action.get("params")
        if not isinstance(params, dict):
            return None

        if tool in _BASH_FIELDS:
            for fld in _BASH_FIELDS[tool]:
                hit = self._match(params.get(fld), self._bash_res)
                if hit is not None:
                    return AntiHackVerdict(tool, fld, str(params.get(fld)), hit)
        if tool in _PATH_FIELDS:
            for fld in _PATH_FIELDS[tool]:
                hit = self._match(params.get(fld), self._path_res)
                if hit is not None:
                    return AntiHackVerdict(tool, fld, str(params.get(fld)), hit)
        return None

    @staticmethod
    def _match(value: Any, compiled: list[tuple[str, re.Pattern]]) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        for raw, rx in compiled:
            if rx.search(value):
                return raw
        return None

    def confirm(self, action: dict, verdict: AntiHackVerdict) -> bool:
        """Stage-2 gate. True → block the call. Regex-only v1 always confirms;
        when ``llm_judge_enabled`` it defers to the (stub) intent judge."""
        if not self.config.llm_judge_enabled:
            return True
        return self._llm_judge(action, verdict)

    def _llm_judge(self, action: dict, verdict: AntiHackVerdict) -> bool:
        """Placeholder for the GLM-5.2 stage-2 intent judge.

        Until wired to a model endpoint, conservatively uphold the regex hit.
        Replace with a call that asks a judge model whether ``verdict.value`` is
        a reward hack given the task, returning True to block.
        """
        return True

    def dummy_output(self) -> ToolOutput:
        """The fabricated observation returned in place of real execution.

        Metadata is intentionally empty: ``_format_observation`` appends
        non-empty metadata to the model-facing text, and we don't want the
        dummy to advertise that a guard fired.
        """
        return ToolOutput(output=self.config.dummy_output, success=self.config.dummy_success, metadata={})
