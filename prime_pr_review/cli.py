"""prime-review — one console entry point for the whole toolkit.

Installed via [project.scripts]; wraps the operational scripts so the agent is
usable from any working directory:

    prime-review pr owner/my-service 2567         review exactly one PR
    prime-review sweep --repo my-service          sweep a lane
    prime-review replay --repo my-service --state open --count 10
    prime-review score                            check the demo answer key
    prime-review check                            preflight (config/secrets/gh)
    prime-review cochange --repo <path> --out <file>

The scripts live in scripts/ (not inside the package) because tests import them
by that path; this module loads them from the agent root at call time. Every
cwd-relative default (--config, --reviews-dir) is pinned to the agent root when
the caller does not pass it, so the commands behave identically from anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = AGENT_ROOT / "scripts"

_SCRIPT_FOR = {
    "sweep": "run_sweep",
    "replay": "replay_corpus",
    "score": "score_demo",
    "cochange": "build_cochange",
    "eval": "eval_swecare",
}

_USAGE = """prime-review <command> [options]

commands:
  pr <repo> <number>   review exactly one PR (sugar for: sweep --repo R --pr N)
  sweep                run a lane sweep            (scripts/run_sweep.py)
  replay               replay a PR corpus, report  (scripts/replay_corpus.py)
  score                score the demo answer key   (scripts/score_demo.py)
  check                preflight config/secrets/gh (python -m prime_pr_review)
  cochange             mine a co-change graph      (scripts/build_cochange.py)
  eval run|score|report   SWE-CARE ablation harness (docs/superpowers/specs/2026-09-11-swecare-ablation-harness-design.md)

Every command accepts its script's own flags; --config and --reviews-dir
default to the agent's own files regardless of the current directory.
"""


def _load(script: str):
    path = SCRIPTS_DIR / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"scripts.{script}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: stdlib dataclasses resolves cls.__module__ through
    # sys.modules at class-creation time, and an unregistered module crashes any
    # script that defines a module-level @dataclass (e.g. build_cochange).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _with_default(rest: list[str], flag: str, value: str) -> list[str]:
    return rest if flag in rest else [*rest, flag, value]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_USAGE)
        return 0

    command, rest = args[0], args[1:]

    if command == "pr":
        if len(rest) < 2:
            print("usage: prime-review pr <repo> <number> [extra flags]", file=sys.stderr)
            return 2
        command = "sweep"
        rest = ["--repo", rest[0], "--pr", rest[1], *rest[2:]]

    if command == "check":
        from prime_pr_review.__main__ import main as check_main

        rest = _with_default(rest, "--config", str(AGENT_ROOT / "config.toml"))
        return check_main(["check", *rest])

    script = _SCRIPT_FOR.get(command)
    if script is None:
        print(f"unknown command {command!r}\n\n{_USAGE}", file=sys.stderr)
        return 2

    if command in {"sweep", "replay"}:
        rest = _with_default(rest, "--config", str(AGENT_ROOT / "config.toml"))
    if command == "score":
        rest = _with_default(rest, "--reviews-dir", str(AGENT_ROOT / "reviews"))

    module = _load(script)
    sys.argv = [f"prime-review {command}", *rest]
    return int(module.main() or 0)


if __name__ == "__main__":
    sys.exit(main())
