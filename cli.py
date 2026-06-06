"""Task Boundary Detector — CLI.

Usage:
  python cli.py                          # latest Claude Code session, one-off check
  python cli.py --session <path>         # specific session file
  python cli.py --source codex           # use Codex CLI history instead
  python cli.py --source gemini          # use Gemini CLI sessions
  python cli.py --watch                  # daemon mode (re-check on file change)
  python cli.py --json                   # machine-readable output
  python cli.py --lookback 10            # how many recent user prompts to analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import (
    DEFAULT_CLAUDE_PROJECTS,
    classify_boundary,
    context_value_score,
    extract_user_prompts,
    find_latest_session,
    read_session,
)
from adapters import ADAPTERS, get_prompts, normalize_source


COLOR_MAP = {
    "new_task":       ("\033[33m", "⚡"),   # yellow
    "task_complete":  ("\033[36m", "⚠️"),   # cyan
    "continuation":   ("\033[32m", "✅"),   # green
    "unclear":        ("\033[37m", "•"),    # grey
}
RESET = "\033[0m"


def analyze(session_path: Path, lookback: int, source: str = "claude_code") -> dict:
    if source == "claude_code" and session_path and session_path.suffix == ".jsonl":
        events = read_session(session_path)
        prompts = extract_user_prompts(events)
    else:
        prompts = get_prompts(source, path=session_path)
    boundary = classify_boundary(prompts, lookback=lookback)
    cvs = context_value_score(prompts)
    return {
        "session": str(session_path),
        "n_user_prompts": len(prompts),
        "context_tokens_estimate": sum(p.tokens for p in prompts),
        "context_value_score": round(cvs, 3),
        "boundary": {
            "decision": boundary.decision,
            "confidence": round(boundary.confidence, 3),
            "reasoning": boundary.reasoning,
            "recommendation": boundary.recommendation,
            "scores": {
                "new_task": boundary.new_task_hits,
                "task_complete": boundary.complete_hits,
                "continuation": boundary.continuation_hits,
            },
        },
    }


def render(result: dict, color: bool = True) -> str:
    b = result["boundary"]
    decision = b["decision"]
    col, icon = COLOR_MAP.get(decision, ("", "?"))
    if not color:
        col, RESET_ = "", ""
    else:
        RESET_ = RESET

    lines = [
        f"{col}{icon} {decision.upper()}{RESET_}  (confidence: {b['confidence']:.0%})",
        f"   Sesja: {Path(result['session']).name}",
        f"   Historia: {result['n_user_prompts']} user prompts, ~{result['context_tokens_estimate']:,} tokens",
        f"   Context value: {result['context_value_score']:.0%}",
        f"   Sygnały: new={b['scores']['new_task']} complete={b['scores']['task_complete']} cont={b['scores']['continuation']}",
        f"   {b['recommendation']}",
    ]
    return "\n".join(lines)


def watch(session_path: Path, lookback: int, json_out: bool, color: bool,
          source: str = "claude_code") -> None:
    last_mtime = 0
    last_decision = None
    print(f"Watching: {session_path} (source={source})", file=sys.stderr)
    while True:
        try:
            mtime = session_path.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                result = analyze(session_path, lookback, source=source)
                decision = result["boundary"]["decision"]
                # Only print if decision changed OR significant
                if decision != last_decision or decision in ("new_task", "task_complete"):
                    last_decision = decision
                    if json_out:
                        print(json.dumps(result, ensure_ascii=False))
                    else:
                        print("\n" + render(result, color=color))
                    sys.stdout.flush()
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nStopped.", file=sys.stderr)
            return
        except FileNotFoundError:
            print(f"Session vanished: {session_path}", file=sys.stderr)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude Code Task Boundary Detector")
    parser.add_argument("--session", type=Path, default=None,
                        help="Session JSONL path (default: latest from ~/.claude/projects/)")
    parser.add_argument("--watch", action="store_true",
                        help="Daemon mode — keep watching for changes")
    parser.add_argument("--json", action="store_true",
                        help="JSON output (machine-readable)")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Number of recent user prompts to analyze (default 5)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors")
    parser.add_argument("--projects-dir", type=Path, default=DEFAULT_CLAUDE_PROJECTS,
                        help="Override Claude projects dir")
    parser.add_argument("--source", default="claude_code", choices=list(ADAPTERS),
                        help=f"CLI tool source: {', '.join(ADAPTERS)}")
    args = parser.parse_args()

    source = normalize_source(args.source)
    session_path = args.session
    if session_path is None and source == "claude_code":
        session_path = find_latest_session(args.projects_dir)
        if session_path is None:
            print(f"No sessions found in {args.projects_dir}", file=sys.stderr)
            return 1

    if args.watch:
        watch(session_path, args.lookback, json_out=args.json, color=not args.no_color,
              source=source)
        return 0

    result = analyze(session_path, args.lookback, source=source)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result, color=not args.no_color))
    return 0


if __name__ == "__main__":
    sys.exit(main())
