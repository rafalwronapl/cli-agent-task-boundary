"""Source adapters — różne CLI tools mają różne formaty session logs.

Każdy adapter zwraca list[UserPrompt] dla detector.classify_boundary().
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from detector import UserPrompt


# ---- Claude Code (default, current impl in detector.py) ---------------------

def claude_code_path() -> Path:
    return Path.home() / ".claude" / "projects"


def claude_code_latest_session() -> Path | None:
    """Alias of detector.find_latest_session — kept for callers that import
    via adapters. Single implementation lives in detector.py."""
    from detector import find_latest_session
    return find_latest_session()


# ---- Codex CLI (OpenAI) ------------------------------------------------------

def codex_cli_path() -> Path:
    return Path.home() / ".codex" / "history.jsonl"


def parse_codex_cli(path: Path) -> list[UserPrompt]:
    """Codex CLI format: {session_id, ts, text}, each line = user prompt.

    Note: Codex stores ALL prompts in single history.jsonl across ALL sessions,
    distinguished by session_id. We return only the LATEST session.
    """
    if not path.exists():
        return []

    sessions: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sid = d.get("session_id")
                if sid:
                    sessions.setdefault(sid, []).append(d)
            except json.JSONDecodeError:
                continue

    if not sessions:
        return []

    # Latest session = session whose last ts is highest
    latest_sid = max(sessions, key=lambda s: max(e.get("ts", 0) for e in sessions[s]))
    events = sessions[latest_sid]

    prompts = []
    for ev in events:
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        prompts.append(UserPrompt(
            text=text,
            timestamp=str(ev.get("ts", "")),
            tokens=len(text) // 4,
        ))
    return prompts


# ---- Dispatch ---------------------------------------------------------------

ADAPTERS = {
    "claude_code": "Claude Code (~/.claude/projects/)",
    "codex":       "OpenAI Codex CLI (~/.codex/history.jsonl)",
    "gemini":      "Google Gemini CLI (~/.gemini/tmp/.../chats/)",
}

# Adapters kept in code but NOT exposed in UI because they are unverified or
# fundamentally blocked from reading message bodies. See parse_cursor_sqlite
# and parse_opencode_session for details. To re-enable, add the key back to
# ADAPTERS above.
_HIDDEN_ADAPTERS = {
    # Cursor stores message bodies in their cloud — local SQLite only has
    # composer metadata (composerId, line counts, mode). Useless for boundary
    # detection. Would need to call Cursor's non-public API to fetch content.
    "cursor":   "Cursor IDE — message bodies stored remotely (BLOCKED)",
    # OpenCode (sst/opencode, TS/Bun) uses XDG paths + SQLite (opencode.db) with
    # relational sessions/messages/parts tables. The current parse_opencode_session
    # implementation assumes per-session JSON files in ~/.opencode/sessions/ —
    # WRONG. Needs SQLite rewrite against a real install before re-enabling.
    "opencode": "OpenCode — adapter is wrong (SQLite, not JSON). TODO rewrite.",
}

# Backwards-compat: callers that imported SOURCE_ALIAS / normalize_source still work.
SOURCE_ALIAS: dict[str, str] = {}


def normalize_source(name: str) -> str:
    """Resolve a UI-facing source name to the canonical adapter key."""
    return SOURCE_ALIAS.get(name, name)


# ---- last AI response extraction (for detail panel) -----------------------

def get_last_ai_response(source: str, session_meta: dict, max_chars: int = 600) -> str:
    """Return last AI/assistant response text for the session."""
    path = Path(session_meta["path"])
    if not path.exists():
        return ""

    if source == "claude_code":
        last_text = ""
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("type") != "assistant":
                        continue
                    msg = d.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        last_text = content
                    elif isinstance(content, list):
                        texts = [c.get("text", "") for c in content
                                 if isinstance(c, dict) and c.get("type") == "text"]
                        if texts:
                            last_text = " ".join(texts)
                except json.JSONDecodeError:
                    continue
        return last_text[:max_chars].strip()

    # Codex doesn't store AI responses in history.jsonl (only user prompts)
    return ""


# ---- OpenCode ---------------------------------------------------------------

def opencode_paths() -> list[Path]:
    """Possible OpenCode session storage locations."""
    home = Path.home()
    return [
        home / ".opencode" / "sessions",
        home / ".local" / "share" / "opencode" / "sessions",
        home / "AppData" / "Roaming" / "opencode" / "sessions",
    ]


def parse_opencode_session(path: Path) -> list[UserPrompt]:
    """OpenCode session JSON: {messages: [{role, content, ts}]}"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    prompts = []
    msgs = data.get("messages", []) if isinstance(data, dict) else data
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        text = str(content).strip()
        if text:
            prompts.append(UserPrompt(
                text=text,
                timestamp=str(m.get("ts", m.get("timestamp", ""))),
                tokens=len(text) // 4,
            ))
    return prompts


# ---- Gemini CLI -------------------------------------------------------------

def gemini_chats_roots() -> list[Path]:
    """Return all `<.gemini>/tmp/*/chats/` directories — one per project."""
    base = Path.home() / ".gemini" / "tmp"
    if not base.exists():
        return []
    out = []
    for project_dir in base.iterdir():
        if not project_dir.is_dir():
            continue
        chats = project_dir / "chats"
        if chats.exists():
            out.append(chats)
    return out


def parse_gemini_session(path: Path) -> list[UserPrompt]:
    """Gemini CLI session JSONL.

    Line 1 = session header (skip — sessionId/projectHash/startTime).
    Subsequent lines are either:
      - message records: {type, content, timestamp, ...}
      - delta updates: {"$set": {...}}  — skip

    User messages: type="user", content=[{text:"..."}] OR content="..."
    """
    if not path.exists():
        return []
    prompts: list[UserPrompt] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0 and "sessionId" in d:
                continue  # session header
            if "$set" in d:
                continue  # delta update
            if d.get("type") != "user":
                continue
            content = d.get("content") or d.get("message")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("text")
                )
            text = text.strip()
            if not text:
                continue
            prompts.append(UserPrompt(
                text=text,
                timestamp=str(d.get("timestamp", "")),
                tokens=len(text) // 4,
            ))
    return prompts


# ---- Cursor IDE -------------------------------------------------------------

def cursor_workspace_storage() -> Path:
    return (Path.home() / "AppData" / "Roaming" / "Cursor" /
            "User" / "workspaceStorage")


def parse_cursor_sqlite(db_path: Path) -> list[UserPrompt]:
    """Cursor stores composer metadata in workspace state.vscdb.

    NOTE: Actual messages may be stored elsewhere (cloud sync, separate DB).
    This adapter returns BEST-EFFORT — composer creation timestamps + ids only.
    For full message history we'd need to reverse-engineer Cursor's storage.
    """
    if not db_path.exists():
        return []
    import sqlite3
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
        ).fetchone()
        con.close()
        if not row:
            return []
        d = json.loads(row[0])
        composers = d.get("allComposers", [])
        prompts = []
        for c in composers:
            if c.get("isArchived"):
                continue
            # Best-effort: composer metadata as pseudo-prompt
            ts = c.get("createdAt", 0)
            text = (f"[Cursor composer {c.get('composerId', '')[:12]}] "
                    f"mode={c.get('unifiedMode', '?')} "
                    f"+{c.get('totalLinesAdded', 0)}/-{c.get('totalLinesRemoved', 0)}")
            prompts.append(UserPrompt(text=text, timestamp=str(ts), tokens=len(text)//4))
        return prompts
    except Exception:
        return []


# ---- multi-session listing -------------------------------------------------

import time as _time


def list_sessions(source: str, max_age_hours: float = 24) -> list[dict]:
    """List active sessions for a source.

    Returns list of {session_id, label, last_activity, n_prompts, path}.
    Sorted by last activity desc.
    """
    now = _time.time()
    cutoff = now - max_age_hours * 3600
    out = []

    if source == "claude_code":
        root = claude_code_path()
        if not root.exists():
            return []
        for path in root.rglob("*.jsonl"):
            mtime = path.stat().st_mtime
            if mtime < cutoff:
                continue
            cache_key = (str(path), mtime)
            cached = _SESSION_META_CACHE.get(cache_key)
            if cached is not None:
                out.append(cached)
                continue
            try:
                from detector import extract_user_prompts, read_session
                events = read_session(path)
                prompts = extract_user_prompts(events)
                # Populate the prompts cache too so the second-stage parse
                # is a hit instead of another full JSONL scan.
                prompts_key = (str(path), source, path.stem, mtime)
                _PROMPTS_CACHE[prompts_key] = prompts
                cwd = ""
                for ev in events[:10]:
                    if ev.get("cwd"):
                        cwd = ev["cwd"]
                        break
                if not prompts:
                    continue
                meta = {
                    "session_id": path.stem,
                    "label": f"{path.stem[:12]} ({cwd or '?'})",
                    "last_activity": mtime,
                    "n_prompts": len(prompts),
                    "path": str(path),
                }
                _SESSION_META_CACHE[cache_key] = meta
                # Cap cache size
                if len(_SESSION_META_CACHE) > 64:
                    for k in list(_SESSION_META_CACHE.keys())[:16]:
                        _SESSION_META_CACHE.pop(k, None)
                out.append(meta)
            except Exception:
                continue

    elif source == "codex":
        path = codex_cli_path()
        if not path.exists():
            return []
        sessions: dict[str, list[dict]] = {}
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    sid = d.get("session_id")
                    if sid:
                        sessions.setdefault(sid, []).append(d)
                except json.JSONDecodeError:
                    continue
        for sid, events in sessions.items():
            ts_list = [e.get("ts", 0) for e in events if e.get("ts")]
            if not ts_list:
                continue
            last = max(ts_list)
            if last < cutoff:
                continue
            # First prompt preview
            first_text = events[0].get("text", "")[:60]
            out.append({
                "session_id": sid,
                "label": f"{sid[:12]} ({first_text[:50]}{'...' if len(first_text) > 50 else ''})",
                "last_activity": float(last),
                "n_prompts": len(events),
                "path": str(path),
                "_codex_sid": sid,
            })

    elif source == "gemini":
        for chats_root in gemini_chats_roots():
            for path in chats_root.glob("session-*.jsonl"):
                mtime = path.stat().st_mtime
                if mtime < cutoff:
                    continue
                try:
                    prompts = parse_gemini_session(path)
                    if not prompts:
                        continue
                    # session id from filename suffix
                    sid = path.stem.split("-")[-1]
                    project_user = chats_root.parent.name
                    out.append({
                        "session_id": sid,
                        "label": f"{sid[:10]} ({project_user})",
                        "last_activity": mtime,
                        "n_prompts": len(prompts),
                        "path": str(path),
                    })
                except Exception:
                    continue

    elif source == "opencode":
        for root in opencode_paths():
            if not root.exists():
                continue
            for path in list(root.glob("*.json")) + list(root.glob("*.jsonl")):
                mtime = path.stat().st_mtime
                if mtime < cutoff:
                    continue
                try:
                    prompts = parse_opencode_session(path)
                    if not prompts:
                        continue
                    out.append({
                        "session_id": path.stem,
                        "label": path.stem[:40],
                        "last_activity": mtime,
                        "n_prompts": len(prompts),
                        "path": str(path),
                    })
                except Exception:
                    continue

    elif source == "cursor":
        root = cursor_workspace_storage()
        if not root.exists():
            return []
        for ws_dir in root.iterdir():
            if not ws_dir.is_dir():
                continue
            db = ws_dir / "state.vscdb"
            if not db.exists():
                continue
            mtime = db.stat().st_mtime
            if mtime < cutoff:
                continue
            try:
                prompts = parse_cursor_sqlite(db)
                if not prompts:
                    continue
                out.append({
                    "session_id": ws_dir.name,
                    "label": f"workspace {ws_dir.name[:14]}",
                    "last_activity": mtime,
                    "n_prompts": len(prompts),
                    "path": str(db),
                })
            except Exception:
                continue

    out.sort(key=lambda x: x["last_activity"], reverse=True)
    return out


# Per-(path, mtime) cache to avoid re-parsing huge JSONLs on refresh
_PROMPTS_CACHE: dict[tuple, list[UserPrompt]] = {}
# Per-(path, mtime) cache for list_sessions metadata — avoids reading every
# JSONL twice (once for listing, once for prompts) on each refresh.
_SESSION_META_CACHE: dict[tuple, dict] = {}


def get_prompts_for_session(source: str, session_meta: dict) -> list[UserPrompt]:
    """Get prompts for a specific session (from list_sessions output)."""
    path = Path(session_meta["path"])
    cache_key = (str(path), source, session_meta.get("session_id"),
                  path.stat().st_mtime if path.exists() else 0)
    cached = _PROMPTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Cap cache size
    if len(_PROMPTS_CACHE) > 64:
        _PROMPTS_CACHE.clear()

    if source == "claude_code":
        from detector import extract_user_prompts, read_session
        result = extract_user_prompts(read_session(path))
        _PROMPTS_CACHE[cache_key] = result
        return result
    if source == "codex":
        sid = session_meta.get("_codex_sid") or session_meta["session_id"]
        prompts = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("session_id") != sid:
                        continue
                    text = (d.get("text") or "").strip()
                    if text:
                        prompts.append(UserPrompt(
                            text=text, timestamp=str(d.get("ts", "")),
                            tokens=len(text) // 4,
                        ))
                except json.JSONDecodeError:
                    continue
        _PROMPTS_CACHE[cache_key] = prompts
        return prompts
    if source == "gemini":
        result = parse_gemini_session(path)
        _PROMPTS_CACHE[cache_key] = result
        return result
    if source == "opencode":
        result = parse_opencode_session(path)
        _PROMPTS_CACHE[cache_key] = result
        return result
    if source == "cursor":
        result = parse_cursor_sqlite(path)
        _PROMPTS_CACHE[cache_key] = result
        return result
    return []


def get_prompts(source: str, path: Path | None = None) -> list[UserPrompt]:
    """Get user prompts from any supported source."""
    if source == "claude_code":
        from detector import extract_user_prompts, read_session
        p = path or claude_code_latest_session()
        if not p:
            return []
        return extract_user_prompts(read_session(p))

    if source == "codex":
        return parse_codex_cli(path or codex_cli_path())

    if source == "opencode":
        if not path:
            for root in opencode_paths():
                if root.exists():
                    files = sorted(root.glob("*.json"),
                                   key=lambda p: p.stat().st_mtime, reverse=True)
                    if files:
                        path = files[0]
                        break
        return parse_opencode_session(path) if path else []

    if source == "cursor":
        if not path:
            root = cursor_workspace_storage()
            if root.exists():
                dbs = []
                for ws in root.iterdir():
                    db = ws / "state.vscdb"
                    if db.exists():
                        dbs.append(db)
                if dbs:
                    path = max(dbs, key=lambda p: p.stat().st_mtime)
        return parse_cursor_sqlite(path) if path else []

    raise ValueError(f"Unknown source: {source}. Available: {list(ADAPTERS)}")
