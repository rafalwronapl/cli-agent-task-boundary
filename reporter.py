"""Privacy-first FinOps reporter for local AI coding sessions.

The reporter reads local session logs through the existing adapters, computes
cost and boundary metrics locally, and sends only anonymous numeric snapshots
to the FinOps server. Prompt text, response text, paths, and project names are
never included in the outgoing payload.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS, get_prompts_for_session, list_sessions
from detector import classify_boundary, find_last_new_task_prompt
from token_economy import get_token_stats

SCHEMA_VERSION = 1
DEFAULT_SERVER_URL = "http://127.0.0.1:8787"
DEFAULT_IDENTITY_PATH = Path.home() / ".task-boundary-detector-agent"


def _utc_iso(timestamp: float | None = None) -> str:
    dt = datetime.fromtimestamp(timestamp, timezone.utc) if timestamp else datetime.now(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def load_or_create_agent_secret(path: Path = DEFAULT_IDENTITY_PATH) -> str:
    """Return a local secret used for non-reversible, stable identifiers."""
    env_secret = os.environ.get("FINOPS_AGENT_SECRET", "").strip()
    if env_secret:
        return env_secret
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    secret = secrets.token_hex(32)
    path.write_text(secret + "\n", encoding="ascii")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return secret


def _anonymous_id(secret: str, value: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def _event_id(metric: dict[str, Any]) -> str:
    stable = "|".join(
        str(metric[key])
        for key in (
            "agent_id_hash", "session_id_hash", "source", "observed_at",
            "n_prompts", "tokens_billed", "cost_usd", "decision",
        )
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]


def session_metric(
    source: str,
    meta: dict[str, Any],
    secret: str,
    team_id: str,
    with_embeddings: bool = False,
) -> dict[str, Any] | None:
    """Compute one anonymous metric. Returns None for unreadable sessions."""
    prompts = get_prompts_for_session(source, meta)
    if not prompts:
        return None
    boundary = classify_boundary(prompts)
    decision = "fresh" if len(prompts) < 2 else boundary.decision
    confidence = boundary.confidence
    drift_score = None

    if with_embeddings:
        from embeddings_detector import combine_with_regex, compute_topic_drift

        drift = compute_topic_drift(prompts)
        drift_score = drift.drift_score
        decision, confidence, _ = combine_with_regex(decision, confidence, drift)

    stats = get_token_stats(source, meta, prompts)
    boundary_idx = find_last_new_task_prompt(prompts)
    boundary_age = None if boundary_idx is None else len(prompts) - boundary_idx - 1
    observed_at = _utc_iso(float(meta["last_activity"]))
    real_session_id = str(meta.get("session_id", ""))
    metric: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "agent_id_hash": _anonymous_id(secret, "agent"),
        "session_id_hash": _anonymous_id(secret, f"{source}:{real_session_id}"),
        "team_id": team_id,
        "source": source,
        "model": stats.model or "unknown",
        "n_prompts": len(prompts),
        "tokens_billed": stats.total_billed,
        "context_used": stats.latest_context,
        "context_pct": round(stats.context_used_pct, 6),
        "cost_usd": round(stats.cost_usd, 8),
        "decision": decision,
        "confidence": round(float(confidence), 6),
        "boundary_age_prompts": boundary_age,
        "drift_score": None if drift_score is None else round(float(drift_score), 6),
        "duration_sec": round(stats.duration_sec, 3),
        "is_exact_cost": bool(stats.is_exact and source == "claude_code"),
        "observed_at": observed_at,
        "reported_at": _utc_iso(),
    }
    metric["event_id"] = _event_id(metric)
    return metric


def collect_metrics(
    sources: list[str],
    max_age_hours: float,
    secret: str,
    team_id: str = "default",
    with_embeddings: bool = False,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for source in sources:
        for meta in list_sessions(source, max_age_hours=max_age_hours):
            try:
                metric = session_metric(source, meta, secret, team_id, with_embeddings)
            except Exception as exc:
                print(f"Skipping {source} session: {exc}", file=sys.stderr)
                continue
            if metric:
                metrics.append(metric)
    return metrics


def post_metrics(server_url: str, metrics: list[dict[str, Any]], api_token: str = "") -> dict[str, Any]:
    endpoint = server_url.rstrip("/") + "/metrics"
    payload = json.dumps({"schema_version": SCHEMA_VERSION, "metrics": metrics}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_once(args: argparse.Namespace, secret: str) -> int:
    sources = list(ADAPTERS) if args.source == "all" else [args.source]
    metrics = collect_metrics(
        sources=sources,
        max_age_hours=args.max_age_hours,
        secret=secret,
        team_id=args.team,
        with_embeddings=args.with_embeddings,
    )
    if args.dry_run:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "metrics": metrics}, indent=2))
        return 0
    try:
        result = post_metrics(args.server_url, metrics, args.api_token)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"FinOps upload failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Reported {len(metrics)} snapshot(s); "
        f"inserted={result.get('inserted', '?')}, duplicate={result.get('duplicate', '?')}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Anonymous FinOps metric reporter")
    parser.add_argument("--server-url", default=os.environ.get("FINOPS_SERVER_URL", DEFAULT_SERVER_URL))
    parser.add_argument("--api-token", default=os.environ.get("FINOPS_API_TOKEN", ""))
    parser.add_argument("--team", default=os.environ.get("FINOPS_TEAM", "default"))
    parser.add_argument("--source", choices=["all", *ADAPTERS.keys()], default="all")
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between uploads in watch mode.")
    parser.add_argument("--watch", action="store_true", help="Continue reporting at --interval.")
    parser.add_argument("--dry-run", action="store_true", help="Print payload and do not upload.")
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Compute optional local semantic drift. May download/load the embedding model.",
    )
    args = parser.parse_args()

    secret = load_or_create_agent_secret()
    if not args.watch:
        return _run_once(args, secret)
    while True:
        status = _run_once(args, secret)
        if status:
            print("Retrying at the next interval.", file=sys.stderr)
        try:
            time.sleep(max(args.interval, 5.0))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
