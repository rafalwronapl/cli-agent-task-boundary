"""FastAPI + DuckDB backend for anonymous Task Boundary FinOps metrics."""
from __future__ import annotations

import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "finops.duckdb"
DASHBOARD = ROOT / "dashboard"
DB_PATH = Path(os.environ.get("FINOPS_DB_PATH", DEFAULT_DB))
_LOCK = threading.Lock()


class MetricIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    event_id: str = Field(min_length=16, max_length=64)
    agent_id_hash: str = Field(min_length=8, max_length=64)
    session_id_hash: str = Field(min_length=8, max_length=64)
    team_id: str = Field(default="default", min_length=1, max_length=80)
    source: Literal["claude_code", "codex", "gemini"]
    model: str = Field(default="unknown", max_length=120)
    n_prompts: int = Field(ge=0)
    tokens_billed: int = Field(ge=0)
    context_used: int = Field(ge=0)
    context_pct: float = Field(ge=0.0, le=10.0)
    cost_usd: float = Field(ge=0.0)
    decision: Literal["new_task", "continuation", "task_complete", "fresh", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    boundary_age_prompts: int | None = Field(default=None, ge=0)
    drift_score: float | None = Field(default=None, ge=-1.0, le=2.0)
    duration_sec: float = Field(ge=0.0)
    is_exact_cost: bool = False
    observed_at: datetime
    reported_at: datetime


class MetricsBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    metrics: list[MetricIn] = Field(default_factory=list, max_length=5000)


def _connect() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def configured_api_token() -> str:
    return os.environ.get("FINOPS_API_TOKEN", "").strip()


def require_api_token(
    authorization: str | None = Header(default=None),
    x_finops_token: str | None = Header(default=None),
) -> None:
    expected = configured_api_token()
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    elif x_finops_token:
        supplied = x_finops_token.strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid FinOps API token.",
        )


def init_db() -> None:
    with _LOCK:
        con = _connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    event_id VARCHAR PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    agent_id_hash VARCHAR NOT NULL,
                    session_id_hash VARCHAR NOT NULL,
                    team_id VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    model VARCHAR NOT NULL,
                    n_prompts INTEGER NOT NULL,
                    tokens_billed BIGINT NOT NULL,
                    context_used BIGINT NOT NULL,
                    context_pct DOUBLE NOT NULL,
                    cost_usd DOUBLE NOT NULL,
                    decision VARCHAR NOT NULL,
                    confidence DOUBLE NOT NULL,
                    boundary_age_prompts INTEGER,
                    drift_score DOUBLE,
                    duration_sec DOUBLE NOT NULL,
                    is_exact_cost BOOLEAN NOT NULL,
                    observed_at TIMESTAMP NOT NULL,
                    reported_at TIMESTAMP NOT NULL,
                    received_at TIMESTAMP NOT NULL
                )
                """
            )
        finally:
            con.close()


def _dt(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Task Boundary FinOps",
    description="Anonymous cost and context-waste metrics for AI-assisted development sessions.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    init_db()
    return {"status": "ok"}


@app.post("/metrics")
def ingest(batch: MetricsBatch, _: None = Depends(require_api_token)) -> dict:
    init_db()
    received_at = datetime.now(timezone.utc).replace(tzinfo=None)
    inserted = 0
    with _LOCK:
        con = _connect()
        try:
            for metric in batch.metrics:
                before = con.execute("SELECT COUNT(*) FROM metrics WHERE event_id = ?", [metric.event_id]).fetchone()[0]
                if before:
                    continue
                con.execute(
                    """
                    INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        metric.event_id, metric.schema_version, metric.agent_id_hash,
                        metric.session_id_hash, metric.team_id, metric.source, metric.model,
                        metric.n_prompts, metric.tokens_billed, metric.context_used,
                        metric.context_pct, metric.cost_usd, metric.decision, metric.confidence,
                        metric.boundary_age_prompts, metric.drift_score, metric.duration_sec,
                        metric.is_exact_cost, _dt(metric.observed_at), _dt(metric.reported_at),
                        received_at,
                    ],
                )
                inserted += 1
        finally:
            con.close()
    return {"received": len(batch.metrics), "inserted": inserted, "duplicate": len(batch.metrics) - inserted}


@app.get("/aggregate")
def aggregate(
    days: int = Query(default=30, ge=1, le=365),
    team: str | None = None,
    _: None = Depends(require_api_token),
) -> dict:
    init_db()
    condition = "observed_at >= current_timestamp - (? * INTERVAL '1 day')"
    params: list[object] = [days]
    if team:
        condition += " AND team_id = ?"
        params.append(team)
    with _LOCK:
        con = _connect()
        try:
            cte = f"""
                WITH filtered AS (
                    SELECT * FROM metrics WHERE {condition}
                ),
                latest AS (
                    SELECT * EXCLUDE (rn) FROM (
                        SELECT *, row_number() OVER (
                            PARTITION BY agent_id_hash, session_id_hash, source
                            ORDER BY observed_at DESC, received_at DESC
                        ) AS rn FROM filtered
                    ) WHERE rn = 1
                )
            """
            summary = con.execute(
                cte
                + """
                SELECT count(*) AS sessions,
                       count(DISTINCT agent_id_hash) AS agents,
                       coalesce(sum(cost_usd), 0) AS cost_usd,
                       coalesce(sum(tokens_billed), 0) AS tokens_billed,
                       count(*) FILTER (WHERE decision = 'new_task') AS leaky_sessions,
                       coalesce(sum(cost_usd) FILTER (WHERE decision = 'new_task'), 0) AS leaky_cost_usd
                FROM latest
                """,
                params,
            ).fetchone()
            teams = con.execute(
                cte
                + """
                SELECT team_id, count(*) AS sessions, sum(cost_usd) AS cost_usd,
                       sum(tokens_billed) AS tokens_billed,
                       count(*) FILTER (WHERE decision = 'new_task') AS leaky_sessions
                FROM latest GROUP BY team_id ORDER BY cost_usd DESC
                """,
                params,
            ).fetchall()
            top_sessions = con.execute(
                cte
                + """
                SELECT session_id_hash, team_id, source, model, decision, n_prompts,
                       tokens_billed, context_pct, cost_usd, boundary_age_prompts,
                       observed_at
                FROM latest
                ORDER BY CASE WHEN decision = 'new_task' THEN 0 ELSE 1 END,
                         cost_usd DESC LIMIT 20
                """,
                params,
            ).fetchall()
            trend = con.execute(
                cte
                + """
                , deltas AS (
                    SELECT date_trunc('day', observed_at) AS day, agent_id_hash,
                           session_id_hash, source, cost_usd,
                           lag(cost_usd) OVER (
                               PARTITION BY agent_id_hash, session_id_hash, source
                               ORDER BY observed_at, received_at
                           ) AS previous_cost
                    FROM filtered
                )
                SELECT CAST(day AS DATE), sum(
                    greatest(cost_usd - coalesce(previous_cost, 0), 0)
                ) AS incremental_cost
                FROM deltas GROUP BY day ORDER BY day
                """,
                params,
            ).fetchall()
            decisions = con.execute(
                cte
                + "SELECT decision, count(*) FROM latest GROUP BY decision ORDER BY count(*) DESC",
                params,
            ).fetchall()
        finally:
            con.close()
    return {
        "days": days,
        "team": team,
        "summary": {
            "sessions": summary[0],
            "agents": summary[1],
            "cost_usd": round(summary[2], 6),
            "tokens_billed": summary[3],
            "leaky_sessions": summary[4],
            "leaky_cost_usd": round(summary[5], 6),
        },
        "teams": [
            {"team_id": row[0], "sessions": row[1], "cost_usd": round(row[2], 6),
             "tokens_billed": row[3], "leaky_sessions": row[4]}
            for row in teams
        ],
        "top_sessions": [
            {"session_id_hash": row[0], "team_id": row[1], "source": row[2],
             "model": row[3], "decision": row[4], "n_prompts": row[5],
             "tokens_billed": row[6], "context_pct": row[7], "cost_usd": round(row[8], 6),
             "boundary_age_prompts": row[9], "observed_at": row[10].isoformat() + "Z"}
            for row in top_sessions
        ],
        "trend": [{"day": str(row[0]), "cost_usd": round(row[1], 6)} for row in trend],
        "decisions": [{"decision": row[0], "sessions": row[1]} for row in decisions],
    }


if DASHBOARD.exists():
    app.mount("/assets", StaticFiles(directory=str(DASHBOARD)), name="assets")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD / "index.html")
