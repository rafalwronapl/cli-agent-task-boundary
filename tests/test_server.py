from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

import server.app as backend
import server.__main__ as server_main


def _metric(event_id: str = "event-id-1234567890", cost: float = 1.25) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "event_id": event_id,
        "agent_id_hash": "agenthash12345678",
        "session_id_hash": "sessionhash12345",
        "team_id": "engineering",
        "source": "claude_code",
        "model": "claude-sonnet",
        "n_prompts": 12,
        "tokens_billed": 1000,
        "context_used": 500,
        "context_pct": 0.25,
        "cost_usd": cost,
        "decision": "new_task",
        "confidence": 0.8,
        "boundary_age_prompts": 1,
        "drift_score": None,
        "duration_sec": 300,
        "is_exact_cost": True,
        "observed_at": now,
        "reported_at": now,
    }


def test_metrics_ingest_deduplicates_and_aggregates(tmp_path, monkeypatch):
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.duckdb")
    with TestClient(backend.app) as client:
        first = client.post("/metrics", json={"metrics": [_metric()]})
        second = client.post("/metrics", json={"metrics": [_metric()]})
        aggregate = client.get("/aggregate?days=30")

    assert first.status_code == 200
    assert first.json()["inserted"] == 1
    assert second.json()["duplicate"] == 1
    assert aggregate.status_code == 200
    assert aggregate.json()["summary"]["sessions"] == 1
    assert aggregate.json()["summary"]["leaky_sessions"] == 1
    assert aggregate.json()["summary"]["cost_usd"] == 1.25


def test_metrics_reject_unknown_privacy_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.duckdb")
    bad = _metric()
    bad["prompt_text"] = "must never arrive here"
    with TestClient(backend.app) as client:
        response = client.post("/metrics", json={"metrics": [bad]})

    assert response.status_code == 422


def test_health_does_not_expose_database_path(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.duckdb")
    with TestClient(backend.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_token_protects_metrics_and_aggregate(tmp_path, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "secret-token")
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.duckdb")
    with TestClient(backend.app) as client:
        denied_post = client.post("/metrics", json={"metrics": [_metric()]})
        denied_get = client.get("/aggregate?days=30")
        accepted = client.post(
            "/metrics",
            json={"metrics": [_metric()]},
            headers={"Authorization": "Bearer secret-token"},
        )
        aggregate = client.get(
            "/aggregate?days=30",
            headers={"X-FinOps-Token": "secret-token"},
        )

    assert denied_post.status_code == 401
    assert denied_get.status_code == 401
    assert accepted.status_code == 200
    assert aggregate.status_code == 200
    assert aggregate.json()["summary"]["sessions"] == 1


def test_aggregate_team_filter_is_parameterized(tmp_path, monkeypatch):
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)
    monkeypatch.setattr(backend, "DB_PATH", tmp_path / "test.duckdb")
    with TestClient(backend.app) as client:
        first = _metric(event_id="event-id-1234567890")
        first["team_id"] = "engineering"
        second = _metric(event_id="event-id-abcdef123456")
        second["team_id"] = "sales' OR 1=1 --"
        client.post("/metrics", json={"metrics": [first, second]})
        filtered = client.get("/aggregate", params={"days": 30, "team": "sales' OR 1=1 --"})

    assert filtered.status_code == 200
    assert filtered.json()["summary"]["sessions"] == 1
    assert filtered.json()["teams"][0]["team_id"] == "sales' OR 1=1 --"


def test_public_bind_requires_api_token(monkeypatch, capsys):
    monkeypatch.setenv("FINOPS_HOST", "0.0.0.0")
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)

    status = server_main.main()

    assert status == 2
    assert "FINOPS_API_TOKEN" in capsys.readouterr().err
