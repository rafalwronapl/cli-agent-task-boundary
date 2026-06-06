from __future__ import annotations

import json
from contextlib import contextmanager

import reporter
from detector import UserPrompt
from token_economy import TokenStats


def test_session_metric_is_anonymous(monkeypatch):
    session_id = "real-session-id"
    prompts = [
        UserPrompt(text="fix parser", tokens=2),
        UserPrompt(text="switching gears, new task", tokens=6),
    ]
    stats = TokenStats(source="claude_code", n_turns=2, input_tokens=100, output_tokens=40)
    stats._latest_context = 100
    monkeypatch.setattr(reporter, "get_prompts_for_session", lambda source, meta: prompts)
    monkeypatch.setattr(reporter, "get_token_stats", lambda source, meta, p: stats)

    metric = reporter.session_metric(
        "claude_code",
        {"session_id": session_id, "last_activity": 1_700_000_000, "path": "private/path"},
        "test-secret",
        "engineering",
    )
    serialized = json.dumps(metric)

    assert metric is not None
    assert metric["session_id_hash"] != session_id
    assert metric["agent_id_hash"]
    assert metric["decision"] == "new_task"
    assert session_id not in serialized
    assert "private/path" not in serialized
    assert "fix parser" not in serialized
    assert "switching gears" not in serialized
    assert "prompt_text" not in serialized.lower()
    assert "response" not in serialized.lower()
    assert "path" not in serialized.lower()


def test_event_id_is_stable_for_unchanged_snapshot(monkeypatch):
    prompts = [UserPrompt(text="hello", tokens=1)]
    stats = TokenStats(source="codex", input_tokens=3, output_tokens=2, is_exact=False)
    stats._latest_context = 3
    monkeypatch.setattr(reporter, "get_prompts_for_session", lambda source, meta: prompts)
    monkeypatch.setattr(reporter, "get_token_stats", lambda source, meta, p: stats)
    meta = {"session_id": "same", "last_activity": 1_700_000_000}

    first = reporter.session_metric("codex", meta, "secret", "team")
    second = reporter.session_metric("codex", meta, "secret", "team")

    assert first["event_id"] == second["event_id"]


def test_post_metrics_sends_bearer_token(monkeypatch):
    captured = {}

    @contextmanager
    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers.get("Authorization")
        captured["timeout"] = timeout

        class Response:
            def read(self):
                return b'{"inserted": 1, "duplicate": 0}'

        yield Response()

    monkeypatch.setattr(reporter.urllib.request, "urlopen", fake_urlopen)

    result = reporter.post_metrics("http://server.local/", [{"event_id": "abc"}], "secret-token")

    assert result["inserted"] == 1
    assert captured["url"] == "http://server.local/metrics"
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 15
