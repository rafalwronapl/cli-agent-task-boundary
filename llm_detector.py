"""Layer 3 — LLM-based task boundary detection.

Two pluggable backends:
  * OpenRouter (cloud, paid)  — set OPENROUTER_API_KEY
  * Ollama (local, free)      — set LLM_BACKEND=ollama and have Ollama running

Backend selection (in order):
  1. LLM_BACKEND env var — explicit choice ("openrouter" | "ollama")
  2. Auto: if OPENROUTER_API_KEY present → openrouter
  3. Auto: if Ollama reachable at localhost:11434 → ollama
  4. Otherwise: L3 disabled (L1+L2 still work)

Cost: OpenRouter DeepSeek V4 Flash ≈ $1e-7 per check. Ollama: $0.
Cached per (session_id, last_prompt_hash) — no call if prompts unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

# Lazy state
_AVAILABLE: bool | None = None
_BACKEND: str | None = None  # "openrouter" | "ollama" | "off"
_CACHE: dict[str, "LLMVerdict"] = {}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Pricing per M tokens (OpenRouter, 2026-05-23). Ollama is free → 0.
PRICE_INPUT = 0.0001 / 1_000_000
PRICE_OUTPUT = 0.0002 / 1_000_000


def _backend() -> str:
    """Pick the LLM backend once per process. See module docstring."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    _load_env()
    explicit = os.environ.get("LLM_BACKEND", "").strip().lower()
    if explicit in ("openrouter", "ollama"):
        _BACKEND = explicit
        return _BACKEND
    if os.environ.get("OPENROUTER_API_KEY", "").strip().startswith("sk-"):
        _BACKEND = "openrouter"
        return _BACKEND
    if _ollama_reachable():
        _BACKEND = "ollama"
        return _BACKEND
    _BACKEND = "off"
    return _BACKEND


def _ollama_reachable() -> bool:
    """Quick TCP check — does Ollama answer on its default port?"""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        host, _, port = url.removeprefix("http://").removeprefix("https://").partition(":")
        port_int = int((port or "11434").rstrip("/"))
        with socket.create_connection((host or "localhost", port_int), timeout=0.3):
            return True
    except Exception:
        return False


def _model_id() -> str:
    """Resolve the model id from env, with sensible per-backend defaults."""
    if _backend() == "ollama":
        return os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    return os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")


def _api_url() -> str:
    if _backend() == "ollama":
        base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        return f"{base}/v1/chat/completions"
    return OPENROUTER_URL


def _api_headers() -> dict:
    if _backend() == "ollama":
        # Ollama's OpenAI-compatible endpoint ignores auth, but the OpenAI
        # client conventionally sends one. Any non-empty value works.
        return {"Authorization": "Bearer ollama", "Content-Type": "application/json"}
    key = os.environ["OPENROUTER_API_KEY"]
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/claude-task-boundary",
        "X-Title": "Task Boundary Detector",
    }




SYSTEM_PROMPT = """Jesteś klasyfikatorem zmian tematu w konwersacji programistycznej.

Dostaniesz ostatnie N user-promptów z sesji Claude Code / Codex CLI.
Twoje zadanie: czy OSTATNI prompt to NOWY TASK czy KONTYNUACJA poprzedniego?

KATEGORIE:
- "new_task" — użytkownik wyraźnie zaczyna nowy cel/temat (zmiana domeny, eksplicytne "zmienmy temat", przejście do innego projektu, off-topic pytanie)
- "task_complete" — poprzedni task wygląda na zakończony ("thanks", "działa", "gotowe" + brak follow-up)
- "continuation" — debugging/poprawki/iteracje tego samego zadania, dodawanie features, doprecyzowywanie

ZASADY:
1. Patrz na OSTATNI prompt + jego relację do POPRZEDNICH 4
2. Jeden mocny sygnał ("zmienmy temat") = new_task. Subtelne zmiany domeny też = new_task.
3. Nie myl debug→fix→test (= continuation) z prawdziwym task switch
4. Off-topic pytanie ("powiedz cos o mnie" gdy rozmawiacie o kodzie) = new_task

ODPOWIEDŹ — TYLKO JSON, bez markdown:
{"decision": "new_task|continuation|task_complete", "confidence": 0.0-1.0, "reasoning": "1 zdanie po polsku"}

Confidence:
- 0.9-1.0: bardzo pewne (oczywiste słowa-klucze)
- 0.7-0.8: pewne (semantic shift jasny)
- 0.5-0.6: nieoczywiste
- <0.5: bardzo wątpliwe (nie dawaj)
"""


@dataclass
class LLMVerdict:
    available: bool
    decision: str = "unclear"
    confidence: float = 0.0
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str = ""


def is_available() -> bool:
    """L3 is available if either OpenRouter has a key or Ollama is reachable."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    _AVAILABLE = _backend() != "off"
    return _AVAILABLE


def backend_name() -> str:
    """Public — which backend is being used? 'openrouter' | 'ollama' | 'off'."""
    return _backend()


def _load_env() -> None:
    """Try to load OPENROUTER_API_KEY from common .env locations."""
    candidates = [
        Path(__file__).parent / ".env",
        Path.home() / ".task-boundary-detector.env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
        except Exception:
            continue


def _hash_prompts(prompts: list) -> str:
    """Hash last 6 prompts to detect changes."""
    h = hashlib.sha256()
    for p in prompts[-6:]:
        h.update(p.text[:500].encode("utf-8", errors="replace"))
        h.update(b"\n||\n")
    return h.hexdigest()[:16]


def classify_via_llm(prompts: list, session_id: str = "",
                     lookback: int = 5) -> LLMVerdict:
    """Send last N user prompts to LLM, return classification."""
    if not is_available():
        return LLMVerdict(available=False,
                          error="L3 wyłączone — brak OPENROUTER_API_KEY ani Ollamy")
    if len(prompts) < 2:
        return LLMVerdict(available=True, decision="unclear",
                           confidence=0.0, error="Za mało promptów (<2)")

    # Cache key
    cache_key = f"{session_id}::{_hash_prompts(prompts)}"
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        # Return shallow copy with cached=True
        return LLMVerdict(
            available=cached.available, decision=cached.decision,
            confidence=cached.confidence, reasoning=cached.reasoning,
            input_tokens=cached.input_tokens, output_tokens=cached.output_tokens,
            cost_usd=cached.cost_usd, cached=True,
        )

    # Build user prompt
    recent = prompts[-(lookback+1):]
    if len(recent) < 2:
        return LLMVerdict(available=True, decision="unclear",
                           confidence=0.0, error="Za mało historii")

    lines = []
    for i, p in enumerate(recent):
        marker = "LAST" if i == len(recent) - 1 else f"prev-{len(recent)-1-i}"
        # Cap each prompt to keep cost low
        text = p.text[:600].replace("\n", " ")
        lines.append(f"[{marker}] {text}")
    user_prompt = "Ostatnie prompts:\n" + "\n".join(lines)

    try:
        import urllib.request
        payload = {
            "model": _model_id(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 200,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            _api_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=_api_headers(),
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return LLMVerdict(available=True, error=f"LLM call failed: {exc}")

    try:
        text = data["choices"][0]["message"]["content"]
        # Some models prepend reasoning — extract JSON object
        text = text.strip()
        if not text.startswith("{"):
            # Find first { and last }
            i = text.find("{")
            j = text.rfind("}")
            if i >= 0 and j > i:
                text = text[i:j+1]
        verdict_json = json.loads(text)
        usage = data.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        # Ollama runs locally → $0. OpenRouter uses real pricing.
        if _backend() == "ollama":
            cost = 0.0
        else:
            cost = in_tok * PRICE_INPUT + out_tok * PRICE_OUTPUT

        result = LLMVerdict(
            available=True,
            decision=verdict_json.get("decision", "unclear"),
            confidence=float(verdict_json.get("confidence", 0.0)),
            reasoning=str(verdict_json.get("reasoning", "")),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
        _CACHE[cache_key] = result
        # Cap cache
        if len(_CACHE) > 200:
            for k in list(_CACHE.keys())[:50]:
                _CACHE.pop(k, None)
        return result
    except Exception as exc:
        return LLMVerdict(available=True,
                           error=f"Failed to parse LLM response: {exc}",
                           reasoning=str(data)[:300])


def find_boundary_index(prompts: list, session_id: str = "") -> dict:
    """LLM identifies WHERE the new task started in the prompt list.

    Returns {boundary_index, reasoning, cost_usd}.
    boundary_index = position in prompts where new task begins (0 = all old, len = all continuation).
    """
    if not is_available() or len(prompts) < 2:
        return {"boundary_index": None, "reasoning": "n/a", "cost_usd": 0.0}

    _load_env()
    import os, json, urllib.request

    # Last 12 prompts max
    window = prompts[-12:]
    offset = len(prompts) - len(window)
    lines = []
    for i, p in enumerate(window):
        idx_global = i + offset
        text = p.text[:300].replace("\n", " ")
        lines.append(f"[{idx_global}] {text}")
    user_prompt = "Ostatnie prompts:\n" + "\n".join(lines)

    sys_prompt = (
        "Znajdź INDEKS pierwszego promptu który zaczyna NOWY TASK. "
        "Jeśli wszystko to kontynuacja jednego tasku → boundary_index = null. "
        "Jeśli ostatni prompt jest nowym taskiem ale poprzednie nie → boundary_index = ostatni index. "
        "Reasoning max 1 zdanie po polsku.\n\n"
        "Odpowiedź TYLKO JSON: "
        '{"boundary_index": int|null, "reasoning": "str"}'
    )

    try:
        payload = {
            "model": _model_id(),
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 200,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            _api_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=_api_headers(),
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        if not text.startswith("{"):
            i = text.find("{"); j = text.rfind("}")
            if i >= 0 and j > i: text = text[i:j+1]
        out = json.loads(text)
        usage = data.get("usage", {})
        if _backend() == "ollama":
            cost = 0.0
        else:
            cost = (int(usage.get("prompt_tokens", 0)) * PRICE_INPUT
                    + int(usage.get("completion_tokens", 0)) * PRICE_OUTPUT)
        bi = out.get("boundary_index")
        if isinstance(bi, str):
            try: bi = int(bi)
            except ValueError: bi = None
        return {"boundary_index": bi,
                "reasoning": str(out.get("reasoning", "")),
                "cost_usd": cost}
    except Exception as exc:
        return {"boundary_index": None, "reasoning": f"error: {exc}", "cost_usd": 0.0}


def combine_with_lower_layers(regex_decision: str, regex_confidence: float,
                              drift_score: float | None,
                              llm: LLMVerdict) -> tuple[str, float, str]:
    """L3 dominates if available + high confidence. Otherwise blend L1+L2."""
    if not llm.available or llm.error:
        # Fall back to lower layers (handled by caller via embeddings_detector)
        return (regex_decision, regex_confidence,
                f"L1+L2 (L3 unavailable: {llm.error or 'no API key'})")

    # L3 is trusted if confidence >= 0.7
    if llm.confidence >= 0.7:
        return (llm.decision, llm.confidence,
                f"L3 ({llm.reasoning})")

    # L3 uncertain — average with L1 if they agree
    if llm.decision == regex_decision:
        avg_conf = (llm.confidence + regex_confidence) / 2.0
        return (llm.decision, min(0.95, avg_conf),
                f"L1+L3 agree: {llm.reasoning}")

    # Disagree — pick higher confidence, mark uncertain
    if llm.confidence > regex_confidence:
        return (llm.decision, llm.confidence * 0.85,
                f"L3 (low conf) vs L1: {llm.reasoning}")
    return (regex_decision, regex_confidence * 0.85,
            f"L1 vs L3 (low conf {llm.confidence:.2f}): {llm.reasoning}")
