"""Token economy — precise counts from Claude Code JSONL, burn rate, ETA.

Claude Code stores exact usage per assistant turn:
- input_tokens (fresh)
- cache_creation_input_tokens (~25% cheaper)
- cache_read_input_tokens (~10x cheaper)
- output_tokens

Context window limits:
- Claude Sonnet/Opus: 200k tokens (auto-compact at ~187k)
- Codex: depends on model, estimate from text

Pricing (Anthropic May 2026, approx, $/MTok):
- Sonnet 4: $3 input / $15 output / $0.30 cache_read
- Opus 4: $15 input / $75 output / $1.50 cache_read
- We default to Sonnet pricing (most common).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

CLAUDE_CONTEXT_WINDOW_STANDARD = 200_000
CLAUDE_CONTEXT_WINDOW_EXTENDED = 1_000_000  # Opus 4.x / Sonnet 4.x 1M tier
CLAUDE_AUTO_COMPACT_PCT = 0.95               # observed Claude Code trigger (~95%)
CODEX_CONTEXT_WINDOW = 200_000

# Per-family pricing ($/MTok). Picked per assistant turn from message.model.
PRICING = {
    "opus": {
        "input":        15.00 / 1_000_000,
        "output":       75.00 / 1_000_000,
        "cache_create": 18.75 / 1_000_000,
        "cache_read":    1.50 / 1_000_000,
    },
    "sonnet": {
        "input":         3.00 / 1_000_000,
        "output":       15.00 / 1_000_000,
        "cache_create":  3.75 / 1_000_000,
        "cache_read":    0.30 / 1_000_000,
    },
    "haiku": {
        "input":         0.80 / 1_000_000,
        "output":        4.00 / 1_000_000,
        "cache_create":  1.00 / 1_000_000,
        "cache_read":    0.08 / 1_000_000,
    },
}
DEFAULT_PRICING_FAMILY = "sonnet"


def _family_from_model(model_id: str) -> str:
    """Map a Claude model ID (e.g. 'claude-opus-4-7-20260515') to a pricing family."""
    if not model_id:
        return DEFAULT_PRICING_FAMILY
    m = model_id.lower()
    if "opus" in m:
        return "opus"
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    return DEFAULT_PRICING_FAMILY


# Backwards-compat alias (kept so external scripts using SONNET_PRICING still work)
SONNET_PRICING = PRICING["sonnet"]


@dataclass
class TokenStats:
    source: str
    n_turns: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    first_turn_ts: float | None = None
    last_turn_ts: float | None = None
    context_window: int = CLAUDE_CONTEXT_WINDOW_STANDARD
    is_exact: bool = True               # False if we estimated
    # Cost accumulator — summed per-turn using the model active at that turn,
    # so Opus/Sonnet/Haiku mixes within one session are priced correctly.
    cost_accumulated_usd: float = 0.0
    # Last seen model id (informational — shown in detail panel)
    model: str = ""

    @property
    def total_billed(self) -> int:
        """Tokens that actually cost money — fresh+create+read+output."""
        return (self.input_tokens + self.cache_creation_tokens
                + self.cache_read_tokens + self.output_tokens)

    @property
    def latest_context(self) -> int:
        """Bytes actually loaded into the model's context window at the last
        turn. Set explicitly by the JSONL parser (`_latest_context`); callers
        should prefer this over computing from the per-session totals, which
        sum across all turns and don't represent any single point in time.
        """
        return getattr(self, "_latest_context", 0)

    @property
    def context_used_pct(self) -> float:
        return self.latest_context / self.context_window if self.context_window else 0.0

    @property
    def cost_usd(self) -> float:
        # Use per-turn accumulator if available (handles Opus/Sonnet/Haiku mix).
        if self.cost_accumulated_usd > 0:
            return self.cost_accumulated_usd
        # Fallback: price everything against the detected family, or Sonnet.
        p = PRICING.get(_family_from_model(self.model), PRICING[DEFAULT_PRICING_FAMILY])
        return (self.input_tokens * p["input"]
                + self.cache_creation_tokens * p["cache_create"]
                + self.cache_read_tokens * p["cache_read"]
                + self.output_tokens * p["output"])

    @property
    def duration_sec(self) -> float:
        if self.first_turn_ts is None or self.last_turn_ts is None:
            return 0.0
        return max(0.0, self.last_turn_ts - self.first_turn_ts)

    @property
    def burn_rate_per_min(self) -> float | None:
        """Tokens billed per minute over the session."""
        dur = self.duration_sec
        if dur < 60:
            return None
        return self.total_billed / (dur / 60.0)

    def eta_to_autocompact(self, latest_context: int) -> str:
        """Estimate minutes to autocompact.

        Uses the empirical context-growth rate from THIS session instead of a
        magic fraction of burn_rate. Real context grows by input_tokens +
        cache_creation_tokens per turn (cache_read is reused, doesn't grow).
        """
        if self.duration_sec < 60:
            return "n/a (za krótka sesja)"
        trigger = int(self.context_window * CLAUDE_AUTO_COMPACT_PCT)
        remaining = trigger - latest_context
        if remaining <= 0:
            return f"BLISKO/PRZEKROCZONO (próg {trigger:,})"
        # New tokens that actually ADD to the context window this session
        new_context_tokens = self.input_tokens + self.cache_creation_tokens
        growth_per_min = new_context_tokens / (self.duration_sec / 60.0)
        if growth_per_min <= 0:
            return "n/a"
        min_to_trigger = remaining / growth_per_min
        if min_to_trigger < 5:
            return f"~{min_to_trigger:.1f} min (UWAGA)"
        if min_to_trigger < 60:
            return f"~{min_to_trigger:.0f} min"
        return f"~{min_to_trigger/60:.1f} h"


def _parse_ts(ts_str: str) -> float | None:
    """Parse Claude Code timestamp (ISO 8601) to epoch seconds."""
    if not ts_str:
        return None
    try:
        # Format: "2026-05-22T10:30:00.000Z"
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str).timestamp()
    except (ValueError, TypeError):
        return None


# Cache for parsed JSONL token stats — keyed by (path_str, mtime). Big sessions
# can be 5-50 MB, re-parsing every refresh is a major source of GUI lag.
_TOKEN_STATS_CACHE: dict[tuple[str, float], "TokenStats"] = {}


def claude_code_token_stats(jsonl_path: Path) -> TokenStats:
    """Parse exact token usage from Claude Code JSONL session file.

    Context size estimate: takes the MAX single-iteration input from the LAST
    assistant turn. Within one turn there can be multiple iterations (tool
    calls), each is a separate API call so we want the largest single call as
    the "current context window load."
    """
    if not jsonl_path.exists():
        return TokenStats(source="claude_code")

    cache_key = (str(jsonl_path), jsonl_path.stat().st_mtime)
    cached = _TOKEN_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Cap cache size to avoid memory growth
    if len(_TOKEN_STATS_CACHE) > 32:
        _TOKEN_STATS_CACHE.clear()

    stats = TokenStats(source="claude_code")
    last_turn_context = 0

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            if d.get("type") != "assistant":
                continue

            ts = _parse_ts(d.get("timestamp", ""))
            if ts is not None:
                if stats.first_turn_ts is None:
                    stats.first_turn_ts = ts
                stats.last_turn_ts = ts

            msg = d.get("message", {})
            usage = msg.get("usage", {})
            if not usage:
                continue

            stats.n_turns += 1
            in_tok = int(usage.get("input_tokens", 0))
            cc_tok = int(usage.get("cache_creation_input_tokens", 0))
            cr_tok = int(usage.get("cache_read_input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            stats.input_tokens += in_tok
            stats.cache_creation_tokens += cc_tok
            stats.cache_read_tokens += cr_tok
            stats.output_tokens += out_tok

            # Price this turn against its own model (handles Opus/Sonnet/Haiku swap)
            model_id = msg.get("model", "") or ""
            if model_id:
                stats.model = model_id
            p = PRICING.get(_family_from_model(model_id), PRICING[DEFAULT_PRICING_FAMILY])
            stats.cost_accumulated_usd += (
                in_tok * p["input"]
                + cc_tok * p["cache_create"]
                + cr_tok * p["cache_read"]
                + out_tok * p["output"]
            )

            # Context size = MAX single iteration input within this turn
            # (iterations within one turn = parallel tool calls etc.)
            iters = usage.get("iterations", [])
            if iters:
                # Each iteration is a separate API call, take MAX as context indicator
                max_iter_ctx = max(
                    (int(it.get("input_tokens", 0))
                     + int(it.get("cache_creation_input_tokens", 0))
                     + int(it.get("cache_read_input_tokens", 0)))
                    for it in iters
                )
                last_turn_context = max_iter_ctx
            else:
                last_turn_context = (
                    int(usage.get("input_tokens", 0))
                    + int(usage.get("cache_creation_input_tokens", 0))
                    + int(usage.get("cache_read_input_tokens", 0))
                )

    stats._latest_context = last_turn_context
    # Auto-detect extended context (1M tier) — if observed context exceeds standard
    if last_turn_context > CLAUDE_CONTEXT_WINDOW_STANDARD * 0.95:
        stats.context_window = CLAUDE_CONTEXT_WINDOW_EXTENDED
    _TOKEN_STATS_CACHE[cache_key] = stats
    return stats


def codex_token_stats(prompts: list) -> TokenStats:
    """Estimate token usage for Codex (no usage field — estimate from text)."""
    stats = TokenStats(source="codex", is_exact=False,
                        context_window=CODEX_CONTEXT_WINDOW)
    if not prompts:
        return stats

    # rough estimate: 4 chars = 1 token
    input_est = sum(len(p.text) // 4 for p in prompts)
    # Assume output ~= input on average (rough)
    stats.input_tokens = input_est
    stats.output_tokens = int(input_est * 1.5)  # heuristic
    stats.n_turns = len(prompts)

    # Timestamps from prompts (Codex stores ts as unix epoch)
    timestamps = []
    for p in prompts:
        try:
            timestamps.append(float(p.timestamp))
        except (ValueError, TypeError):
            continue
    if timestamps:
        stats.first_turn_ts = min(timestamps)
        stats.last_turn_ts = max(timestamps)

    stats._latest_context = input_est  # whole history = context
    return stats


def get_token_stats(source: str, session_meta: dict, prompts: list | None = None) -> TokenStats:
    """Dispatch to right parser per source."""
    if source == "claude_code":
        path = Path(session_meta["path"])
        return claude_code_token_stats(path)
    if source == "codex":
        return codex_token_stats(prompts or [])
    # Generic estimate for sources that don't store exact usage (gemini, etc.)
    if prompts is None:
        prompts = []
    return codex_token_stats(prompts)


def latest_context_estimate(stats: TokenStats) -> int:
    """Return what's likely IN the model's current context window."""
    return getattr(stats, "_latest_context", 0)
