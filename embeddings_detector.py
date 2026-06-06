"""Layer 2 — topic drift detection via sentence embeddings.

Lazy import sentence-transformers: jeśli nie zainstalowany, Layer 1 wciąż działa.
Model: all-MiniLM-L6-v2 (80MB, CPU-friendly, ~5ms per embedding).

Strategia:
- Embed każdą user-prompt
- Dla ostatnich 2N promptów: embed [first half] vs [second half]
- Cosine similarity między średnimi embeddings
- Niska similarity → topic drift → task boundary signal
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Lazy state
_MODEL = None
_AVAILABLE: bool | None = None
_EMB_CACHE: dict[str, "any"] = {}    # text → numpy embedding


def is_available() -> bool:
    """Return True if sentence-transformers is installed."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        import sentence_transformers  # noqa: F401
        import numpy  # noqa: F401
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    cache_dir = Path.home() / ".cache" / "task-boundary-detector"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2",
                                  cache_folder=str(cache_dir))
    return _MODEL


@dataclass
class DriftResult:
    available: bool
    similarity: float | None
    drift_score: float | None    # 1 - similarity, 0 = no drift, 1 = max
    window_size: int
    n_compared: int
    notes: str = ""


def compute_topic_drift(prompts: list, window: int = 5,
                        min_prompts: int = 3) -> DriftResult:
    """Compute semantic drift — TWO strategies combined:

    Strategy A: LAST PROMPT vs MEAN OF PREVIOUS N
      → catches single-prompt topic switches ("zmienmy temat" → new topic)
    Strategy B: MEAN OF RECENT N vs MEAN OF PREVIOUS N
      → catches gradual topic drift over multiple prompts

    Returns DriftResult with the MAX drift_score from both strategies.
    drift_score > 0.40 = significant shift.
    """
    if not is_available():
        return DriftResult(available=False, similarity=None,
                            drift_score=None, window_size=window,
                            n_compared=0,
                            notes="sentence-transformers nie zainstalowane")

    if len(prompts) < min_prompts:
        return DriftResult(available=True, similarity=None,
                            drift_score=None, window_size=window,
                            n_compared=len(prompts),
                            notes=f"Za mało promptów ({len(prompts)} < {min_prompts})")

    import numpy as np

    # Adapt window
    eff_window = min(window, max(1, (len(prompts) - 1)))

    last_text = prompts[-1].text
    # Previous N (excluding last)
    prev_texts = [p.text for p in prompts[-(eff_window+1):-1]]
    if not prev_texts:
        return DriftResult(available=True, similarity=None, drift_score=None,
                            window_size=eff_window, n_compared=len(prompts),
                            notes="Brak previous block")

    try:
        model = _get_model()
        all_texts = [last_text] + prev_texts
        # Cache hits first
        to_encode = [t for t in all_texts if t not in _EMB_CACHE]
        if to_encode:
            new_embs = model.encode(to_encode, convert_to_numpy=True,
                                      show_progress_bar=False)
            for t, e in zip(to_encode, new_embs):
                _EMB_CACHE[t] = e
            # Cap cache (FIFO)
            if len(_EMB_CACHE) > 2000:
                keys_to_drop = list(_EMB_CACHE.keys())[:500]
                for k in keys_to_drop:
                    _EMB_CACHE.pop(k, None)
        embs = np.array([_EMB_CACHE[t] for t in all_texts])
    except Exception as exc:
        return DriftResult(available=True, similarity=None, drift_score=None,
                            window_size=eff_window, n_compared=len(prompts),
                            notes=f"Embedding error: {exc}")

    last_emb = embs[0]
    prev_embs = embs[1:]
    prev_mean = prev_embs.mean(axis=0)

    def _cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Strategy A: last vs mean of previous
    sim_a = _cos(last_emb, prev_mean)
    drift_a = 1.0 - sim_a

    # Strategy B: mean of last N (if we have it) vs mean of previous N
    drift_b = drift_a  # fallback
    sim_b = sim_a
    if len(prompts) >= 2 * eff_window:
        recent_texts2 = [p.text for p in prompts[-eff_window:]]
        prev_texts2 = [p.text for p in prompts[-2*eff_window:-eff_window]]
        if prev_texts2:
            all2 = recent_texts2 + prev_texts2
            to_encode2 = [t for t in all2 if t not in _EMB_CACHE]
            if to_encode2:
                new_embs2 = model.encode(to_encode2, convert_to_numpy=True,
                                           show_progress_bar=False)
                for t, e in zip(to_encode2, new_embs2):
                    _EMB_CACHE[t] = e
            embs2 = np.array([_EMB_CACHE[t] for t in all2])
            recent_mean = embs2[:len(recent_texts2)].mean(axis=0)
            prev_mean2 = embs2[len(recent_texts2):].mean(axis=0)
            sim_b = _cos(recent_mean, prev_mean2)
            drift_b = 1.0 - sim_b

    # MAX of both — sensitive to sudden shifts AND gradual drift
    if drift_a >= drift_b:
        final_drift = drift_a
        final_sim = sim_a
        note = f"OK (last-vs-prev: {drift_a:.2f})"
    else:
        final_drift = drift_b
        final_sim = sim_b
        note = f"OK (gradual: {drift_b:.2f}, last-vs-prev: {drift_a:.2f})"

    return DriftResult(available=True, similarity=final_sim,
                        drift_score=final_drift,
                        window_size=eff_window,
                        n_compared=1 + len(prev_texts),
                        notes=note)


def combine_with_regex(regex_decision: str, regex_confidence: float,
                       drift: DriftResult,
                       drift_threshold: float = 0.40) -> tuple[str, float, str]:
    """Combine Layer 1 (regex) with Layer 2 (embeddings).

    Returns (final_decision, final_confidence, explanation).
    """
    if not drift.available or drift.drift_score is None:
        return (regex_decision, regex_confidence,
                f"L1 only ({drift.notes})")

    high_drift = drift.drift_score > drift_threshold
    low_drift = drift.drift_score < 0.15  # very similar = continuation

    # Override rules
    if high_drift and regex_decision in ("unclear", "continuation"):
        # Embeddings vote for new task even if regex was unclear/cont
        new_conf = max(regex_confidence, 0.5 + 0.3 * drift.drift_score)
        return ("new_task", min(0.95, new_conf),
                f"L1={regex_decision} + L2 drift={drift.drift_score:.2f} → upgrade to new_task")

    if low_drift and regex_decision == "new_task":
        # Embeddings disagree — say continuation. Lower confidence.
        return ("unclear", 0.4,
                f"L1=new_task but L2 sim={drift.similarity:.2f} (continuation-like) → uncertain")

    # Otherwise: regex decision stays, but bump confidence
    if regex_decision == "continuation" and low_drift:
        return (regex_decision, min(0.95, regex_confidence + 0.1),
                f"L1+L2 agree: continuation (sim={drift.similarity:.2f})")

    if regex_decision == "new_task" and high_drift:
        return (regex_decision, min(0.95, regex_confidence + 0.15),
                f"L1+L2 agree: new_task (drift={drift.drift_score:.2f})")

    return (regex_decision, regex_confidence,
            f"L1={regex_decision} + L2 drift={drift.drift_score:.2f}")
