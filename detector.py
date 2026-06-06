"""Task Boundary Detector — heurystyczna warstwa 1.

Analizuje sesję Claude Code (JSONL) i wykrywa task boundaries.
Bez zewnętrznych API. Wszystko lokalnie.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__version__ = "0.3.0"


# ---- regex signals -----------------------------------------------------------

NEW_TASK_SIGNALS = [
    # === PL: eksplicytne przejście (zmiana tematu) ===
    r"\bzmień(?:my|cie)?\s+(?:temat|tematu|projekt)",       # zmieńmy temat
    r"\bzmie[nń](?:my|cie)?\s+(?:temat|tematu|projekt)",    # zmienmy temat (bez ogonka)
    r"\bzmiana\s+(?:tematu|projektu)\b",
    r"\bolejmy\s+(?:to|tamto|tamten|tamtą)",                # olejmy to (idiom)
    r"\bzapomnijmy\s+(?:o|to|tamto)",
    r"\b(?:przejd[zź](?:my|cie)?|przechodzimy)\s+(?:do|na)\b",
    # "teraz" as a topic-switch marker — only paired with verbs that imply a
    # new subject, NOT with debug-flow verbs like "popraw" / "sprawdź" / "zrób"
    # which are typical mid-task continuations.
    r"\bteraz\s+(?:porozmawiaj|pogadaj|opowied[zż]|powied[zż]|przejd[zź])",
    r"\bokay\s*[,.]?\s*(?:teraz|to)\b",
    r"\bok\s*[,.]?\s*(?:zmie|teraz|to\s+inne|inne)",
    r"\bco\s+innego\b",
    r"\b(?:inn(?:y|a|ą|e)\s+(?:temat|sprawa|sprawę|rzecz|pytanie))\b",
    r"\b(?:nowy|nowa|nowe)\s+(?:temat|projekt|task|zadanie|pytanie)\b",
    # === PL: pytania "powiedz o" / "co o" — meta-talk shift ===
    r"\b(?:powied[zż]|opowied[zż]|opowiedz)\s+(?:cos|coś|mi|nam)\s+(?:o|na\s+temat)",
    r"\bco\s+(?:powiesz|s[ąa]dzisz|wiesz)\s+(?:o|na\s+temat)",
    r"\bporozmawiaj(?:my|cie)?\s+(?:o|na\s+temat)",
    r"\bpogadaj(?:my|cie)?\s+(?:o|na\s+temat)",
    r"\bp(?:y|i)tanie\s+do\s+ciebie",
    # === PL: reset intent ===
    r"\bzapomnij\b.*\b(?:poprzedn|wcze[sś]niej|wszystk)",
    r"\bignor(?:uj|owanie)\b.*\b(?:poprzedn|wcze[sś]niej)",
    r"\bzaczynamy\s+(?:od|nowy)",
    r"\bod\s+nowa\b",
    # === EN: explicit transition ===
    r"\bchange\s+(?:topic|subject|projects?)\b",
    r"\b(?:different|new|another)\s+(?:topic|subject|thing|task|project)\b",
    r"\bsomething\s+(?:else|different)\b",
    r"\bnext\s+(?:task|topic|thing|question)\b",
    r"\bnow\s+let'?s\b",
    r"\bokay\s+now\s+let",
    r"\bmoving\s+on\b",
    r"\bswitching\s+(?:to|topics?|gears)\b",
    r"\blet'?s\s+(?:talk|discuss)\s+about\b",
    r"\btell\s+me\s+(?:about|something\s+about)",
    r"\bwhat\s+do\s+you\s+(?:think|know)\s+about\b",
    # === EN: reset intent ===
    r"\bforget\b.*\b(?:about|previous|earlier|prior)",
    r"\bignore\b.*\b(?:previous|earlier|prior|above)",
    r"\bstart\s+(?:over|fresh|again)\b",
    r"\b(?:reset|wipe)\s+(?:conversation|context)",
]

TASK_COMPLETE_SIGNALS = [
    # acknowledgment without follow-up
    r"^\s*(?:thanks|thank\s+you|thx)\b",
    r"^\s*(?:perfect|great|awesome|excellent)\s*[.!]?$",
    r"^\s*(?:działa|gotowe|super|świetnie|dobra)\s*[.!]?$",
    r"^\s*(?:done|finished|complete)\s*[.!]?$",
    r"^\s*(?:ok|okay|okej)\s*[.!]?$",
    # nothing more wanted
    r"\bto\s+wszystko\b",
    r"\bna\s+(?:dziś|teraz)\s+(?:tyle|koniec)\b",
    r"\bthat'?s\s+(?:all|it)\b",
]

CONTINUATION_SIGNALS = [
    # bezpośrednie referencje
    r"\b(?:ta|ten|tej|tym|tego)\s+(?:funkc|plik|klas|metod|skrypt|kod)",
    r"\bthe\s+(?:function|file|class|method|script|code|same)\b",
    r"\b(?:jeszcze|także|również|i|plus)\b",
    r"\b(?:still|also|and|plus)\s+(?:add|fix|check|test)",
    # poprawki / ulepszenia
    r"\b(?:popraw|zmień|dodaj|usuń)\b",
    r"\b(?:fix|change|add|remove|update)\b",
    # zaimki wskazujące
    r"^(?:to|this|that)\b",
    # debug flow
    r"\b(?:błąd|error|wyjątek|exception|trace)",
]


@dataclass
class UserPrompt:
    text: str
    timestamp: str = ""
    tokens: int = 0


@dataclass
class BoundarySignal:
    decision: str          # "new_task" | "continuation" | "task_complete" | "unclear"
    confidence: float      # 0.0-1.0
    reasoning: str
    new_task_hits: int = 0
    complete_hits: int = 0
    continuation_hits: int = 0
    total_prompts_analyzed: int = 0
    context_tokens_estimate: int = 0
    recommendation: str = ""
    matched_signals: list[str] = field(default_factory=list)


# ---- session parsing ---------------------------------------------------------

def read_session(path: Path) -> list[dict]:
    """Read JSONL session file, returns raw events."""
    if not path.exists():
        raise FileNotFoundError(path)
    events = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# Tags injected by Claude Code into user-role messages that aren't real user text.
# Stripping them prevents regex/embedding/LLM from misreading injected context as
# actual user intent (e.g. a system-reminder mentioning "popraw" → false continuation hit).
_INJECTED_TAG_PATTERNS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<command-name>.*?</command-name>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<command-message>.*?</command-message>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<command-args>.*?</command-args>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<command-stdout>.*?</command-stdout>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<command-stderr>.*?</command-stderr>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<local-command-stderr>.*?</local-command-stderr>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<bash-input>.*?</bash-input>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<bash-stdout>.*?</bash-stdout>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<bash-stderr>.*?</bash-stderr>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<user-prompt-submit-hook>.*?</user-prompt-submit-hook>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<ide_(?:opened_file|selection)>.*?</ide_(?:opened_file|selection)>", re.DOTALL | re.IGNORECASE),
]


def _strip_injected_tags(text: str) -> str:
    for pat in _INJECTED_TAG_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def extract_user_prompts(events: list[dict], limit: int | None = None) -> list[UserPrompt]:
    """Pull only top-level user prompts (skip tool_result blocks).

    Claude Code distinguishes 'real' user input from tool_result via content shape:
    real user text = string OR list with text-only blocks.
    Tool results = list with type='tool_result' blocks.

    Also strips injected XML-ish tags (system-reminder, command-*, local-command-*,
    user-prompt-submit-hook, bash-*) so detection layers see only actual user text.
    """
    prompts = []
    for ev in events:
        if ev.get("type") != "user":
            continue
        msg = ev.get("message") or {}
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # skip if contains tool_result (this is hook response, not user input)
            if any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content):
                continue
            text = " ".join(c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "text")
        text = _strip_injected_tags(text)
        if not text.strip():
            continue
        prompts.append(UserPrompt(
            text=text.strip(),
            timestamp=ev.get("timestamp", ""),
            tokens=len(text) // 4,  # rough estimate
        ))
    if limit:
        prompts = prompts[-limit:]
    return prompts


# ---- classification ----------------------------------------------------------

def _count_matches(text: str, patterns: list[str]) -> tuple[int, list[str]]:
    matches = []
    for p in patterns:
        if re.search(p, text, re.IGNORECASE | re.MULTILINE):
            matches.append(p)
    return len(matches), matches


def find_last_new_task_prompt(prompts: list[UserPrompt],
                              lookback: int = 50) -> int | None:
    """Walk prompts last-to-first (capped at `lookback`), return index of the
    most recent prompt that contains any NEW_TASK_SIGNALS regex.

    The cap matters: scanning 300+ prompts × 30 regex patterns when there's
    no match is a noticeable GUI hitch. A boundary further back than `lookback`
    is no longer "current" anyway — the user has moved on.
    """
    if not prompts:
        return None
    n = len(prompts)
    start = max(0, n - lookback)
    for i in range(n - 1, start - 1, -1):
        hits, _ = _count_matches(prompts[i].text, NEW_TASK_SIGNALS)
        if hits:
            return i
    return None


def classify_boundary(prompts: list[UserPrompt], lookback: int = 5) -> BoundarySignal:
    """Apply heuristic rules to the most recent N user prompts."""
    if not prompts:
        return BoundarySignal(
            decision="unclear", confidence=0.0,
            reasoning="No user prompts found.",
            recommendation="(brak danych)",
        )

    recent = prompts[-lookback:]
    full_text = "\n".join(p.text for p in recent)

    new_n, new_match = _count_matches(full_text, NEW_TASK_SIGNALS)
    complete_n, complete_match = _count_matches(full_text, TASK_COMPLETE_SIGNALS)
    cont_n, cont_match = _count_matches(full_text, CONTINUATION_SIGNALS)

    # check the very latest prompt separately — strongest signal
    latest = recent[-1].text
    latest_new = _count_matches(latest, NEW_TASK_SIGNALS)[0]
    latest_complete = _count_matches(latest, TASK_COMPLETE_SIGNALS)[0]
    latest_cont = _count_matches(latest, CONTINUATION_SIGNALS)[0]

    # double weight on most recent prompt
    new_score = new_n + 2 * latest_new
    complete_score = complete_n + 2 * latest_complete
    cont_score = cont_n + 2 * latest_cont

    total_score = new_score + complete_score + cont_score
    if total_score == 0:
        return BoundarySignal(
            decision="unclear", confidence=0.3,
            reasoning="No strong signals — neutral conversation.",
            new_task_hits=0, complete_hits=0, continuation_hits=0,
            total_prompts_analyzed=len(recent),
            recommendation="Brak akcji wymagany — kontynuuj.",
        )

    # decision tree
    if new_score >= 2 and new_score > cont_score:
        decision = "new_task"
        confidence = min(0.95, 0.4 + 0.2 * new_score)
        rec = (f"⚡ TASK BOUNDARY — wykryto sygnały nowego tasku ({new_score} matches). "
               f"Rozważ /compact lub nową sesję — bieżąca historia może być balastem.")
    elif complete_score >= 2 and complete_score > cont_score:
        decision = "task_complete"
        confidence = min(0.9, 0.3 + 0.2 * complete_score)
        rec = (f"⚠️ TASK COMPLETE — poprzedni task wygląda na zakończony ({complete_score} matches). "
               f"Jeśli zaczynasz nowy task → /compact lub nowa sesja.")
    elif cont_score > new_score + complete_score:
        decision = "continuation"
        confidence = min(0.85, 0.3 + 0.15 * cont_score)
        rec = f"✅ KONTYNUACJA — historia relewantna ({cont_score} matches). Brak akcji wymagany."
    else:
        decision = "unclear"
        confidence = 0.3
        rec = ("Mieszane sygnały — sprawdź ręcznie. "
               f"new={new_score} complete={complete_score} cont={cont_score}.")

    reasoning_parts = []
    if new_match:
        reasoning_parts.append(f"new_task signals: {len(new_match)}")
    if complete_match:
        reasoning_parts.append(f"complete signals: {len(complete_match)}")
    if cont_match:
        reasoning_parts.append(f"continuation signals: {len(cont_match)}")
    reasoning = " | ".join(reasoning_parts) or "neutral"

    return BoundarySignal(
        decision=decision,
        confidence=confidence,
        reasoning=reasoning,
        new_task_hits=new_score,
        complete_hits=complete_score,
        continuation_hits=cont_score,
        total_prompts_analyzed=len(recent),
        context_tokens_estimate=sum(p.tokens for p in prompts),
        recommendation=rec,
        matched_signals=(new_match + complete_match + cont_match)[:10],
    )


# ---- context value score -----------------------------------------------------

def context_value_score(prompts: list[UserPrompt]) -> float:
    """0.0-1.0 — jak 'cenna' jest obecna historia."""
    if not prompts:
        return 1.0
    if len(prompts) < 3:
        return 1.0

    # Recency weight: ostatnie 30% wiadomości
    recent_cutoff = max(1, int(len(prompts) * 0.7))
    recent_tokens = sum(p.tokens for p in prompts[recent_cutoff:])
    total_tokens = sum(p.tokens for p in prompts)
    recency_weight = recent_tokens / total_tokens if total_tokens else 0.0

    # Coherence: czy ostatnie wiadomości mają continuation signals
    last_5 = "\n".join(p.text for p in prompts[-5:])
    cont_n = _count_matches(last_5, CONTINUATION_SIGNALS)[0]
    new_n = _count_matches(last_5, NEW_TASK_SIGNALS)[0]
    # high cont, low new → high coherence
    if cont_n + new_n == 0:
        coherence = 0.5
    else:
        coherence = cont_n / (cont_n + new_n)

    return min(1.0, recency_weight * 0.4 + coherence * 0.6)


# ---- session discovery -------------------------------------------------------

DEFAULT_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def find_latest_session(projects_dir: Path = DEFAULT_CLAUDE_PROJECTS) -> Path | None:
    """Find the most recently modified JSONL session file."""
    if not projects_dir.exists():
        return None
    candidates = list(projects_dir.rglob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
