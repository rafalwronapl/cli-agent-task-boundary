"""Sanity tests for the L1 regex classifier + tag stripping.

Run: python -m pytest tests/ -v
or:  python tests/test_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import (
    UserPrompt, classify_boundary, extract_user_prompts, _strip_injected_tags,
)


def _prompts(*texts: str) -> list[UserPrompt]:
    return [UserPrompt(text=t, tokens=len(t) // 4) for t in texts]


# ---- new_task signals -------------------------------------------------------

def test_pl_explicit_topic_switch():
    p = _prompts(
        "fix this function please",
        "okay teraz zmieńmy temat, powiedz mi coś o sobie",
    )
    r = classify_boundary(p)
    assert r.decision == "new_task", r


def test_pl_olejmy_to_idiom():
    p = _prompts(
        "let me debug the parser first",
        "dobra olejmy to, zmienmy temat",
    )
    r = classify_boundary(p)
    assert r.decision == "new_task", r


def test_pl_powiedz_cos_o():
    p = _prompts(
        "the function still throws on edge case",
        "powiedz cos o mnie",
    )
    r = classify_boundary(p)
    assert r.decision == "new_task", r


def test_en_moving_on():
    p = _prompts(
        "fix the bug",
        "okay, moving on — tell me about Python decorators",
    )
    r = classify_boundary(p)
    assert r.decision == "new_task", r


def test_en_switching_gears():
    p = _prompts(
        "the test passes now",
        "switching gears, let's talk about database design",
    )
    r = classify_boundary(p)
    assert r.decision == "new_task", r


# ---- continuation signals ---------------------------------------------------

def test_continuation_pl_popraw():
    p = _prompts(
        "napisz parser dla CSV",
        "popraw obsługę cudzysłowów",
        "i jeszcze dodaj wsparcie dla TSV",
    )
    r = classify_boundary(p)
    assert r.decision == "continuation", r


def test_continuation_en_fix():
    p = _prompts(
        "write a sorting function",
        "fix the edge case with empty arrays",
        "also handle negative numbers",
    )
    r = classify_boundary(p)
    assert r.decision == "continuation", r


# ---- task_complete signals --------------------------------------------------

def test_task_complete_pl():
    p = _prompts(
        "zoptymalizuj zapytanie SQL",
        "ok wystarczy",
        "to wszystko, dzięki",
    )
    r = classify_boundary(p)
    assert r.decision in ("task_complete", "unclear"), r


# ---- injected tag stripping -------------------------------------------------

def test_strip_system_reminder():
    raw = ("popraw <system-reminder>\nLong injected note about coding style.\n"
           "</system-reminder> ten bug")
    cleaned = _strip_injected_tags(raw)
    assert "system-reminder" not in cleaned
    assert "popraw" in cleaned
    assert "ten bug" in cleaned


def test_strip_command_blocks():
    raw = "<command-name>/foo</command-name>\n<command-stdout>x</command-stdout>\nreal user text"
    cleaned = _strip_injected_tags(raw)
    assert "command-name" not in cleaned
    assert "command-stdout" not in cleaned
    assert "real user text" in cleaned


def test_extract_user_prompts_strips_tags():
    events = [{
        "type": "user",
        "message": {
            "role": "user",
            "content": ("<system-reminder>ignore this</system-reminder>"
                        " zmieńmy temat"),
        },
    }]
    prompts = extract_user_prompts(events)
    assert len(prompts) == 1
    assert "system-reminder" not in prompts[0].text
    assert "zmieńmy temat" in prompts[0].text


def test_extract_user_prompts_skips_tool_result():
    events = [{
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": "ls output"}],
        },
    }, {
        "type": "user",
        "message": {
            "role": "user",
            "content": "real prompt",
        },
    }]
    prompts = extract_user_prompts(events)
    assert len(prompts) == 1
    assert prompts[0].text == "real prompt"


# ---- empty / edge cases -----------------------------------------------------

def test_empty_prompts():
    r = classify_boundary([])
    assert r.decision == "unclear"


def test_neutral_conversation():
    p = _prompts("hello", "what is 2+2", "thanks for the help with that one")
    r = classify_boundary(p)
    # Last prompt has "thanks" which is task_complete signal, accept that.
    assert r.decision in ("unclear", "continuation", "task_complete"), r


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
        except Exception:
            failed += 1
            print(f"  ERR   {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
