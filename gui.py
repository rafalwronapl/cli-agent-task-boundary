"""Task Boundary Detector — GUI v2.

Wyświetla LISTĘ wszystkich aktywnych sesji (claude_code + codex + continue),
każda z własnym boundary verdict. Source filter zawęża widok.

Layer 1 (regex) zawsze działa. Layer 2 (embeddings) jeśli sentence-transformers
zainstalowane — daje dokładniejsze topic drift detection.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import (
    ADAPTERS, list_sessions, get_prompts_for_session, get_last_ai_response,
    normalize_source,
)
from detector import classify_boundary, context_value_score, find_last_new_task_prompt
from embeddings_detector import (
    compute_topic_drift, combine_with_regex, is_available as l2_available,
)
from llm_detector import (
    classify_via_llm, combine_with_lower_layers, find_boundary_index,
    is_available as l3_available, backend_name as l3_backend_name,
)
from settings_dialog import open_settings_dialog
from token_economy import get_token_stats, latest_context_estimate

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

DECISION_COLORS = {
    "new_task":      "#ffd166",
    "task_complete": "#8cc8ff",
    "continuation":  "#7fdb95",
    "unclear":       "#a7adba",
    "fresh":         "#9ddfff",
}
DECISION_ICONS = {
    "new_task":      "⚡",
    "task_complete": "⚠",
    "continuation":  "✓",
    "unclear":       "•",
    "fresh":         "✨",
}
SOURCE_COLORS = {
    "claude_code": "#cb9bff",
    "codex":       "#7fb5ff",
    "gemini":      "#ffd57f",
}


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Task Boundary Detector")
        root.geometry("960x680")
        root.configure(bg="#1a1d23")

        self.watch_stop = threading.Event()
        # L2 + L3 always on by default if available (no Advanced toggle needed)
        self.l2_enabled = tk.IntVar(value=1 if l2_available() else 0)
        self.l3_enabled = tk.IntVar(value=1 if l3_available() else 0)
        # L3 results cached per session_id — fetched async, displayed when ready
        self._l3_results: dict[str, object] = {}
        self._l3_inflight: set[str] = set()
        # Debounce token for batching multiple L3 completions into one redraw
        self._l3_refresh_pending: bool = False
        # Cache boundary-location per (session_key, n_prompts) — invalidates
        # automatically when a new prompt is appended.
        self._boundary_loc_cache: dict[tuple[str, int], int | None] = {}
        self.session_widgets: list[tk.Frame] = []
        self.expanded_session: str | None = None
        self.detail_data: dict[str, dict] = {}
        self._cached_boundary: dict[str, int] = {}  # session_key → boundary_index

        self._build_ui()
        self.root.after(150, self._refresh)
        # Nudge: if Layer 3 is off, surface it so the user knows it's
        # configurable. Status bar message is non-modal — won't block start-up.
        if not l3_available():
            self.root.after(1200, lambda: self._set_status(
                "Layer 3 wyłączone (brak klucza OpenRouter ani Ollamy). "
                "Kliknij ⚙ LLM aby skonfigurować."
            ))

    def _open_llm_settings(self) -> None:
        def _on_saved():
            # Re-evaluate availability and refresh the list. L3 checkbox is
            # rebuilt only on next launch (tkinter limitation), but the layer
            # itself activates immediately.
            self.l3_enabled.set(1 if l3_available() else 0)
            self._set_status(f"LLM backend: {l3_backend_name()}. Odświeżam...")
            self._refresh()
        open_settings_dialog(self.root, on_saved=_on_saved)

    # ----- UI build -----
    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TButton", background="#5fb3f7", foreground="#000",
                        font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TCombobox", fieldbackground="#252932",
                        background="#252932", foreground="#e8e8e8")
        style.configure("TLabel", background="#1a1d23", foreground="#e8e8e8",
                        font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#1a1d23",
                        foreground="#e8e8e8")

        # Toolbar
        top = tk.Frame(self.root, bg="#1a1d23")
        top.pack(fill="x", padx=14, pady=12)

        ttk.Label(top, text="Filter source:").pack(side="left")
        self.source_var = tk.StringVar(value="all")
        self.source_combo = ttk.Combobox(top, textvariable=self.source_var,
                                          values=["all"] + list(ADAPTERS.keys()),
                                          state="readonly", width=14)
        self.source_combo.pack(side="left", padx=(6, 12))
        self.source_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh())

        ttk.Label(top, text="Pokaż:").pack(side="left")
        # Presets — "active" filters to very recent mtime, "all" shows 24h
        self.scope_var = tk.StringVar(value="aktywne (15 min)")
        scope_combo = ttk.Combobox(top, textvariable=self.scope_var,
                                    values=["aktywne (15 min)",
                                             "ostatnia 1h",
                                             "ostatnie 8h",
                                             "ostatnie 24h"],
                                    state="readonly", width=18)
        scope_combo.pack(side="left", padx=(4, 12))
        scope_combo.bind("<<ComboboxSelected>>", lambda _: self._refresh())
        # Kept for backward compat with anything reading it
        self.max_age_var = tk.StringVar(value="0.25")

        ttk.Button(top, text="Refresh", command=self._refresh).pack(side="left")

        self.watch_var = tk.IntVar(value=0)
        ttk.Checkbutton(top, text="Auto-refresh 10s",
                        variable=self.watch_var,
                        command=self._toggle_watch).pack(side="left", padx=(12, 0))

        # Layer toggles in main toolbar — but compact, no descriptions (always default ON)
        if l2_available():
            ttk.Checkbutton(top, text="L2 (embed)", variable=self.l2_enabled,
                            command=self._on_layer_change).pack(side="left", padx=(8, 0))
        if l3_available():
            ttk.Checkbutton(top, text="L3 (LLM)", variable=self.l3_enabled,
                            command=self._on_layer_change).pack(side="left", padx=(8, 0))

        # Settings button — always present so recipient can configure LLM
        ttk.Button(top, text="⚙ LLM",
                    command=self._open_llm_settings).pack(side="right")

        # Summary bar
        self.summary_label = tk.Label(self.root, text="", bg="#1a1d23",
                                       fg="#8a8e97", font=("Segoe UI", 9))
        self.summary_label.pack(fill="x", padx=14, pady=(0, 4))

        # Main split: list (left, ~60%) + detail (right, ~40%)
        body = tk.Frame(self.root, bg="#1a1d23")
        body.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        # Sessions list (scrollable)
        list_frame = tk.Frame(body, bg="#1a1d23")
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Label(list_frame, text="Aktywne sesje",
                  bg="#1a1d23", fg="#5fb3f7",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")

        canvas = tk.Canvas(list_frame, bg="#1a1d23", highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                   command=canvas.yview)
        self.list_inner = tk.Frame(canvas, bg="#1a1d23")
        self.list_inner.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.list_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._canvas = canvas
        # Bind mousewheel scrolling — only when cursor is INSIDE the session list.
        # bind_all matches every widget (incl. the detail Text), so it must be
        # gated by enter/leave events.
        def _on_enter(_e):
            canvas.bind_all("<MouseWheel>",
                            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        def _on_leave(_e):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _on_enter)
        canvas.bind("<Leave>", _on_leave)
        self.list_inner.bind("<Enter>", _on_enter)
        self.list_inner.bind("<Leave>", _on_leave)

        # Detail panel
        detail_frame = tk.Frame(body, bg="#1a1d23", width=350)
        detail_frame.pack(side="right", fill="both")
        detail_frame.pack_propagate(False)
        tk.Label(detail_frame, text="Szczegóły",
                  bg="#1a1d23", fg="#5fb3f7",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        # Action buttons — 2 rows so they always fit even in narrow panel
        action_frame = tk.Frame(detail_frame, bg="#1a1d23")
        action_frame.pack(fill="x", pady=(4, 4))
        self.copy_compact_btn = ttk.Button(action_frame,
                                            text="Copy /compact",
                                            command=self._copy_compact_command,
                                            state="disabled")
        self.copy_compact_btn.pack(side="left", padx=(0, 4), fill="x", expand=True)
        self.copy_summary_btn = ttk.Button(action_frame,
                                            text="Copy summary",
                                            command=self._copy_summary_brief,
                                            state="disabled")
        self.copy_summary_btn.pack(side="left", padx=(0, 0), fill="x", expand=True)

        action_frame2 = tk.Frame(detail_frame, bg="#1a1d23")
        action_frame2.pack(fill="x", pady=(0, 4))
        self.reset_new_btn = ttk.Button(action_frame2,
                                         text="Reset to new task only",
                                         command=self._reset_to_new_task,
                                         state="disabled")
        self.reset_new_btn.pack(fill="x", expand=True)
        self._active_session_key: str | None = None

        self.detail_text = tk.Text(detail_frame, bg="#252932", fg="#e8e8e8",
                                    font=("Consolas", 9), wrap="word",
                                    relief="flat", padx=10, pady=10)
        self.detail_text.pack(fill="both", expand=True, pady=(4, 0))
        self.detail_text.configure(state="disabled")

        # Status
        self.status = tk.Label(self.root, text="",
                                bg="#1a1d23", fg="#5fb3f7",
                                font=("Segoe UI", 9, "italic"),
                                anchor="w")
        self.status.pack(fill="x", padx=14, pady=(0, 8))

    # ----- main flow -----
    def _on_layer_change(self) -> None:
        if self.l2_enabled.get() and not l2_available():
            self._set_status("Layer 2 niedostępne — `pip install sentence-transformers numpy`")
            self.l2_enabled.set(0)
            return
        if self.l3_enabled.get() and not l3_available():
            self._set_status(
                "Layer 3 niedostępne — dodaj OPENROUTER_API_KEY do .env lub uruchom Ollamę"
            )
            self.l3_enabled.set(0)
            return
        self._refresh()

    def _refresh(self) -> None:
        self._set_status("Loading...")
        self.root.update_idletasks()
        src = self.source_var.get()
        scope = getattr(self, "scope_var", None)
        scope_label = scope.get() if scope else "ostatnie 24h"
        # Map preset → hours
        max_age = {
            "aktywne (15 min)": 0.25,
            "ostatnia 1h":      1.0,
            "ostatnie 8h":      8.0,
            "ostatnie 24h":     24.0,
        }.get(scope_label, 24.0)

        if src == "all":
            sources_to_scan = [normalize_source(k) for k in ADAPTERS.keys()]
        else:
            sources_to_scan = [normalize_source(src)]

        all_sessions = []
        for s in sources_to_scan:
            try:
                for meta in list_sessions(s, max_age_hours=max_age):
                    meta["_source"] = s
                    all_sessions.append(meta)
            except Exception:
                continue
        all_sessions.sort(key=lambda x: x["last_activity"], reverse=True)

        # Clear list
        for w in self.session_widgets:
            w.destroy()
        self.session_widgets = []
        self.detail_data = {}

        if not all_sessions:
            empty = tk.Label(self.list_inner, bg="#1a1d23", fg="#a7adba",
                              text=f"Brak aktywnych sesji w ostatnich {max_age:.0f}h.\n"
                                   f"Source filter: {src}",
                              font=("Segoe UI", 11), pady=20)
            empty.pack(fill="x")
            self.session_widgets.append(empty)
            self.summary_label.config(text="0 sesji")
            self._set_status("Brak danych")
            return

        # Analyze each + render row
        per_source_count: dict[str, int] = {}
        decision_counts: dict[str, int] = {}
        for meta in all_sessions:
            self._analyze_and_render_row(meta)
            per_source_count[meta["_source"]] = per_source_count.get(meta["_source"], 0) + 1
            d = meta.get("_decision", "unclear")
            decision_counts[d] = decision_counts.get(d, 0) + 1

        # Update summary
        src_summary = " | ".join(f"{k}: {v}" for k, v in per_source_count.items())
        dec_summary = " | ".join(f"{DECISION_ICONS.get(k,'?')}{k}: {v}"
                                  for k, v in decision_counts.items())
        l2_tag = "L2" if self.l2_enabled.get() and l2_available() else "L1"
        self.summary_label.config(
            text=f"{len(all_sessions)} sesji ({src_summary}) | "
                 f"Decyzje: {dec_summary} | Mode: {l2_tag}"
        )
        self._set_status(f"Sprawdzone: {time.strftime('%H:%M:%S')}")

    def _analyze_and_render_row(self, meta: dict) -> None:
        source = meta["_source"]
        try:
            prompts = get_prompts_for_session(source, meta)
        except Exception:
            prompts = []

        if not prompts:
            return

        session_key_pre = f"{source}::{meta['session_id']}"
        boundary = classify_boundary(prompts, lookback=5)
        cvs = context_value_score(prompts)
        total_tokens = sum(p.tokens for p in prompts)

        # Token economy (precise for claude_code, estimated for others)
        token_stats = None
        try:
            token_stats = get_token_stats(source, meta, prompts)
        except Exception:
            pass

        # Layer 2: combine if enabled
        final_decision = boundary.decision
        final_confidence = boundary.confidence
        explanation = boundary.reasoning
        drift_info = None
        if self.l2_enabled.get() and l2_available():
            drift = compute_topic_drift(prompts, window=5)
            drift_info = drift
            if drift.available and drift.drift_score is not None:
                final_decision, final_confidence, explanation = combine_with_regex(
                    boundary.decision, boundary.confidence, drift)

        # Layer 3: LLM — ASYNC. Use cached result if available; otherwise spawn
        # background thread and render row without L3 now. Row updates when L3 arrives.
        llm_verdict = None
        if self.l3_enabled.get() and l3_available():
            sid = meta["session_id"]
            cached = self._l3_results.get(sid)
            if cached is not None:
                llm_verdict = cached
                if llm_verdict.available and not llm_verdict.error:
                    final_decision, final_confidence, explanation = combine_with_lower_layers(
                        final_decision, final_confidence,
                        drift_info.drift_score if drift_info else None,
                        llm_verdict)
            else:
                # Spawn background L3 call. Cap at 4 concurrent OpenRouter
                # requests — at 20 active sessions we'd otherwise hammer the API
                # and trigger rate limits. Excess sessions just wait for the
                # next refresh tick.
                if sid not in self._l3_inflight and len(self._l3_inflight) < 4:
                    self._l3_inflight.add(sid)
                    threading.Thread(
                        target=self._fetch_l3_background,
                        args=(sid, prompts, session_key_pre),
                        daemon=True,
                    ).start()

        # Fresh-session override: if there are <5 user prompts AND the session
        # is younger than 30 min, this is the START of work, not a topic switch
        # within an existing conversation. Force decision to a neutral label.
        session_age_s = time.time() - meta["last_activity"]
        is_fresh = len(prompts) < 5 and session_age_s < 1800
        if is_fresh:
            final_decision = "fresh"
            final_confidence = max(final_confidence, 0.8)
            explanation = "Świeża sesja — początek pracy, nie zmiana tematu."

        meta["_decision"] = final_decision
        meta["_confidence"] = final_confidence

        # Auto-set boundary marker. Three sources, any one is enough.
        # Fresh sessions: no boundary line — the whole thing IS the new task.
        if is_fresh:
            self._cached_boundary.pop(session_key_pre, None)
        else:
            cache_key_b = (session_key_pre, len(prompts))
            if cache_key_b in self._boundary_loc_cache:
                boundary_loc = self._boundary_loc_cache[cache_key_b]
            else:
                boundary_loc = find_last_new_task_prompt(prompts)
                if boundary_loc is None and drift_info and drift_info.drift_score is not None:
                    if drift_info.drift_score >= 0.40:
                        boundary_loc = len(prompts) - 1
                if boundary_loc is None and llm_verdict and not llm_verdict.error:
                    if llm_verdict.decision == "new_task" and llm_verdict.confidence >= 0.6:
                        boundary_loc = len(prompts) - 1
                self._boundary_loc_cache[cache_key_b] = boundary_loc
                # Cap cache
                if len(self._boundary_loc_cache) > 128:
                    for k in list(self._boundary_loc_cache.keys())[:32]:
                        self._boundary_loc_cache.pop(k, None)
            if boundary_loc is not None:
                self._cached_boundary[session_key_pre] = boundary_loc
            else:
                self._cached_boundary.pop(session_key_pre, None)

        # Build row
        color = DECISION_COLORS.get(final_decision, "#a7adba")
        icon = DECISION_ICONS.get(final_decision, "?")
        src_color = SOURCE_COLORS.get(source, "#a7adba")

        row = tk.Frame(self.list_inner, bg="#252932", cursor="hand2")
        row.pack(fill="x", pady=4)

        # Vertical decision color stripe
        stripe = tk.Frame(row, bg=color, width=6)
        stripe.pack(side="left", fill="y")

        inner = tk.Frame(row, bg="#252932")
        inner.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        line1 = tk.Frame(inner, bg="#252932")
        line1.pack(fill="x")
        tk.Label(line1, text=f"{icon} {final_decision.upper()}",
                  bg="#252932", fg=color,
                  font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(line1, text=f"  {final_confidence*100:.0f}%",
                  bg="#252932", fg="#e8e8e8",
                  font=("Segoe UI", 10)).pack(side="left")
        tk.Label(line1, text=f"  [{source}]",
                  bg="#252932", fg=src_color,
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8, 0))

        ago = time.time() - meta["last_activity"]
        ago_str = f"{int(ago/60)}m ago" if ago < 3600 else f"{ago/3600:.1f}h ago"
        tk.Label(line1, text=ago_str, bg="#252932", fg="#8a8e97",
                  font=("Segoe UI", 9)).pack(side="right")

        line2 = tk.Label(inner, text=meta["label"][:90],
                          bg="#252932", fg="#a7adba",
                          font=("Consolas", 9),
                          anchor="w", justify="left")
        line2.pack(fill="x", pady=(2, 0))

        # Line 3: token economy (precise if available)
        if token_stats and token_stats.n_turns > 0:
            ctx = latest_context_estimate(token_stats)
            pct = 100 * ctx / token_stats.context_window
            pct_color = "#7fdb95" if pct < 50 else ("#ffd166" if pct < 80 else "#ff7676")
            window_label = (f"{token_stats.context_window // 1000}k"
                             if token_stats.context_window >= 1000 else
                             str(token_stats.context_window))
            line3_text = (
                f"{len(prompts)} prompts | ctx {ctx:,}/{window_label} "
                f"({pct:.0f}%) | "
                f"billed {token_stats.total_billed:,} tok"
            )
            if token_stats.is_exact:
                line3_text += " (exact)"
            else:
                line3_text += " (est)"
        else:
            pct_color = "#8a8e97"
            line3_text = f"{len(prompts)} prompts | ~{total_tokens:,} tok (est)"

        tk.Label(inner, text=line3_text, bg="#252932", fg=pct_color,
                  font=("Consolas", 9), anchor="w").pack(fill="x", pady=(2, 0))

        # Line 4: signals + drift + ETA
        line4_parts = [
            f"CVS {cvs*100:.0f}%",
            f"new={boundary.new_task_hits} cpl={boundary.complete_hits} cnt={boundary.continuation_hits}",
        ]
        if drift_info and drift_info.drift_score is not None:
            line4_parts.append(f"drift {drift_info.drift_score:.2f}")
        if token_stats and token_stats.burn_rate_per_min:
            line4_parts.append(f"{token_stats.burn_rate_per_min:.0f} tok/min")
            ctx = latest_context_estimate(token_stats)
            line4_parts.append(f"ETA: {token_stats.eta_to_autocompact(ctx)}")
        tk.Label(inner, text=" | ".join(line4_parts),
                  bg="#252932", fg="#8a8e97",
                  font=("Consolas", 9), anchor="w").pack(fill="x", pady=(2, 0))

        # If L3 used — show its reasoning (more useful than generic recommendation)
        if llm_verdict and llm_verdict.available and not llm_verdict.error and llm_verdict.reasoning:
            l3_reason = f"🤖 L3 ({llm_verdict.confidence*100:.0f}%): {llm_verdict.reasoning}"
            if len(l3_reason) > 200:
                l3_reason = l3_reason[:200] + "..."
            tk.Label(inner, text=l3_reason, bg="#252932", fg="#c9d4e6",
                      font=("Segoe UI", 9, "italic"), anchor="w",
                      wraplength=560, justify="left").pack(fill="x", pady=(4, 0))
        else:
            rec = boundary.recommendation
            if len(rec) > 110:
                rec = rec[:110] + "..."
            tk.Label(inner, text=rec, bg="#252932", fg="#dde0e6",
                      font=("Segoe UI", 9), anchor="w",
                      wraplength=560, justify="left").pack(fill="x", pady=(4, 0))

        # Click → show detail
        session_key = f"{source}::{meta['session_id']}"
        self.detail_data[session_key] = {
            "meta": meta,
            "prompts": prompts,
            "boundary": boundary,
            "cvs": cvs,
            "total_tokens": total_tokens,
            "token_stats": token_stats,
            "drift": drift_info,
            "llm": llm_verdict,
            "final_decision": final_decision,
            "final_confidence": final_confidence,
            "explanation": explanation,
        }
        for w in (row, inner, line1, line2):
            w.bind("<Button-1>", lambda e, k=session_key: self._show_detail(k))
        for c in inner.winfo_children():
            c.bind("<Button-1>", lambda e, k=session_key: self._show_detail(k))

        self.session_widgets.append(row)

    def _show_detail(self, session_key: str) -> None:
        d = self.detail_data.get(session_key)
        if not d:
            return
        self._active_session_key = session_key
        self.copy_compact_btn.config(state="normal")
        self.copy_summary_btn.config(state="normal")
        self.reset_new_btn.config(state="normal")
        m = d["meta"]
        prompts = d["prompts"]
        try:
            last_ai = get_last_ai_response(m["_source"], m, max_chars=400)
        except Exception:
            last_ai = ""

        ts = d.get("token_stats")
        llm = d.get("llm")
        decision = d["final_decision"]
        confidence = d["final_confidence"]

        # ---- TOP: plain-language verdict ----
        verdict_lines = []
        if decision == "fresh":
            verdict_lines.append("✨ Świeża sesja — początek pracy")
        elif decision == "new_task":
            verdict_lines.append(f"⚡ Zmiana tematu w sesji ({confidence*100:.0f}% pewności)")
        elif decision == "task_complete":
            verdict_lines.append(f"⚠ Poprzedni task wygląda na zakończony ({confidence*100:.0f}%)")
        elif decision == "continuation":
            verdict_lines.append(f"✓ Kontynuacja tego samego tematu ({confidence*100:.0f}%)")
        else:
            verdict_lines.append(f"• Niejednoznaczne — sprawdź sam ({confidence*100:.0f}%)")

        # Why? Single line from L3 if available, otherwise from L1
        if llm and not llm.error and llm.reasoning:
            verdict_lines.append(f"Dlaczego: {llm.reasoning}")
        elif d.get("explanation"):
            verdict_lines.append(f"Dlaczego: {d['explanation']}")

        # Suggestion
        if decision == "new_task":
            verdict_lines.append(
                "→ Sugestia: kliknij \"Reset to new task only\" żeby zachować "
                "tylko kontekst nowego tematu."
            )
        elif decision == "task_complete":
            verdict_lines.append(
                "→ Sugestia: jeśli zaczynasz coś nowego — \"Copy /compact\" lub "
                "\"Reset to new task only\"."
            )

        text = "\n".join(verdict_lines) + "\n\n"

        # ---- MIDDLE: human-readable session status ----
        if ts and ts.n_turns > 0:
            ctx = latest_context_estimate(ts)
            pct = 100 * ctx / ts.context_window
            model_short = ts.model.replace("claude-", "") if ts.model else "?"
            duration_h = ts.duration_sec / 3600
            # Friendly labels for ctx and cost — no raw "billed tokens" jargon
            def _fmt_tok(n: int) -> str:
                if n >= 1_000_000:
                    return f"{n/1_000_000:.1f}M"
                if n >= 1_000:
                    return f"{n/1_000:.0f}k"
                return str(n)

            ctx_fill = "▰" * int(pct / 5) + "▱" * (20 - int(pct / 5))
            text += "Sesja\n"
            text += f"  Model:        {model_short}\n"
            text += f"  Czas:         {duration_h:.1f}h, {ts.n_turns} odpowiedzi AI, {len(prompts)} promptów\n"
            text += f"  Kontekst:     {_fmt_tok(ctx)} / {_fmt_tok(ts.context_window)}  {ctx_fill}  {pct:.0f}%\n"
            text += f"  Koszt:        ${ts.cost_usd:.2f}\n"
            if ctx > 0:
                eta = ts.eta_to_autocompact(ctx)
                text += f"  Auto-compact: za {eta}\n"
            text += "\n"

        # ---- BOTTOM: recent prompts with boundary line ----
        boundary_idx = self._cached_boundary.get(self._active_session_key)
        recent = prompts[-8:]
        offset = len(prompts) - len(recent)

        text += "Ostatnie prompty (najnowszy na dole)\n"

        # If boundary is BEFORE the recent window, show a note + line at top
        if boundary_idx is not None and boundary_idx < offset:
            ago = offset - boundary_idx
            text += f"  ═══ NOWY TASK zaczął się {ago} promptów wcześniej ═══\n"
        for i, p in enumerate(recent):
            idx_global = offset + i
            if boundary_idx is not None and idx_global == boundary_idx:
                text += "  ═══════ NOWY TASK ═══════\n"
            preview = p.text[:120].replace("\n", " ")
            if len(p.text) > 120:
                preview += "…"
            marker = "TY" if i == len(recent) - 1 else f"#{idx_global}"
            text += f"  [{marker}] {preview}\n"

        text += "\n"
        if last_ai:
            text += f"Ostatnia odpowiedź Claude:\n  {last_ai}"
        else:
            text += "(brak zapisanej odpowiedzi AI dla tego źródła)"

        # ---- FOOTER: technical breakdown (small, at the very bottom) ----
        b = d["boundary"]
        tech_parts = [
            f"L1 regex: new={b.new_task_hits} cpl={b.complete_hits} cnt={b.continuation_hits}"
        ]
        drift = d.get("drift")
        if drift and drift.drift_score is not None:
            tech_parts.append(f"L2 drift: {drift.drift_score:.2f}")
        if llm and not llm.error:
            tech_parts.append(
                f"L3: {llm.decision} ({llm.confidence*100:.0f}%, ${llm.cost_usd:.6f})"
            )
        text += "\n\n— szczegóły techniczne —\n  " + " | ".join(tech_parts)

        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _toggle_watch(self) -> None:
        if self.watch_var.get():
            self.watch_stop.clear()
            threading.Thread(target=self._watch_loop, daemon=True).start()
            self._set_status("Auto-refresh ON (10s)")
        else:
            self.watch_stop.set()
            self._set_status("Auto-refresh OFF")

    def _fetch_l3_background(self, session_id: str, prompts: list,
                             session_key: str = "") -> None:
        """Run L3 LLM call in background thread, schedule UI refresh when done.

        If verdict = new_task with high confidence, also fetch precise boundary index
        so the detail panel can draw the line at the exact prompt.
        """
        try:
            verdict = classify_via_llm(prompts, session_id=session_id)
        except Exception as exc:
            from llm_detector import LLMVerdict
            verdict = LLMVerdict(available=True, error=str(exc))
        self._l3_results[session_id] = verdict

        # Refine boundary index if L3 says new_task
        if (session_key and verdict.available and not verdict.error
                and verdict.decision == "new_task" and verdict.confidence >= 0.6):
            try:
                bi_result = find_boundary_index(prompts, session_id=session_id)
                bi = bi_result.get("boundary_index")
                if isinstance(bi, int) and 0 <= bi < len(prompts):
                    self._cached_boundary[session_key] = bi
            except Exception:
                pass  # keep the default last-prompt boundary

        self._l3_inflight.discard(session_id)
        # Cap cache
        if len(self._l3_results) > 64:
            for k in list(self._l3_results.keys())[:16]:
                self._l3_results.pop(k, None)
        # Debounced UI refresh — batch multiple L3 completions into one redraw
        # (500ms after the LAST one). Avoids 5× back-to-back refresh when 5
        # sessions finish L3 within seconds of each other.
        if not self._l3_refresh_pending:
            self._l3_refresh_pending = True
            self.root.after(500, self._refresh_after_l3)

    def _refresh_after_l3(self) -> None:
        """Re-render after a batch of background L3 calls finishes."""
        self._l3_refresh_pending = False
        # If more L3 calls are still in flight, wait for them before refreshing
        if self._l3_inflight:
            self._l3_refresh_pending = True
            self.root.after(500, self._refresh_after_l3)
            return
        self._refresh()
        # After list refresh: if user is viewing a session's detail, re-render
        # it so any new boundary line from L3 actually shows up.
        if self._active_session_key and self._active_session_key in self.detail_data:
            self._show_detail(self._active_session_key)

    def _watch_loop(self) -> None:
        while not self.watch_stop.is_set():
            self.root.after(0, self._refresh)
            for _ in range(100):  # 100 × 100ms = 10s
                if self.watch_stop.is_set():
                    return
                time.sleep(0.1)

    def _set_status(self, msg: str) -> None:
        self.status.config(text=msg)

    def _copy_to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def _copy_compact_command(self) -> None:
        if not self._active_session_key:
            return
        d = self.detail_data.get(self._active_session_key)
        if not d:
            return
        source = d["meta"]["_source"]
        if source == "claude_code":
            cmd = "/compact"
            note = "Wklej w Claude Code terminal aby skompresować historię."
        elif source == "codex":
            cmd = "/clear"
            note = "Wklej w Codex CLI aby zresetować context."
        else:
            cmd = "/compact"
            note = "Wklej w aktywnym CLI."
        self._copy_to_clipboard(cmd)
        self._set_status(f"Skopiowano: {cmd}  ({note})")

    def _reset_to_new_task(self) -> None:
        """Use L3 to find boundary, extract ONLY new task context, copy as reset prompt."""
        if not self._active_session_key:
            return
        d = self.detail_data.get(self._active_session_key)
        if not d:
            return
        prompts = d["prompts"]
        if len(prompts) < 2:
            self._set_status("Za mało historii")
            return

        if not l3_available():
            self._set_status("Reset wymaga L3 — ustaw OPENROUTER_API_KEY lub Ollamę w .env")
            return

        self._set_status("Identyfikuję boundary (L3, ~$0.0001)...")
        self.root.update_idletasks()

        result = find_boundary_index(prompts, session_id=d["meta"]["session_id"])
        bi = result.get("boundary_index")

        if bi is None:
            # No clear boundary — fallback to last prompt only
            new_task_prompts = prompts[-1:]
            note = "(brak wyraźnego boundary — bierze tylko ostatni prompt)"
        else:
            new_task_prompts = prompts[bi:]
            note = f"(boundary znaleziony przy [#{bi}]: {result.get('reasoning','')[:80]})"

        self._cached_boundary[self._active_session_key] = bi if bi is not None else len(prompts) - 1

        # Build reset prompt
        new_text = "\n".join(f"- {p.text[:200]}" for p in new_task_prompts[:8])
        reset_prompt = (
            "# Nowa sesja — kontekst nowego tasku\n\n"
            "Poprzednia rozmowa została zakończona. Zaczynamy od czystego kontekstu.\n\n"
            "## To czego dotyczy nowy task:\n\n"
            f"{new_text}\n\n"
            "Proszę o kontynuację bazując wyłącznie na tym kontekście, ignoruj poprzednie wątki."
        )
        self._copy_to_clipboard(reset_prompt)
        self._set_status(
            f"Skopiowano reset prompt {note}. "
            f"Otwórz nową sesję Claude/Codex i wklej."
        )
        # Refresh detail to show boundary line
        self._show_detail(self._active_session_key)

    def _copy_summary_brief(self) -> None:
        if not self._active_session_key:
            return
        d = self.detail_data.get(self._active_session_key)
        if not d:
            return
        prompts = d["prompts"]
        if len(prompts) < 2:
            self._set_status("Za mało historii do podsumowania")
            return
        self._set_status("Generuję podsumowanie (LLM, ~$0.0002)...")
        self.root.update_idletasks()

        if not l3_available():
            # Fallback: simple concat of last 10 user prompts
            recent = prompts[-10:]
            brief = ("# Brief z poprzedniej sesji\n\n"
                     + "\n".join(f"- {p.text[:150]}" for p in recent))
            self._copy_to_clipboard(brief)
            self._set_status("Skopiowano fallback brief (bez LLM)")
            return

        from llm_detector import (
            _load_env, _api_url, _api_headers, _model_id, backend_name,
            PRICE_INPUT, PRICE_OUTPUT,
        )
        _load_env()
        import json, urllib.request

        # Build prompt
        recent = prompts[-15:]
        history_text = "\n".join(f"[{i+1}] {p.text[:400]}" for i, p in enumerate(recent))
        sys_prompt = ("Jesteś asystentem podsumowującym sesję programistyczną. "
                       "Wygeneruj BRIEF (max 200 słów, po polsku) który użytkownik wklei "
                       "jako kontekst do NOWEJ sesji Claude/Codex. Zawiera: "
                       "1) Główny cel/temat sesji, 2) Co już zrobione, "
                       "3) Co jest 'w toku' lub do dokończenia. "
                       "NIE pisz markdown headers — zwykły tekst, krótko.")
        try:
            payload = {
                "model": _model_id(),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Ostatnie prompts:\n{history_text}"},
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            }
            req = urllib.request.Request(
                _api_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers=_api_headers(),
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            brief = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            if backend_name() == "ollama":
                cost = 0.0
            else:
                cost = (int(usage.get("prompt_tokens", 0)) * PRICE_INPUT
                        + int(usage.get("completion_tokens", 0)) * PRICE_OUTPUT)
            final = (
                "# Brief z poprzedniej sesji (kontynuujemy w nowej sesji)\n\n"
                + brief.strip()
                + "\n\nProszę o kontynuację pracy bazując na tym kontekście."
            )
            self._copy_to_clipboard(final)
            self._set_status(f"Skopiowano brief LLM (cost: ${cost:.6f}). "
                              f"Otwórz nową sesję Claude/Codex i wklej.")
        except Exception as exc:
            self._set_status(f"LLM error: {exc}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    # When launched via pythonw, stderr is null. Log crashes to file.
    import traceback
    log_path = Path(__file__).parent / "crash.log"
    try:
        main()
    except Exception:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"GUI crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            traceback.print_exc(file=f)
        # Try to show error in fallback dialog
        try:
            import tkinter.messagebox as mb
            mb.showerror("Task Boundary Detector - crash",
                          f"Aplikacja crashowała.\nLog: {log_path}\n\n"
                          + traceback.format_exc()[:500])
        except Exception:
            pass
        raise
