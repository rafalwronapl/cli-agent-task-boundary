"""Settings dialog — pick LLM backend, paste key, save to .env.

Opened from the main toolbar via "⚙ LLM". Modal Toplevel; on save it rewrites
the project-local .env file so a recipient can configure L3 without touching
a text editor.
"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ENV_PATH = Path(__file__).resolve().parent / ".env"


def _read_env() -> dict[str, str]:
    """Parse .env into a dict (uncommented lines only)."""
    if not ENV_PATH.exists():
        return {}
    out = {}
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env(backend: str, openrouter_key: str = "",
                ollama_model: str = "llama3.1:8b",
                ollama_url: str = "http://localhost:11434") -> None:
    """Replace .env with a clean, well-commented file for the chosen backend."""
    if backend == "openrouter":
        body = (
            "# OpenRouter (cloud). Pricing: ~$1e-7 per check with DeepSeek V4 Flash.\n"
            f"OPENROUTER_API_KEY={openrouter_key}\n"
            "# Optional override:\n"
            "# OPENROUTER_MODEL=deepseek/deepseek-v4-flash\n"
        )
    elif backend == "ollama":
        body = (
            "# Ollama (local, free). Requires `ollama serve` running.\n"
            "LLM_BACKEND=ollama\n"
            f"OLLAMA_MODEL={ollama_model}\n"
            f"OLLAMA_URL={ollama_url}\n"
        )
    else:  # off
        body = (
            "# Layer 3 disabled — only L1 (regex) + L2 (embeddings) will run.\n"
            "# To enable, open the LLM dialog in the GUI or edit this file.\n"
        )
    ENV_PATH.write_text(body, encoding="utf-8")
    # Also propagate into the current process so the change takes effect
    # without restarting the app.
    if backend == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        os.environ.pop("LLM_BACKEND", None)
    elif backend == "ollama":
        os.environ["LLM_BACKEND"] = "ollama"
        os.environ["OLLAMA_MODEL"] = ollama_model
        os.environ["OLLAMA_URL"] = ollama_url
        os.environ.pop("OPENROUTER_API_KEY", None)
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("LLM_BACKEND", None)


def _test_backend(backend: str, openrouter_key: str = "",
                   ollama_model: str = "llama3.1:8b",
                   ollama_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """Send a minimal request to verify the backend works. Returns (ok, message)."""
    import json
    import urllib.request

    if backend == "openrouter":
        if not openrouter_key.startswith("sk-"):
            return False, "Klucz powinien zaczynać się od 'sk-or-...'"
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {openrouter_key}",
                   "Content-Type": "application/json"}
        payload = {"model": "deepseek/deepseek-v4-flash",
                   "messages": [{"role": "user", "content": "ping"}],
                   "max_tokens": 5}
    elif backend == "ollama":
        url = f"{ollama_url.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": "Bearer ollama",
                   "Content-Type": "application/json"}
        payload = {"model": ollama_model,
                   "messages": [{"role": "user", "content": "ping"}],
                   "max_tokens": 5}
    else:
        return True, "L3 wyłączone — L1 + L2 wystarczą."

    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers,
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "choices" in data and data["choices"]:
            return True, "OK — backend odpowiada."
        return False, f"Niespodziewana odpowiedź: {str(data)[:120]}"
    except urllib.request.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        return False, f"HTTP {e.code}: {body or e.reason}"
    except Exception as e:
        if backend == "ollama":
            return False, (f"Nie udało się połączyć z Ollamą pod {ollama_url}.\n"
                           f"Czy `ollama serve` jest uruchomiona?\nBłąd: {e}")
        return False, f"Błąd: {e}"


def open_settings_dialog(parent: tk.Tk, on_saved=None) -> None:
    """Open the modal LLM settings dialog. Calls `on_saved()` after a save."""
    current = _read_env()
    if current.get("LLM_BACKEND", "").lower() == "ollama":
        current_backend = "ollama"
    elif current.get("OPENROUTER_API_KEY", "").startswith("sk-"):
        current_backend = "openrouter"
    else:
        current_backend = "off"

    dlg = tk.Toplevel(parent)
    dlg.title("LLM backend — konfiguracja")
    dlg.configure(bg="#1a1d23")
    dlg.geometry("520x440")
    dlg.transient(parent)
    dlg.grab_set()

    style = ttk.Style(dlg)

    backend_var = tk.StringVar(value=current_backend)
    or_key_var = tk.StringVar(value=current.get("OPENROUTER_API_KEY", ""))
    oll_model_var = tk.StringVar(value=current.get("OLLAMA_MODEL", "llama3.1:8b"))
    oll_url_var = tk.StringVar(value=current.get("OLLAMA_URL", "http://localhost:11434"))

    # Header
    tk.Label(dlg, text="Layer 3 — wybierz backend LLM",
             bg="#1a1d23", fg="#5fb3f7",
             font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=18, pady=(14, 6))
    tk.Label(dlg, text="L1 (regex) i L2 (embeddings) działają zawsze.\n"
                       "L3 daje semantyczną klasyfikację + przyciski Reset/Summary.",
             bg="#1a1d23", fg="#a7adba", justify="left",
             font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(0, 14))

    # Radio buttons
    radio_frame = tk.Frame(dlg, bg="#1a1d23")
    radio_frame.pack(fill="x", padx=18)
    for value, label in [
        ("openrouter", "OpenRouter (cloud, ~$1e-7 / sprawdzenie)"),
        ("ollama",     "Ollama (lokalnie, za darmo, wymaga `ollama serve`)"),
        ("off",        "Wyłączone (tylko L1 + L2)"),
    ]:
        tk.Radiobutton(radio_frame, text=label, value=value,
                       variable=backend_var,
                       bg="#1a1d23", fg="#e8e8e8",
                       selectcolor="#252932",
                       activebackground="#1a1d23",
                       activeforeground="#5fb3f7",
                       font=("Segoe UI", 10),
                       command=lambda: _toggle_fields()).pack(anchor="w", pady=2)

    # OpenRouter fields
    or_frame = tk.LabelFrame(dlg, text=" OpenRouter ", bg="#1a1d23", fg="#e8e8e8",
                              font=("Segoe UI", 9, "bold"))
    tk.Label(or_frame, text="API key (https://openrouter.ai/keys):",
             bg="#1a1d23", fg="#a7adba",
             font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))
    or_entry = tk.Entry(or_frame, textvariable=or_key_var,
                         bg="#252932", fg="#e8e8e8", insertbackground="#fff",
                         relief="flat", font=("Consolas", 9), show="*")
    or_entry.pack(fill="x", padx=10, pady=(0, 4))
    # Show/hide key
    show_var = tk.IntVar(value=0)
    def _toggle_show():
        or_entry.config(show="" if show_var.get() else "*")
    tk.Checkbutton(or_frame, text="Pokaż klucz", variable=show_var,
                   bg="#1a1d23", fg="#a7adba", selectcolor="#252932",
                   activebackground="#1a1d23",
                   command=_toggle_show,
                   font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 8))

    # Ollama fields
    oll_frame = tk.LabelFrame(dlg, text=" Ollama ", bg="#1a1d23", fg="#e8e8e8",
                               font=("Segoe UI", 9, "bold"))
    tk.Label(oll_frame, text="Model (musi być pulled: `ollama pull llama3.1:8b`):",
             bg="#1a1d23", fg="#a7adba",
             font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))
    tk.Entry(oll_frame, textvariable=oll_model_var,
             bg="#252932", fg="#e8e8e8", insertbackground="#fff",
             relief="flat", font=("Consolas", 9)).pack(fill="x", padx=10, pady=(0, 6))
    tk.Label(oll_frame, text="URL serwera:", bg="#1a1d23", fg="#a7adba",
             font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(0, 2))
    tk.Entry(oll_frame, textvariable=oll_url_var,
             bg="#252932", fg="#e8e8e8", insertbackground="#fff",
             relief="flat", font=("Consolas", 9)).pack(fill="x", padx=10, pady=(0, 8))

    def _toggle_fields() -> None:
        b = backend_var.get()
        or_frame.pack_forget()
        oll_frame.pack_forget()
        if b == "openrouter":
            or_frame.pack(fill="x", padx=18, pady=(10, 0))
        elif b == "ollama":
            oll_frame.pack(fill="x", padx=18, pady=(10, 0))

    _toggle_fields()

    # Status line
    status = tk.Label(dlg, text="", bg="#1a1d23", fg="#a7adba",
                      font=("Segoe UI", 9), wraplength=470, justify="left")
    status.pack(fill="x", padx=18, pady=(10, 4))

    # Buttons
    btn_frame = tk.Frame(dlg, bg="#1a1d23")
    btn_frame.pack(fill="x", side="bottom", padx=18, pady=12)

    def _do_test():
        b = backend_var.get()
        status.config(text="Testuję połączenie...", fg="#a7adba")
        dlg.update_idletasks()
        ok, msg = _test_backend(b, or_key_var.get().strip(),
                                 oll_model_var.get().strip(),
                                 oll_url_var.get().strip())
        status.config(text=msg, fg="#7fdb95" if ok else "#ff7676")

    def _do_save():
        b = backend_var.get()
        _write_env(b,
                   openrouter_key=or_key_var.get().strip(),
                   ollama_model=oll_model_var.get().strip() or "llama3.1:8b",
                   ollama_url=oll_url_var.get().strip() or "http://localhost:11434")
        # Reset llm_detector backend cache so next call re-detects
        try:
            import llm_detector
            llm_detector._BACKEND = None
            llm_detector._AVAILABLE = None
        except Exception:
            pass
        status.config(text=f"Zapisano do .env. Backend: {b}.", fg="#7fdb95")
        if on_saved:
            on_saved()
        dlg.after(600, dlg.destroy)

    ttk.Button(btn_frame, text="Testuj",
               command=_do_test).pack(side="left")
    ttk.Button(btn_frame, text="Zapisz", command=_do_save).pack(side="right", padx=(6, 0))
    ttk.Button(btn_frame, text="Anuluj",
               command=dlg.destroy).pack(side="right")

    dlg.wait_window()
