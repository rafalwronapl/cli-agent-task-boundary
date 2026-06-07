# CLI Agent Task Boundary Detector

[![CI](https://github.com/rafalwronapl/cli-agent-task-boundary/actions/workflows/ci.yml/badge.svg)](https://github.com/rafalwronapl/cli-agent-task-boundary/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

Local task-boundary detection for Claude Code, OpenAI Codex CLI, and Gemini CLI
sessions, with an optional FinOps dashboard for anonymous team metrics.

The core question is simple: did the current assistant session drift into a new
task, so it is time to compact, reset, or start a fresh session?

The project is positioned as a multi-CLI agent-session boundary detector for
Claude Code, OpenAI Codex CLI, Gemini CLI, and similar local assistant tools.

## Run In 60 Seconds

```bash
python -m pip install -e ".[test,server]"
python -m server
```

In a second terminal:

```bash
python reporter.py --team engineering
```

Open `http://127.0.0.1:8787` to view the local dashboard.

## Why This Matters

Long assistant sessions drift: the user asks a new question, the context still
contains old assumptions, and cost/context metrics become hard to reason about.
For teams using CLI agents, the practical need is a local signal for when a
session should be compacted, reset, or reported as complete.

This project keeps that signal privacy-aware by default. Team dashboards receive
anonymous numeric metrics, while prompt text and file paths stay local.

## Demo

The FinOps dashboard below is shown with generated demo metrics. Real reporter
uploads use anonymous numeric session data only.

![Task Boundary FinOps dashboard demo](docs/screenshots/finops-dashboard-demo.png)

## What It Does

- Reads local CLI assistant session logs.
- Classifies recent prompts as `new_task`, `continuation`, `task_complete`, or
  `unclear`.
- Shows active sessions in a desktop GUI and CLI.
- Estimates token/context usage where local logs contain enough metadata.
- Optionally reports anonymous numeric metrics to a local or shared FinOps
  dashboard.

## Privacy Contract

The FinOps reporter sends only anonymous metrics:

- Sent: salted agent/session hashes, CLI source, model, prompt count,
  token/cost totals, context utilization, boundary verdict, duration, and
  timestamps.
- Never sent by `reporter.py`: prompts, AI responses, file paths,
  repository/project names, user names, or machine names.
- Layer 3 LLM classification is not used by `reporter.py`.
- Optional semantic drift in `reporter.py` is local-only and must be explicitly
  enabled with `--with-embeddings`.

The desktop GUI can optionally use an LLM backend for classification. If you
choose OpenRouter, the last few user prompts are sent to the configured cloud
model. If you choose Ollama, classification stays local.

## Quick Start

Install the server/dashboard dependencies:

```bash
pip install fastapi uvicorn duckdb
```

Start the local dashboard:

```bash
python -m server
```

In a second terminal, submit one anonymous snapshot:

```bash
python reporter.py --team engineering
```

Open:

```text
http://127.0.0.1:8787
```

For continuous collection:

```bash
python reporter.py --team engineering --watch --interval 300
```

On Windows, the same local flow is available through:

- `start_finops_server.bat`
- `report_finops_once.bat`

## API Token

By default the server binds to `127.0.0.1` and does not require a token. For a
shared dashboard or anything reachable beyond your own machine, set
`FINOPS_API_TOKEN` on both the server and reporters.

Server:

```bash
FINOPS_API_TOKEN=change-this-token python -m server
```

Reporter:

```bash
FINOPS_API_TOKEN=change-this-token python reporter.py --team engineering
```

Or pass it explicitly:

```bash
python reporter.py --api-token change-this-token --team engineering
```

The dashboard has an optional API token field. It stores the token in browser
local storage and sends it as `Authorization: Bearer <token>`.

For a non-local deployment, set the bind address explicitly:

```bash
FINOPS_HOST=0.0.0.0 FINOPS_PORT=8787 FINOPS_API_TOKEN=change-this-token python -m server
```

## FinOps API

The server stores local data in `data/finops.duckdb` and exposes:

- `POST /metrics` - idempotent upload of anonymous session snapshots.
- `GET /aggregate?days=30` - dashboard aggregate data.
- `GET /health` - backend health check.

When `FINOPS_API_TOKEN` is set, `/metrics` and `/aggregate` require either:

- `Authorization: Bearer <token>`
- `X-FinOps-Token: <token>`

## Desktop Detector

The original single-user detector remains available:

```bash
python gui.py
python cli.py
python cli.py --watch
```

Windows launcher:

```text
Task Boundary Detector.bat
```

Linux/macOS launcher:

```bash
./run.sh
```

## Detection Layers

Layer 1 is always available:

- Regex heuristics for Polish and English task switches.
- No external dependencies.
- No network.

Layer 2 is optional:

- Semantic topic drift with `sentence-transformers`.
- Local CPU inference after the model is downloaded.

Install:

```bash
pip install -r requirements.txt
```

Layer 3 is optional:

- OpenRouter cloud backend, configured with `OPENROUTER_API_KEY`.
- Ollama local backend, configured with `LLM_BACKEND=ollama`.

Configure through the GUI LLM dialog or copy `.env.example` to `.env`.

OpenRouter example:

```text
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

Ollama example:

```text
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://localhost:11434
```

## Supported CLIs

| CLI | Local storage | Status |
| --- | --- | --- |
| Claude Code | `~/.claude/projects/<hash>/<id>.jsonl` | User prompts, assistant responses, token metadata |
| OpenAI Codex CLI | `~/.codex/history.jsonl` | User prompts and timestamps |
| Google Gemini CLI | `~/.gemini/tmp/<user>/chats/session-*.jsonl` | User prompts and timestamps |

## Not Supported

- Cursor IDE: local SQLite metadata does not contain message bodies.
- OpenCode: current hidden adapter is intentionally not exposed because the
  storage format needs verification against a real install.
- Aider: removed from the exposed adapters for now; it can be restored later.

## Tests

Run:

```bash
pip install -e ".[test]"
python -m pytest
```

Current test coverage checks:

- Layer 1 classification behavior.
- Reporter anonymity contract.
- FinOps ingestion, deduplication, aggregation, and rejection of unexpected
  privacy-sensitive fields.
- Token-protected API behavior.

## Files

```text
cli-agent-task-boundary/
|-- detector.py                  # Layer 1 regex classifier and JSONL parsing
|-- embeddings_detector.py       # Layer 2 semantic drift
|-- llm_detector.py              # Layer 3 OpenRouter/Ollama verdicts
|-- adapters.py                  # Claude/Codex/Gemini session parsers
|-- token_economy.py             # Token and cost estimation
|-- gui.py                       # Tkinter desktop GUI
|-- settings_dialog.py           # LLM backend picker
|-- cli.py                       # Command-line interface
|-- reporter.py                  # Anonymous FinOps metric reporter
|-- server/                      # FastAPI + DuckDB backend
|-- dashboard/                   # Browser dashboard
|-- tests/                       # pytest suite
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- LICENSE
```

## Release Hygiene

Do not commit local state:

- `.env`
- `data/`
- `.pytest_cache/`
- `__pycache__/`
- `crash.log`
- local model caches

These paths are covered by `.gitignore`.

## License

MIT. See `LICENSE`.
