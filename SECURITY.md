# Security Policy

## Data Handling

`reporter.py` is designed to upload only anonymous numeric metrics. It must not
send prompts, assistant responses, local file paths, repository names, user
names, or machine names.

The desktop GUI reads local assistant logs and can display prompt/response
snippets locally. If Layer 3 is configured with OpenRouter, recent user prompts
are sent to the configured OpenRouter model for classification. Use Ollama if
you need Layer 3 to stay local.

## FinOps Server

The server binds to `127.0.0.1` by default. If you bind it to a LAN or public
interface, set `FINOPS_API_TOKEN` and configure reporters with the same token.

Protected endpoints:

- `POST /metrics`
- `GET /aggregate`

Accepted token formats:

- `Authorization: Bearer <token>`
- `X-FinOps-Token: <token>`

Do not publish local state files such as `.env`, `data/finops.duckdb`,
`crash.log`, caches, or model downloads. These paths are covered by
`.gitignore`.

## Reporting Issues

Open a GitHub issue with a minimal reproduction. Do not include real assistant
logs, prompts, responses, API keys, or database files.
