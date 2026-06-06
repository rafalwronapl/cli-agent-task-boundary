import os
import sys

import uvicorn

PUBLIC_HOSTS = {"0.0.0.0", "::", ""}


def main() -> int:
    host = os.environ.get("FINOPS_HOST", "127.0.0.1")
    port = int(os.environ.get("FINOPS_PORT", "8787"))
    token = os.environ.get("FINOPS_API_TOKEN", "").strip()
    if host in PUBLIC_HOSTS and not token:
        print(
            "Refusing to bind FinOps server to a public interface without FINOPS_API_TOKEN.",
            file=sys.stderr,
        )
        return 2
    uvicorn.run("server.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
