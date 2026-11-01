"""Development entry point.

    python run.py

Adds the local ``vendor`` directory to sys.path (the demo ships dependencies
there because the workspace path is non-ASCII), then starts uvicorn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
os.chdir(ROOT)
os.environ.setdefault("PYTHONUTF8", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


def main() -> None:
    import uvicorn

    from app.config import settings

    host = os.environ.get("HOST", settings.host)
    port = int(os.environ.get("PORT", settings.port))
    reload_enabled = os.environ.get("RELOAD", "0") == "1"
    print(f"\n  {settings.app_name} · {settings.app_subtitle}")
    print(f"  Web console : http://{host}:{port}")
    print(f"  API docs    : http://{host}:{port}/docs\n")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
