from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
os.chdir(ROOT)

from app.config import settings  # noqa: E402
from app.context import build_context  # noqa: E402

settings.auto_seed = False
ctx = build_context(settings)
print("datasets:", ctx.eval_runner.list_datasets())
run_id = ctx.eval_runner.start("golden")
print("run:", run_id)
for _ in range(120):
    run = ctx.db.get_eval_run(run_id)
    if run and run["status"] in {"completed", "failed"}:
        print("status:", run["status"], "progress:", run["progress"], "/", run["total"])
        print("metrics:", run["metrics"])
        for case in run["cases"]:
            print(
                f"{case['id']:>3} hit={case['metrics'].get('hit@5')} mrr={case['metrics'].get('mrr@5')} "
                f"faith={case['metrics'].get('faithfulness')} conf={case['confidence']:.2f} "
                f"Q={case['question'][:30]}"
            )
        break
    time.sleep(0.5)
else:
    print("timeout")
