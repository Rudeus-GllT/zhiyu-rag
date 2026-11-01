from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_ctx, require_dashboard
from app.api.schemas import EvalRunRequest
from app.context import AppContext
from app.core.errors import AppError, NotFoundError

router = APIRouter(prefix="/api/v1/eval", tags=["eval"], dependencies=[Depends(require_dashboard)])


@router.get("/datasets")
async def list_datasets(request: Request):
    ctx: AppContext = get_ctx(request)
    return {"datasets": ctx.eval_runner.list_datasets()}


@router.post("/runs")
async def start_run(request: Request, payload: EvalRunRequest):
    ctx: AppContext = get_ctx(request)
    try:
        run_id = await run_in_threadpool(ctx.eval_runner.start, payload.dataset, payload.limit)
    except ValueError as exc:
        raise AppError(str(exc), code="invalid_dataset", status_code=400)
    run = ctx.db.get_eval_run(run_id)
    return {"run": run}


@router.get("/runs")
async def list_runs(request: Request, limit: int = Query(default=20, ge=1, le=100)):
    ctx: AppContext = get_ctx(request)
    runs = ctx.db.list_eval_runs(limit=limit)
    for run in runs:
        run.pop("cases", None)
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    ctx: AppContext = get_ctx(request)
    run = ctx.db.get_eval_run(run_id)
    if not run:
        raise NotFoundError("评测运行不存在")
    return {"run": run}
