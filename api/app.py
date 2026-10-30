"""GroundedGuard HTTP API.

Run from the project root (one worker: one model per process):
    CUDA_VISIBLE_DEVICES=0 uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

The predictor is loaded once at startup. Inference runs in a one-thread pool behind one lock, so
GPU calls are sequential and always come from the same thread (measured: anyio's shared
threadpool, which hops between threads, doubled per-request model time under 4 concurrent clients). Configuration: environment variables, see api/settings.py.
Logs are one JSON object per line and never contain request text unless GG_LOG_CONTENT=true.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import BATCH_EXAMPLES, EXAMPLES, BatchResponse, DetectRequest, DetectResponse, Health
from api.settings import Settings
from groundedguard import __version__
from groundedguard.data.serialize import join_context
from groundedguard.inference.predictor import GroundedGuardPredictor

logger = logging.getLogger("groundedguard.api")
ERRORS = {422: {"description": "Invalid input (schema, empty response, input too long)"},
          503: {"description": "Model not loaded"}}


class ApiError(Exception):
    def __init__(self, status: int, detail):
        self.status, self.detail = status, detail


def _setup_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _log(level: int, **fields) -> None:
    logger.log(level, json.dumps(fields, ensure_ascii=False, default=str))


def to_input(req) -> dict:
    """API request -> GroundedGuardPredictor.predict() arguments."""
    if req.task == "qa":
        return {"response": req.response, "question": req.question, "context": req.passages}
    if req.task == "summary":
        return {"response": req.response, "context": req.document}
    return {"response": req.response, "structured": req.data}


def context_chars(x: dict) -> int:
    return len(join_context(*GroundedGuardPredictor._context(x.get("context"), x.get("question"), x.get("structured"))))


def check_limits(x: dict, settings: Settings, loc: list) -> None:
    errors = []
    if len(x["response"]) > settings.max_response_chars:
        errors.append({"loc": loc + ["response"], "type": "too_long",
                       "msg": f"response has {len(x['response'])} characters; the limit is "
                              f"{settings.max_response_chars} (GG_MAX_RESPONSE_CHARS)"})
    n = context_chars(x)
    if n > settings.max_context_chars:
        errors.append({"loc": loc, "type": "too_long",
                       "msg": f"context has {n} characters after serialization; the limit is "
                              f"{settings.max_context_chars} (GG_MAX_CONTEXT_CHARS)"})
    if errors:
        raise ApiError(422, errors)


def to_response(out: dict, request_id: str) -> dict:
    return {"has_hallucination": out["has_hallucination"], "score": out["response_score"], "threshold": out["threshold"],
            "spans": out["spans"], "windowed": out["windowed"], "n_windows": out["n_windows"], "model": out["model"],
            "latency_ms": out["latency_ms"], "request_id": request_id}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    _setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gg-inference")
        loop = asyncio.get_running_loop()
        try:
            app.state.predictor = await loop.run_in_executor(app.state.executor, lambda: GroundedGuardPredictor(
                settings.checkpoint, device=settings.device, precision=settings.precision))
            p = app.state.predictor
            # One warm-up call, so the first client request does not pay for CUDA initialization.
            t0 = time.perf_counter()
            await loop.run_in_executor(app.state.executor, lambda: p.predict("Warm-up response.", context="Warm-up context."))
            _log(logging.INFO, event="model_loaded", model=p.name, device=str(p.device), precision=settings.precision,
                 threshold=p.threshold, warmup_ms=round(1000 * (time.perf_counter() - t0), 1))
        except Exception as exc:  # the API stays up and answers 503
            _log(logging.ERROR, event="model_load_failed", error=type(exc).__name__)
        yield
        app.state.predictor = None
        app.state.executor.shutdown(wait=True)

    app = FastAPI(title="GroundedGuard", version=__version__, lifespan=lifespan,
                  description="Flags spans of a RAG response that its context does not support. "
                              "Scores are uncalibrated probabilities; compare them with `threshold`.")
    app.state.settings, app.state.lock, app.state.predictor, app.state.executor = settings, threading.Lock(), None, None

    def predictor_or_503(request: Request) -> GroundedGuardPredictor:
        p = request.app.state.predictor
        if p is None:
            raise ApiError(503, "model not loaded")
        return p

    async def run(request: Request, inputs: list[dict]) -> list[dict]:
        p = predictor_or_503(request)

        def locked():
            with request.app.state.lock:
                return p.predict_batch(inputs)

        # The lifespan's one-thread pool when the model was loaded there; the shared pool otherwise (tests).
        executor = request.app.state.executor
        try:
            if executor is None:
                return await run_in_threadpool(locked)
            return await asyncio.get_running_loop().run_in_executor(executor, locked)
        except ValueError as exc:  # e.g. a response too long for the model's max_len
            raise ApiError(422, [{"loc": ["body"], "type": "value_error", "msg": str(exc)}]) from None

    def note(request: Request, xs: list[dict], tasks: list[str]) -> None:
        request.state.log.update(tasks=tasks, n_inputs=len(xs), response_chars=[len(x["response"]) for x in xs],
                                 context_chars=[context_chars(x) for x in xs])
        if settings.log_content:
            request.state.log["response_text"] = [x["response"][:500] for x in xs]

    # ------------------------------------------------------------------------- middleware, errors
    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id, request.state.log = uuid.uuid4().hex[:16], {}
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # never return a stack trace
            _log(logging.ERROR, request_id=request.state.request_id, path=request.url.path, status=500,
                 error=type(exc).__name__)
            response = JSONResponse({"detail": "internal server error", "request_id": request.state.request_id},
                                    status_code=500)
        response.headers["X-Request-ID"] = request.state.request_id
        _log(logging.INFO, request_id=request.state.request_id, method=request.method, path=request.url.path,
             status=response.status_code, latency_ms=round(1000 * (time.perf_counter() - t0), 2), **request.state.log)
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse({"detail": exc.detail, "request_id": request.state.request_id}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        detail = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": detail, "request_id": request.state.request_id}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"detail": exc.detail, "request_id": request.state.request_id}, status_code=exc.status_code)

    # ------------------------------------------------------------------------- endpoints
    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    async def health(request: Request):
        p = request.app.state.predictor
        body = {"status": "ok" if p else "unavailable", "model": p.name if p else settings.model,
                "precision": settings.precision, "device": str(p.device) if p else settings.device,
                "threshold": p.threshold if p else None, "version": __version__}
        return JSONResponse(body, status_code=200 if p else 503)

    @app.post("/detect", response_model=DetectResponse, responses=ERRORS)
    async def detect(request: Request, req: Annotated[DetectRequest, Body(openapi_examples=EXAMPLES)]):
        x = to_input(req)
        note(request, [x], [req.task])
        check_limits(x, settings, ["body"])
        (out,) = await run(request, [x])
        return to_response(out, request.state.request_id)

    @app.post("/detect/batch", response_model=BatchResponse, responses=ERRORS)
    async def detect_batch(request: Request,
                           reqs: Annotated[list[DetectRequest], Body(openapi_examples=BATCH_EXAMPLES)]):
        if not 1 <= len(reqs) <= settings.max_batch:
            raise ApiError(422, [{"loc": ["body"], "type": "too_long" if reqs else "too_short",
                                  "msg": f"a batch holds 1 to {settings.max_batch} requests (GG_MAX_BATCH); "
                                         f"got {len(reqs)}"}])
        xs = [to_input(r) for r in reqs]
        note(request, xs, [r.task for r in reqs])
        for i, x in enumerate(xs):
            check_limits(x, settings, ["body", i])
        outs = await run(request, xs)
        rid = request.state.request_id
        return {"request_id": rid, "results": [to_response(o, f"{rid}-{i}") for i, o in enumerate(outs)]}

    return app


app = create_app()
