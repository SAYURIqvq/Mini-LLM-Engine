import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.config import MODEL_PATH
from ..engine import TinyInferEngine, SamplingParams
from ..runtime import InferenceRuntime, QueueFullError, WorkerUnavailableError


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32768)
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=10, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


class GenerateResponse(BaseModel):
    request_id: int
    prompt: str
    text: str
    num_tokens: int


def create_app(engine_factory=None, max_pending=64, request_timeout=120.0):
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = InferenceRuntime(
            engine_factory or (lambda: TinyInferEngine(model_path=MODEL_PATH)),
            max_pending=max_pending,
        )
        await app.state.runtime.start()
        try:
            yield
        finally:
            await app.state.runtime.close()

    app = FastAPI(title="TinyInfer API", lifespan=lifespan)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest, request: Request):
        runtime = request.app.state.runtime
        task = asyncio.create_task(runtime.generate(
            req.prompt, SamplingParams(req.temperature, req.top_p, req.max_tokens),
            timeout=request_timeout,
        ))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "Client disconnected")
            result = await task
            return GenerateResponse(request_id=result.request_id, prompt=result.prompt,
                                    text=result.output_text, num_tokens=result.num_generated)
        except QueueFullError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except WorkerUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, "Generation deadline exceeded") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if not task.done():
                task.cancel()
            # Consume the task even if a disconnect races a worker failure.
            await asyncio.gather(task, return_exceptions=True)

    @app.get("/health")
    async def health(request: Request):
        runtime = request.app.state.runtime
        return JSONResponse(status_code=200 if runtime.healthy else 503, content={
            "status": "ok" if runtime.healthy else "unavailable",
            "worker_alive": runtime.healthy,
            "pending": len(runtime.jobs),
            "max_pending": runtime.max_pending,
            **runtime.stats,
        })

    return app


app = create_app(max_pending=int(os.getenv("TINYINFER_MAX_PENDING", "64")),
                 request_timeout=float(os.getenv("TINYINFER_REQUEST_TIMEOUT", "120")))
