import asyncio

from fastapi import FastAPI
from pydantic import BaseModel

from ..core.config import MODEL_PATH
from ..engine import TinyInferEngine, SamplingParams

# request/response schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0

class GenerateResponse(BaseModel):
    request_id: int
    prompt: str
    text: str
    num_tokens: int

app = FastAPI(title="TinyInfer API")
engine: TinyInferEngine = None


@app.on_event("startup")
async def startup():
    """Load model and start the engine loop on server startup."""
    global engine
    engine = TinyInferEngine(model_path=MODEL_PATH)
    asyncio.create_task(engine_loop())


async def engine_loop():
    """Background loop: keep running step() while there are pending requests."""
    while True:
        if engine.scheduler.has_pending():
            engine.step()
        await asyncio.sleep(0)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Accept a generate request, wait for result, return response."""
    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    request_id = engine.add_request(req.prompt, params)
    result = await engine.wait_for_result(request_id)

    return GenerateResponse(
        request_id=result.request_id,
        prompt=result.prompt,
        text=result.output_text,
        num_tokens=result.num_generated,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "waiting": engine.scheduler.num_waiting if engine else 0,
        "running": engine.scheduler.num_running if engine else 0,
    }
