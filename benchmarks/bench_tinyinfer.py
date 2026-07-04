"""TinyInfer benchmark: manual prefill/decode loop with KV-cache reuse."""
import asyncio
import os
import sys
import time

# 把项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyinfer import TinyInferEngine
from tinyinfer.engine import SamplingParams

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen2.5-1.5B-Instruct")

# 和其他 benchmark 完全一样的 prompts
PROMPTS = [
    "What is machine learning?",
    "Explain the transformer architecture.",
    "Write a Python function to sort a list.",
    "What is the difference between CPU and GPU?",
    "How does backpropagation work?",
    "Explain attention mechanism in one paragraph.",
    "What is a neural network?",
    "Describe gradient descent briefly.",
]


def main():
    print("=" * 50)
    print("TinyInfer (手动generate + KV cache)")
    print("=" * 50)

    engine = TinyInferEngine(model_path=MODEL_PATH)

    params = SamplingParams(temperature=0.7, max_tokens=100)

    print(f"\nRunning {len(PROMPTS)} prompts, max_tokens={params.max_tokens}")
    print("-" * 50)

    total_start = time.time()
    outputs = asyncio.run(run_batch(engine, PROMPTS, params))
    total_time = time.time() - total_start

    total_tokens = 0
    for i, out in enumerate(outputs):
        total_tokens += out.num_generated
        print(f"[{i+1}/{len(PROMPTS)}] {out.num_generated} tokens | {out.prompt[:40]}...")
        print(f"  -> {out.output_text[:80]}...")
        print()

    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total time:          {total_time:.2f}s")
    print(f"Total tokens:        {total_tokens}")
    print(f"Throughput:          {total_tokens / total_time:.2f} tokens/s")
    print(f"Avg latency/request: {total_time / len(PROMPTS):.2f}s")
    print("=" * 50)


async def run_batch(engine: TinyInferEngine, prompts: list[str], params: SamplingParams):
    request_ids = [engine.add_request(prompt, params) for prompt in prompts]
    wait_tasks = [asyncio.create_task(engine.wait_for_result(rid)) for rid in request_ids]

    while engine.scheduler.has_pending():
        engine.step()
        await asyncio.sleep(0)

    return await asyncio.gather(*wait_tasks)


if __name__ == "__main__":
    main()
