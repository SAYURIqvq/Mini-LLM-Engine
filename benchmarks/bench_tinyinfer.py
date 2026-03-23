"""
TinyInfer benchmark - 测试自己实现的推理引擎
手动 generate 循环 + KV cache 复用
"""
import os
import sys

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

    outputs = engine.generate(PROMPTS, params)

    total_tokens = 0
    total_time = 0
    for i, out in enumerate(outputs):
        total_tokens += len(out.token_ids)
        total_time += out.latency
        print(f"[{i+1}/{len(PROMPTS)}] {out.latency:.2f}s | {len(out.token_ids)} tokens | {out.prompt[:40]}...")
        print(f"  -> {out.text[:80]}...")
        print()

    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total time:          {total_time:.2f}s")
    print(f"Total tokens:        {total_tokens}")
    print(f"Throughput:          {total_tokens / total_time:.2f} tokens/s")
    print(f"Avg latency/request: {total_time / len(PROMPTS):.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
