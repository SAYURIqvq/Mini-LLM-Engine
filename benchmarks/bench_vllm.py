"""
vLLM benchmark - 用 vLLM 的离线推理
vLLM 内部自动做 continuous batching + PagedAttention
同样的 prompts，一次性丢给 vLLM 批量处理
"""
import time
import os
from vllm import LLM, SamplingParams

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen2.5-1.5B-Instruct")

# 和 naive 完全一样的 prompts
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
    print("vLLM Inference (continuous batching + PagedAttention)")
    print("=" * 50)

    # 加载模型
    print("\nLoading model...")
    t0 = time.time()
    llm = LLM(model=MODEL_PATH, dtype="float16", gpu_memory_utilization=0.8)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )

    # vLLM 批量推理 - 内部自动做 continuous batching
    print(f"\nRunning {len(PROMPTS)} prompts, max_tokens={sampling_params.max_tokens}")
    print("-" * 50)

    total_start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    total_time = time.time() - total_start

    # 打印每条结果
    total_tokens = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        output_len = len(output.outputs[0].token_ids)
        total_tokens += output_len
        print(f"[{i+1}/{len(PROMPTS)}] {output_len} tokens | {output.prompt[:40]}...")
        print(f"  -> {text[:80]}...")
        print()

    # 统计
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total time:          {total_time:.2f}s")
    print(f"Total tokens:        {total_tokens}")
    print(f"Throughput:          {total_tokens / total_time:.2f} tokens/s")
    print(f"Avg time/request:    {total_time / len(PROMPTS):.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
