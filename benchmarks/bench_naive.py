"""
Naive baseline - 用 transformers 逐条推理，没有任何优化
每个 prompt 单独跑 model.generate()，串行处理
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen2.5-1.5B-Instruct")

# 测试 prompts
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
    print("Naive Baseline (transformers, 逐条串行)")
    print("=" * 50)

    # 加载模型
    print("\nLoading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    max_new_tokens = 500
    results = []

    # 逐条推理 - 最朴素的方式
    print(f"\nRunning {len(PROMPTS)} prompts, max_new_tokens={max_new_tokens}")
    print("-" * 50)

    total_start = time.time()
    total_tokens = 0

    for i, prompt in enumerate(PROMPTS):
        start = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        output_len = outputs.shape[1] - input_len
        total_tokens += output_len
        latency = time.time() - start

        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append({"latency": latency, "output_tokens": output_len})

        print(f"[{i+1}/{len(PROMPTS)}] {latency:.2f}s | {output_len} tokens | {prompt[:40]}...")
        print(f"  -> {text[:80]}...")
        print()

    total_time = time.time() - total_start

    # 统计
    latencies = [r["latency"] for r in results]
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Total time:          {total_time:.2f}s")
    print(f"Total tokens:        {total_tokens}")
    print(f"Throughput:          {total_tokens / total_time:.2f} tokens/s")
    print(f"Avg latency/request: {sum(latencies) / len(latencies):.2f}s")
    print(f"Min latency:         {min(latencies):.2f}s")
    print(f"Max latency:         {max(latencies):.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
