"""
Test TinyInfer API server with concurrent requests.

Usage:
  1. Start the server:  uvicorn tinyinfer.api.server:app --host 0.0.0.0 --port 8000
  2. Run this script:   python tests/test_api.py
"""
import asyncio
import time
import aiohttp

SERVER_URL = "http://localhost:8000"

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


async def send_request(session: aiohttp.ClientSession, prompt: str, idx: int):
    """Send a single generate request and return the result."""
    start = time.time()
    payload = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    }
    async with session.post(f"{SERVER_URL}/generate", json=payload) as resp:
        result = await resp.json()
        latency = time.time() - start
        print(f"[{idx+1}/{len(PROMPTS)}] {latency:.2f}s | {result['num_tokens']} tokens | {prompt[:40]}...")
        print(f"  -> {result['text'][:80]}...")
        print()
        return {"latency": latency, "num_tokens": result["num_tokens"]}


async def test_sequential():
    """Send requests one by one."""
    print("=" * 50)
    print("Sequential requests (one by one)")
    print("=" * 50)

    total_start = time.time()
    results = []
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(PROMPTS):
            result = await send_request(session, prompt, i)
            results.append(result)

    total_time = time.time() - total_start
    total_tokens = sum(r["num_tokens"] for r in results)
    print("=" * 50)
    print(f"Total time:     {total_time:.2f}s")
    print(f"Total tokens:   {total_tokens}")
    print(f"Throughput:     {total_tokens / total_time:.2f} tokens/s")
    print("=" * 50)
    return total_time


async def test_concurrent():
    """Send all requests at the same time."""
    print("\n" + "=" * 50)
    print("Concurrent requests (all at once)")
    print("=" * 50)

    total_start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, prompt, i) for i, prompt in enumerate(PROMPTS)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - total_start
    total_tokens = sum(r["num_tokens"] for r in results)
    print("=" * 50)
    print(f"Total time:     {total_time:.2f}s")
    print(f"Total tokens:   {total_tokens}")
    print(f"Throughput:     {total_tokens / total_time:.2f} tokens/s")
    print("=" * 50)
    return total_time


async def main():
    # check server is up
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/health") as resp:
                health = await resp.json()
                print(f"Server health: {health}\n")
    except aiohttp.ClientError:
        print("Error: server is not running. Start it with:")
        print("  uvicorn tinyinfer.api.server:app --host 0.0.0.0 --port 8000")
        return

    seq_time = await test_sequential()
    conc_time = await test_concurrent()

    print(f"\n{'=' * 50}")
    print(f"Speedup: {seq_time / conc_time:.2f}x")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
