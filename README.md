# TinyInfer

A lightweight LLM inference engine built from scratch in Python/PyTorch. Features continuous batching, paged KV cache management. Benchmarked against vLLM on throughput and latency.

## Project Structure

```
TinyInfer-/
├── tinyinfer/          # inference engine core
│   ├── api/            # FastAPI HTTP serving layer
│   ├── scheduler/      # continuous batching scheduler
│   └── memory/         # paged KV cache management (TODO)
├── benchmarks/         # performance benchmarks (naive, vLLM, TinyInfer)
├── tests/              # API integration tests
└── models/             # model weights (git ignored)
```

## Quick Start

```bash
# install dependencies
pip install -r requirements.txt

# run naive baseline
python benchmarks/bench_naive.py

# run vLLM benchmark
python benchmarks/bench_vllm.py

# start TinyInfer API server
uvicorn tinyinfer.api.server:app --host 0.0.0.0 --port 8000

# test with concurrent requests
python tests/test_api.py
```
