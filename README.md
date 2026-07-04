# Mini-LLM-Engine

一个轻量级的LLM 推理引擎，完全使用 Python 和 PyTorch 从零构建。它并非对现有服务框架的简单封装，而是手动实现了驱动 vLLM 和 TGI 等生产级系统的核心技术——包括连续批处理、KV 缓存复用以及核采样（nucleus sampling）。

---

## 为什么要从零构建推理引擎？

调用 HuggingFace transformers 库中的 model.generate() 时，幕后其实执行了大量的工作：提示词会在单次前向传播中完成预填充（prefill），Token 会以自回归方式逐个解码，并且系统会在静默中管理 KV 缓存以避免重复计算。然而，这种方式严格串行处理请求。

这对于任何实际的服务场景来说都是一个致命缺陷。Mini-LLM-Engine 剥开了所有的抽象层。它向你展示了 generate() 实际上是如何逐个 Token 执行的，并进一步引入了连续批处理机制，使得多个请求能够在单步循环内共享 GPU 时间。

---

## 架构

<img width="1666" height="720" alt="image" src="https://github.com/user-attachments/assets/3e5637b5-a12d-4227-8bec-dad370e940ef" />

该引擎由四个同心层组成，各层职责单一。最外层负责 HTTP 传输，中间层负责编排推理，最内层负责调度和底层 token 生成。各层之间通过简洁的方法调用自上向下通信，并通过基于 asyncio.Event 的通知机制自下向上通信。

1️⃣ API 层（api/server.py）：一个 FastAPI 应用，对外暴露 /generate 端点和 /health 健康检查接口。启动时，它会初始化引擎并启动一个后台 asyncio 循环，在有挂起请求时持续调用 engine.step()。

2️⃣ 引擎层（engine.py）：核心的 TinyInferEngine 类。它持有模型、分词器、调度器以及所有的请求状态。每次调用 step() 时，引擎会向调度器获取当前批次，然后遍历活跃请求——对新请求执行预填充前向传播，对正在处理的请求执行解码前向传播——并采样下一个 Token。

3️⃣ 调度器层（scheduler/continuous_batch.py）：ContinuousBatchScheduler 维护两个队列——waiting（等待）和 running（运行）。在每次调用 schedule() 时，它会驱逐已完成的请求，将等待队列中的请求提升至空闲的批次槽位（上限为 max_batch_size），并返回活跃批次。

4️⃣ 核心层（core/）：包含模型加载器（对 HuggingFace AutoModelForCausalLM 的轻量封装）、Request 数据类（携带生成状态及用于异步通知的 asyncio.Event），以及 sample_next_token 函数（实现了温度缩放和 top-p 核采样）。

---

## 端到端请求流程

当客户端发送 POST /generate 时，请求会在返回响应前穿越系统的每一层。这段旅程可以划分为三个不同的阶段：提交、推理和完成。下图追踪了单个请求穿过所有三个阶段的过程，并突出了每个组件发挥作用的位置。

<img width="1133" height="1004" alt="image" src="https://github.com/user-attachments/assets/d910e120-cc05-4398-a5e3-13a03623e018" />

这里关键的架构洞见在于异步交接：API 处理器在等待生成时不会阻塞事件循环。相反，它调用 await engine.wait_for_result(request_id)，从而挂起于 request.event.wait()。与此同时，一个独立的后台协程（engine_loop）驱动调度器执行 step()，确保 HTTP 服务器在并发负载下依然保持响应。

---

## Project Structure

```
Mini-LLM-Engine/
├── tinyinfer/
│   ├── __init__.py
│   ├── engine.py                # 核心推理循环 (prefill + decode + sample)
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py            # 带有后台引擎循环的 FastAPI 服务器
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # 模型路径解析 (环境变量或默认值)
│   │   ├── model_loader.py      # HuggingFace 模型与分词器加载
│   │   ├── request.py           # 请求状态机 (WAITING → RUNNING → FINISHED)
│   │   ├── status.py            # 轻量状态枚举
│   │   └── sampler.py           # 温度缩放 + top-p 核采样
│   └── scheduler/
│       ├── __init__.py
│       └── continuous_batch.py  # 带有等待/运行队列的连续批处理调度器
├── benchmarks/
│   ├── bench_naive.py           # 基线：串行 model.generate()
│   ├── bench_tinyinfer.py       # TinyInfer 引擎基准测试
│   └── bench_vllm.py            # vLLM 参考基准测试
├── tests/
│   ├── test_scheduler.py        # 连续批处理调度器测试
│   └── test_sampler.py          # 采样策略测试
├── examples/
│   └── api_smoke_client.py      # 顺序与并发 API 冒烟测试脚本
└── requirements.txt             # torch, transformers, fastapi, vllm, aiohttp 等

```
---
## 特点

| 特性 | 描述 | 实现位置 |
|------|------|----------|
| 连续批处理 | 多个请求在单步循环内共享 GPU 时间；已完成的请求被立即驱逐，新请求被立即提升 | tinyinfer/scheduler/continuous_batch.py |
| KV 缓存复用 | 每个请求在解码步骤间保留其 past_key_values，避免对注意力键值对进行冗余的重复计算 | tinyinfer/engine.py |
| 核采样（Top-p） | 将采样限制在累积概率超过 top_p 的最小 Token 集合中，在多样性与连贯性之间取得平衡 | tinyinfer/core/sampler.py |
| 温度缩放 | 在 softmax 之前将 logits 除以温度值；较低的值会产生更具确定性的输出，较高的值则会增加随机性 | tinyinfer/core/sampler.py |
| 异步事件驱动 API | 每个请求使用独立的 asyncio.Event，使得 HTTP 处理程序可以 await 完成状态而不会阻塞引擎循环 | tinyinfer/core/request.py |
| 健康监控 | /health 端点实时暴露等待中和运行中的请求数量 | tinyinfer/api/server.py |
| 对比基准测试 | 三个基准测试（朴素基线、Mini-LLM-Engine、vLLM）使用相同提示词，便于直接比较吞吐量和延迟 | benchmarks/ |


---

## 运行机制

当客户端发送带有提示词的 POST /generate 请求时，系统会迅速依次执行以下操作：提示词被分词并封装为 Request 对象，随后被放入调度器的等待队列中。一个后台 asyncio 任务会持续调用 engine.step()。在下一步中，调度器将等待中的请求提升至运行批次，引擎随即执行一次预填充前向传播——一次性处理整个提示词。在后续步骤中，引擎执行解码前向传播，仅输入上一个生成的 Token（连同缓存中的键值对）来生成下一个 Token。每一个新生成的 Token 都会根据配置的温度和 top-p 参数进行采样。一旦生成序列结束（EOS）Token 或达到最大 Token 数限制，该请求就会被标记为 FINISHED，其内部的 asyncio.Event 被触发，挂起的 HTTP 处理程序随之唤醒，将生成的文本返回给客户端。与此同时，调度器会立即释放该批次槽位以供下一个等待中的请求使用——这正是连续批处理的精髓所在。

---

## 快速上手

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
python examples/api_smoke_client.py

# run deterministic unit tests
pip install -r requirements-dev.txt
python -m pytest tests -q
```
