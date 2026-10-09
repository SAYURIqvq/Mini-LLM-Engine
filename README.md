# Mini-LLM-Engine

用 Python 与 PyTorch 实现的教学型 LLM 推理引擎，展示逐 token 解码、请求调度、每请求 KV 缓存和一个可处理失败的 HTTP 服务。

## 当前实现与边界

| 能力 | 当前实现 |
|---|---|
| 调度 | waiting/running 队列，完成后补入新请求 |
| 模型计算 | 一个 step 内逐请求前向；**尚未合并成一次 batched forward** |
| KV 缓存 | 每个请求保存 `past_key_values`，结束或取消后释放 |
| 采样 | temperature 与 top-p |
| 服务 | 单个专用推理线程，HTTP 事件循环独立处理请求 |
| 保护 | 有界待处理请求、输入预算、超时、取消清理、worker 故障传播 |
| 尚未实现 | paged KV cache、INT8 量化、真正 batched decode、流式输出 |

调度层支持请求动态加入，但不等同于 vLLM 的 GPU 批处理或分页注意力。已有 benchmark 脚本用于实验，仓库尚未给出经过统一环境与重复运行验证的性能优势结论。

## 快速开始

需要 Python 3.11+ 和支持目标模型的 PyTorch 运行环境。当前模型加载默认使用 float16 与 `device_map="auto"`；CPU-only 环境建议先运行下面的离线测试。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 按 tinyinfer/core/config.py 配置 TINYINFER_MODEL_PATH
uvicorn tinyinfer.api.server:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain KV caching briefly.","max_tokens":64,"temperature":0.7,"top_p":0.9}'
```

成功响应包含 `request_id`、`prompt`、`text` 和 `num_tokens`。模型权重、下载时间和显存要求取决于所选模型，离线测试不下载权重。

## 请求生命周期

`HTTP → 有界等待队列 → 单线程模型执行 → 结果 Future → HTTP 响应`

模型加载、分词、前向计算、调度和缓存回收均串行运行在同一推理线程。事件循环负责请求 Future、截止时间、客户端断开检测和健康检查，不在事件循环内执行模型前向计算。无请求时 worker 等待事件，不进行空转轮询。

| 情况 | HTTP 行为 |
|---|---|
| 非法采样参数、空白 prompt、超长上下文 | 422 |
| 待处理数达到上限 | 429，带 Retry-After |
| 超时（包含排队时间） | 504 |
| 推理 worker 失败 | 已等待及后续请求返回 503；health 返回 503 |
| 客户端断开 | 取消等待并安排清理，不继续占用结果缓存 |

默认 `TINYINFER_MAX_PENDING=64`、`TINYINFER_REQUEST_TIMEOUT=120`（秒）。请求最长 32768 字符、默认最多 4096 输入 tokens，`max_tokens` 为 1–4096；已知模型上下文长度也会检查。`temperature` 为 0–10，`top_p` 在 (0, 1] 内。

取消不能抢占正在执行的 GPU/PyTorch 调用；缓存会在当前 step 返回后清理，清理前仍占用 admission 容量。关闭服务也会等待当前 step 完成。永久卡住的底层调用需要进程级监督和重启，线程 worker 不提供硬中断。发生 worker 故障后应重启服务。

## 离线测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

使用假模型/分词器和真实 PyTorch 张量，覆盖调度、采样、缓存释放、上下文限制、worker 故障、超时、队列背压、HTTP 参数校验及推理阻塞时的 health 响应。测试不证明真实模型吞吐或 GPU 稳定性。

## Benchmark 与后续工作

在支持 vLLM 的 GPU 环境中安装 `requirements-benchmark.txt`，再运行 `benchmarks/` 下脚本。vLLM 不再是基础安装的强制依赖。

发布比较结果前固定模型、dtype、输入/输出 token 数、并发数、预热次数和停止条件，报告吞吐、延迟分位数、显存、失败率与原始记录。逐请求执行不能作为真正 batched decode 的替代基线。

下一步：实现 batched decode，验证不同输入长度及 EOS 的正确性，再评估 paged KV cache 和量化。
