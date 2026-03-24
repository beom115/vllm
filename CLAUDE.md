# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
# Python-only install (no CUDA compilation)
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Full install with CUDA (compiles C++/CUDA kernels)
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install -e . --no-build-isolation
```

## Common Commands

```bash
# Lint/format
pre-commit run -a

# Type checking
pre-commit run --hook-stage manual mypy-3.10

# Run all tests
pytest tests/

# Run a single test file
pytest -s -v tests/test_logger.py

# Run a single test function
pytest -s -v tests/test_logger.py::test_function_name
```

Test markers in `pyproject.toml`: `slow_test`, `core_model`, `distributed`, `cpu_model`, `skip_v1`, `optional`.

## Code Style

- Google Python Style Guide; linted with ruff (E, F, UP, B, ISC, SIM, I, G rules)
- Google C++ Style Guide; formatted with clang-format
- All files require Apache 2.0 SPDX headers
- All commits require `Signed-off-by:` (DCO)
- PR title prefixes: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Hardware][Vendor]`, `[Misc]`

## Architecture Overview

vLLM is an LLM inference and serving engine. The main components are:

### Entry Points (`vllm/entrypoints/`)
- `llm.py` — synchronous `LLM` class (user-facing)
- `openai/` — OpenAI-compatible HTTP API server
- `grpc_server.py` — gRPC server

### Engine (`vllm/v1/engine/`)
The V1 engine is the current primary execution path. `LLMEngine` wraps `EngineCoreClient`, which communicates via ZMQ IPC with `EngineCore`. The engine core runs an inner event loop: schedule → execute → process outputs.

Key classes: `LLMEngine`, `AsyncLLMEngine`, `EngineCore`, `EngineCoreClient`, `InputProcessor`, `OutputProcessor`, `Detokenizer`.

### Scheduler (`vllm/v1/core/sched/scheduler.py`)
`Scheduler.schedule()` is called each iteration to select which requests run. It:
1. Checks available KV cache blocks
2. Selects requests by policy (FCFS/priority)
3. Allocates KV blocks via `KVCacheManager`
4. Returns `SchedulerOutput`

### KV Cache (`vllm/v1/core/kv_cache_manager.py`)
Implements PagedAttention: KV cache is divided into fixed-size blocks. `KVCacheManager` handles allocation, prefix caching (block hashing for deduplication), and eviction.

### Workers (`vllm/v1/worker/`)
Hardware-specific executors (GPU, CPU, TPU, XPU). Each worker contains a `ModelRunner` that:
1. Prepares input tensors (`InputBatch`)
2. Runs the model forward pass
3. Samples tokens via `Sampler`

### Models (`vllm/model_executor/models/`)
All supported model implementations. Models register themselves via class hierarchy. Base mixins: `CausalMixin`, `MultiModalMixin`, `EmbeddingMixin`, `MoEMixin`.

### Configuration (`vllm/config/`)
`VllmConfig` aggregates all configs: `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `ParallelConfig`, `AttentionConfig`, `CompilationConfig`, `LoRAConfig`, `MultiModalConfig`. All validated via Pydantic.

### Distributed Execution (`vllm/distributed/`)
Supports tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), expert parallelism (EP for MoE), and context parallelism. Uses `torch.distributed`.

### Layers and Kernels (`vllm/model_executor/layers/`, `vllm/kernels/`)
Custom attention kernels (Flash Attention, FlashInfer), quantization layers (GPTQ, AWQ, FP8, INT4/8), normalization, and sampling. CUDA graph support for static batch optimization.

### Request Lifecycle
```
User API → InputProcessor (tokenize + multimodal) → Scheduler (batch + KV alloc)
→ Worker/ModelRunner (forward pass) → OutputProcessor (detokenize) → RequestOutput
```

Request states (`vllm/v1/request.py`): `WAITING → RUNNING → FINISHED` (or `PREEMPTED → WAITING`).

## Key Files

| File | Purpose |
|------|---------|
| `vllm/entrypoints/llm.py` | Main user-facing `LLM` class |
| `vllm/v1/engine/core.py` | EngineCore event loop |
| `vllm/v1/core/sched/scheduler.py` | Request scheduler |
| `vllm/v1/core/kv_cache_manager.py` | KV cache management |
| `vllm/v1/worker/gpu_worker.py` | GPU worker implementation |
| `vllm/config/` | All configuration dataclasses |
| `vllm/model_executor/models/` | Model implementations |
