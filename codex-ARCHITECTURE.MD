# ARCHITECTURE: sgl.Engine Offline Batch Inference

> Scope: This document describes the internal architecture and control/data flow of sglang’s offline batch inference API, i.e., the in‑process Python engine `sgl.Engine` (not the HTTP server). It follows the spirit of matklad’s ARCHITECTURE.md guidance and traces initialization, configuration, and the end‑to‑end call stack for `llm.generate(...)`.

- Audience: contributors and advanced users who want to understand how an `Engine` instance launches, how requests flow, and where the actual PyTorch model lives and executes.
- Out of scope: HTTP server/routers, OpenAI‑compatible layer, UI, and non‑Engine client APIs.

## Table of Contents

- [Overview](#overview)
- [High‑Level Dataflow](#high-level-dataflow)
- [Initialization Flow](#initialization-flow)
  - [Engine construction](#engine-construction)
  - [Subprocess topology](#subprocess-topology)
  - [Tokenizer and templates](#tokenizer-and-templates)
- [Configuration (ServerArgs)](#configuration-serverargs)
  - [Model and tokenizer](#model-and-tokenizer)
  - [Runtime and parallelism](#runtime-and-parallelism)
  - [Memory, scheduling, and performance](#memory-scheduling-and-performance)
  - [Logging, tracing, metrics](#logging-tracing-metrics)
  - [LoRA and weight updates](#lora-and-weight-updates)
  - [Speculative decoding](#speculative-decoding)
- [Generate Call Stack](#generate-call-stack)
  - [Synchronous vs asynchronous vs streaming](#synchronous-vs-asynchronous-vs-streaming)
  - [Single‑process vs multi‑process](#single-process-vs-multi-process)
- [Where the Model Lives and Executes](#where-the-model-lives-and-executes)
  - [Model instantiation](#model-instantiation)
  - [GPU execution path](#gpu-execution-path)
  - [Return path (tokens → text)](#return-path-tokens--text)
- [Getting a Model Handle](#getting-a-model-handle)
  - [Supported control surfaces](#supported-control-surfaces)
  - [What is not supported](#what-is-not-supported)
- [Examples](#examples)
- [References](#references)


## Overview

`sgl.Engine` gives a local, offline, in‑process Python API for batch text (and multi‑modal) inference. Internally, the engine is multi‑process:

- Main process hosts the Python API (`Engine`) and the `TokenizerManager`.
- One or more scheduler process(es) host the actual model execution and batching/scheduling logic.
- A detokenizer process maps token streams back to text (or embeddings/images for multimodal).

These processes communicate over ZeroMQ (ZMQ) IPC/TCP sockets. Data paths and control paths are explicitly structured; concurrency uses asyncio in the tokenizer manager and Python subprocesses for GPU workers.

Key entry points:
- `Engine` API: [python/sglang/srt/entrypoints/engine.py#L93](python/sglang/srt/entrypoints/engine.py#L93)
- `TokenizerManager`: [python/sglang/srt/managers/tokenizer_manager.py#L146](python/sglang/srt/managers/tokenizer_manager.py#L146)
- `Scheduler` (GPU worker, batching): [python/sglang/srt/managers/scheduler.py#L280](python/sglang/srt/managers/scheduler.py#L280)
- `DetokenizerManager`: [python/sglang/srt/managers/detokenizer_manager.py#L72](python/sglang/srt/managers/detokenizer_manager.py#L72)


## High‑Level Dataflow

```
Main Process                               Subprocesses

+---------------------+        ZMQ         +-------------------+     ZMQ      +----------------------+
|  Engine (API)       | <----------------> |  Scheduler(s)     | <--------->  |  DetokenizerManager  |
|  + TokenizerManager |  requests/results  |  + TpModelWorker  |   token ids  |   (text decode)      |
+---------------------+                    |  + ModelRunner    |   batched    +----------------------+
                                           +-------------------+

Legend:
- Engine constructs requests, normalizes inputs, handles streaming
- TokenizerManager tokenizes, batches, and routes to Scheduler
- Scheduler performs batching, prefill/decode, sampling on GPU(s)
- Detokenizer converts token ids back to strings and reports metrics
```


## Initialization Flow

### Engine construction

1) User creates `Engine` with keyword args of `ServerArgs`:
- `Engine.__init__` builds or accepts a `ServerArgs` instance and registers an atexit shutdown hook.
- Code: [python/sglang/srt/entrypoints/engine.py#L107-L120](python/sglang/srt/entrypoints/engine.py#L107-L120)

2) Ports and inter‑process sockets are allocated via `PortArgs.init_new(...)`.
- Code: [python/sglang/srt/entrypoints/engine.py#L125-L127](python/sglang/srt/entrypoints/engine.py#L125-L127)

3) Subprocesses are launched via `_launch_subprocesses(...)`:
- Code: [python/sglang/srt/entrypoints/engine.py#L754](python/sglang/srt/entrypoints/engine.py#L754)

This call:
- Downloads/locates model weights if necessary (`prepare_model_and_tokenizer`).
- Spawns one scheduler process per TP/PP (or a DataParallelController if `dp_size > 1`).
- Spawns a single `DetokenizerManager` process.
- Initializes the `TokenizerManager` (or a `MultiTokenizerRouter` if `tokenizer_worker_num > 1`).
- Waits for GPU workers to finish weight loading and reports `scheduler_info`.

4) RPC socket for collective scheduler calls is created.
- Code: [python/sglang/srt/entrypoints/engine.py#L139-L142](python/sglang/srt/entrypoints/engine.py#L139-L142)

5) Optional tracing init if enabled.
- Code: [python/sglang/srt/entrypoints/engine.py#L144-L148](python/sglang/srt/entrypoints/engine.py#L144-L148)

Result: the returned `Engine` object has:
- `self.server_args` (as passed/normalized)
- `self.tokenizer_manager` (in main process)
- `self.template_manager` (prompt templates)
- `self.scheduler_info` (capacity information)

### Subprocess topology

- No data parallelism (`dp_size == 1`):
  - For each TP×PP rank, one scheduler process is spawned.
  - Code: [python/sglang/srt/entrypoints/engine.py#L795-L816](python/sglang/srt/entrypoints/engine.py#L795-L816)

- With data parallelism (`dp_size > 1`):
  - A `DataParallelController` process is spawned, which in turn launches TP/PP schedulers and load‑balances input requests across DP ranks.
  - Code: [python/sglang/srt/entrypoints/engine.py#L823-L831](python/sglang/srt/entrypoints/engine.py#L823-L831), [python/sglang/srt/managers/data_parallel_controller.py#L1](python/sglang/srt/managers/data_parallel_controller.py#L1)

- Detokenizer process is always spawned once per `Engine`.
  - Code: [python/sglang/srt/entrypoints/engine.py#L856-L864](python/sglang/srt/entrypoints/engine.py#L856-L864)

- Multi‑node: non‑zero `node_rank` nodes avoid launching tokenizer/detokenizer locally and wait for schedulers to become ready.
  - Code: [python/sglang/srt/entrypoints/engine.py#L833-L855](python/sglang/srt/entrypoints/engine.py#L833-L855)

### Tokenizer and templates

- In the main process, `TokenizerManager` is instantiated unless `tokenizer_worker_num > 1` (then `MultiTokenizerRouter` is used).
  - Code: [python/sglang/srt/entrypoints/engine.py#L866-L874](python/sglang/srt/entrypoints/engine.py#L866-L874)
- Prompt templates are initialized via `TemplateManager` against the tokenizer manager and model path.
  - Code: [python/sglang/srt/entrypoints/engine.py#L742-L749](python/sglang/srt/entrypoints/engine.py#L742-L749)

`TokenizerManager` loads the tokenizer/processor unless `skip_tokenizer_init=True` and creates ZMQ sockets for communication with the Scheduler and Detokenizer.
- Code: [python/sglang/srt/managers/tokenizer_manager.py#L255-L269](python/sglang/srt/managers/tokenizer_manager.py#L255-L269)


## Configuration (ServerArgs)

`ServerArgs` is the authoritative configuration for the Engine and schedulers. All `sgl.Engine(**kwargs)` map 1:1 to `ServerArgs` fields.
- Definition: [python/sglang/srt/server_args.py#L178](python/sglang/srt/server_args.py#L178)

Below are highlights (not exhaustive). Refer to the source for full semantics and defaults.

### Model and tokenizer
- `model_path`, `revision`, `tokenizer_path`, `tokenizer_mode`, `skip_tokenizer_init`
- `load_format` and `quantization` (e.g., safetensors, gguf, awq/gptq/fp8, etc.)
- `is_embedding`, `enable_multimodal`

### Runtime and parallelism
- `device` (cuda, cpu, xpu, npu, hpu)
- `tp_size`, `pp_size`, `dp_size` and multi‑node `nnodes`, `node_rank`
- `elastic_ep_backend`, `moe_*` for expert parallelism

### Memory, scheduling, and performance
- `mem_fraction_static`, `max_total_tokens`, `max_running_requests`, `max_prefill_tokens`
- `attention_backend`, `kv_cache_dtype`, `page_size`, `chunked_prefill_size`
- Overlap schedule, CUDA graph, deterministic inference knobs

### Logging, tracing, metrics
- `log_level`, `log_requests`, `enable_metrics`, `enable_trace`, `oltp_traces_endpoint`

### LoRA and weight updates
- `enable_lora`, `lora_*`, dynamic load/unload of adapters
- Online weight update control toggles and group settings

### Speculative decoding
- `speculative_algorithm`, `speculative_draft_model_path`, `speculative_num_steps`, etc.


## Generate Call Stack

When the user calls `llm.generate(...)`:

1) API entry: `Engine.generate(...)` builds a `GenerateReqInput` and invokes the tokenizer manager’s async generator. For synchronous usage, it drives the event loop itself.
- Code: [python/sglang/srt/entrypoints/engine.py#L150-L213](python/sglang/srt/entrypoints/engine.py#L150-L213)

2) Tokenization and dispatch: `TokenizerManager.generate_request(...)` normalizes inputs (text/input_ids/embeds, batch vs single), tokenizes if needed, and sends tokenized requests to the scheduler over ZMQ.
- Code: [python/sglang/srt/managers/tokenizer_manager.py#L370-L414](python/sglang/srt/managers/tokenizer_manager.py#L370-L414)
- Send path: [python/sglang/srt/managers/tokenizer_manager.py#L820](python/sglang/srt/managers/tokenizer_manager.py#L820) and [python/sglang/srt/managers/tokenizer_manager.py#L840](python/sglang/srt/managers/tokenizer_manager.py#L840)

3) Scheduler ingress: the `Scheduler` on TP0/PP0 receives requests via `recv_from_tokenizer`, broadcasts as needed to other TP ranks, and enqueues work.
- Code: [python/sglang/srt/managers/scheduler.py#L1200](python/sglang/srt/managers/scheduler.py#L1200)
- Receivers and sockets: [python/sglang/srt/managers/scheduler.py#L340-L373](python/sglang/srt/managers/scheduler.py#L340-L373)

4) Batching and forward passes: the scheduler constructs `ScheduleBatch`es and runs `TpModelWorker` → `ModelRunner.forward(...)` (prefill/decode) and `ModelRunner.sample(...)`. Optional overlap scheduling/pipeline parallelism are handled here.
- Event loop variants: [python/sglang/srt/managers/scheduler.py#L1018](python/sglang/srt/managers/scheduler.py#L1018) (normal), [python/sglang/srt/managers/scheduler.py#L1037](python/sglang/srt/managers/scheduler.py#L1037) (overlap), [python/sglang/srt/managers/scheduler.py#L1065](python/sglang/srt/managers/scheduler.py#L1065) (pipeline parallel)
- GPU worker wrapper: [python/sglang/srt/managers/tp_worker.py#L214](python/sglang/srt/managers/tp_worker.py#L214)
- ModelRunner core: [python/sglang/srt/model_executor/model_runner.py#L236](python/sglang/srt/model_executor/model_runner.py#L236)

5) Output packaging: the scheduler emits token id chunks (and optional logits/logprobs/hidden states) as `BatchTokenIDOutput` to the detokenizer.
- Code: [python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899)

6) Detokenization: `DetokenizerManager` incrementally decodes token ids to text and streams `BatchStrOutput` (or embeddings/multimodal outputs) back to the tokenizer manager.
- Code: [python/sglang/srt/managers/detokenizer_manager.py#L114-L120](python/sglang/srt/managers/detokenizer_manager.py#L114-L120), [python/sglang/srt/managers/detokenizer_manager.py#L227](python/sglang/srt/managers/detokenizer_manager.py#L227)

7) Result assembly and streaming: `TokenizerManager._handle_batch_output(...)` updates request state, metrics, and yields either streaming chunks or the final result to the Engine’s consumer.
- Code: [python/sglang/srt/managers/tokenizer_manager.py#L1306](python/sglang/srt/managers/tokenizer_manager.py#L1306)

8) Engine returns: In synchronous mode, `Engine.generate(...)` blocks until the first (and for non‑stream, final) item is yielded; in streaming mode it returns a Python iterator that pulls the async generator chunk‑by‑chunk.
- Code: [python/sglang/srt/entrypoints/engine.py#L216-L229](python/sglang/srt/entrypoints/engine.py#L216-L229)

### Synchronous vs asynchronous vs streaming

- Synchronous non‑streaming:
  - User calls `Engine.generate(..., stream=False)`. Engine drives the event loop and returns a single dict with `text`, `output_ids`, and `meta_info`.
  - Code: [python/sglang/srt/entrypoints/engine.py#L227-L229](python/sglang/srt/entrypoints/engine.py#L227-L229)

- Synchronous streaming:
  - User calls `Engine.generate(..., stream=True)`. Engine returns a Python iterator. Each iteration invokes the async generator and yields incremental chunks until finish.
  - Code: [python/sglang/srt/entrypoints/engine.py#L216-L226](python/sglang/srt/entrypoints/engine.py#L216-L226)

- Asynchronous (awaitable or async generator):
  - `await Engine.async_generate(..., stream=False)` returns the final dict.
  - `async for chunk in await Engine.async_generate(..., stream=True)` yields chunks.
  - Code: [python/sglang/srt/entrypoints/engine.py#L231-L301](python/sglang/srt/entrypoints/engine.py#L231-L301)

### Single‑process vs multi‑process

- Main process: `Engine`, `TokenizerManager`, `TemplateManager`, tracing and metrics client side.
- Subprocesses: Scheduler(s) and DetokenizerManager. Model execution always occurs in a scheduler process, not in the main process.
- Data parallel: A controller process dispatches incoming tokenized requests to multiple scheduler groups.


## Where the Model Lives and Executes

### Model instantiation

- The `Scheduler` creates a `TpModelWorker`, which constructs a `ModelRunner` that loads/initializes the model, tokenizer (if needed), memory pools, distributed groups, attention backends, and CUDA/NPU/CPU graph runners.
  - Scheduler → worker: [python/sglang/srt/managers/scheduler.py#L340-L373](python/sglang/srt/managers/scheduler.py#L340-L373)
  - Worker → ModelRunner: [python/sglang/srt/managers/tp_worker.py#L120](python/sglang/srt/managers/tp_worker.py#L120)
  - ModelRunner init: [python/sglang/srt/model_executor/model_runner.py#L236](python/sglang/srt/model_executor/model_runner.py#L236)
  - Weight loading (format‑specific via `DefaultModelLoader`): [python/sglang/srt/model_loader/loader.py#L420](python/sglang/srt/model_loader/loader.py#L420)

The PyTorch `nn.Module` instance lives inside the scheduler subprocess and is not exposed directly to the main process.

### GPU execution path

- Prefill/Decode passes are scheduled in the scheduler event loops, which build `ForwardBatch`es and call `ModelRunner.forward(...)` and `ModelRunner.sample(...)`. PP/TP/EP/DP group comms and attention backends are configured here.
  - Event loops: [python/sglang/srt/managers/scheduler.py#L1018](python/sglang/srt/managers/scheduler.py#L1018), [python/sglang/srt/managers/scheduler.py#L1037](python/sglang/srt/managers/scheduler.py#L1037), [python/sglang/srt/managers/scheduler.py#L1065](python/sglang/srt/managers/scheduler.py#L1065)
  - Forward/sample: [python/sglang/srt/managers/tp_worker.py#L214](python/sglang/srt/managers/tp_worker.py#L214)

### Return path (tokens → text)

- The scheduler emits `BatchTokenIDOutput` with token ids and optional logprobs/hidden states to the detokenizer (`send_to_detokenizer`).
  - Code: [python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899)
- `DetokenizerManager` incrementally decodes to strings (`BatchStrOutput`) and pushes to the tokenizer manager.
  - Code: [python/sglang/srt/managers/detokenizer_manager.py#L227](python/sglang/srt/managers/detokenizer_manager.py#L227)
- `TokenizerManager` aggregates per‑RID state and yields to the Engine client.
  - Code: [python/sglang/srt/managers/tokenizer_manager.py#L1306](python/sglang/srt/managers/tokenizer_manager.py#L1306)


## Getting a Model Handle

### Supported control surfaces

Although the actual PyTorch model is in a scheduler subprocess, `Engine` exposes safe control APIs via the tokenizer manager and a small RPC path:

- Get server capacity/state: `Engine.get_server_info()`
  - Code: [python/sglang/srt/entrypoints/engine.py#L402-L412](python/sglang/srt/entrypoints/engine.py#L402-L412)
- Inspect or update weights:
  - `Engine.get_weights_by_name(...)` → copies of tensors (for testing/inspection).
    - Code: [python/sglang/srt/entrypoints/engine.py#L518-L524](python/sglang/srt/entrypoints/engine.py#L518-L524)
  - `Engine.update_weights_from_disk(...)`, `Engine.update_weights_from_tensor(...)`, `Engine.update_weights_from_distributed(...)`.
    - Code: [python/sglang/srt/entrypoints/engine.py#L497-L516](python/sglang/srt/entrypoints/engine.py#L497-L516), [python/sglang/srt/entrypoints/engine.py#L471-L495](python/sglang/srt/entrypoints/engine.py#L471-L495)
- LoRA management: `load_lora_adapter(...)`, `unload_lora_adapter(...)` (adapter lives next to the model in the scheduler).
  - Code: [python/sglang/srt/entrypoints/engine.py#L526-L548](python/sglang/srt/entrypoints/engine.py#L526-L548)
- Collective RPC to scheduler methods (e.g., checkpoint saving): `Engine.collective_rpc(...)`, `save_sharded_model(...)`, `save_remote_model(...)`.
  - Code: [python/sglang/srt/entrypoints/engine.py#L564-L571](python/sglang/srt/entrypoints/engine.py#L564-L571)

These are multi‑process hops; return values are copied/serialized back to the main process. There is no shared‑memory model object in the main process.

### What is not supported

- There is no direct handle to the `torch.nn.Module` model instance from `Engine`. All interactions occur via request/response or RPC APIs. If you need to call custom model methods, add a scheduler RPC handler and invoke it through `collective_rpc`.


## Examples

See the repo example for a complete batch run:
- [examples/runtime/engine/offline_batch_inference.py#L1](examples/runtime/engine/offline_batch_inference.py#L1)

Additional minimal examples are included here for quick reference.

- Synchronous batch (non‑streaming): `codex/examples/offline_generate_sync.py`
- Streaming (synchronous wrapper over async): `codex/examples/offline_generate_stream.py`
- Asynchronous usage: `codex/examples/offline_generate_async.py`


## References

- Engine entry: [python/sglang/srt/entrypoints/engine.py#L93](python/sglang/srt/entrypoints/engine.py#L93)
- Tokenizer manager: [python/sglang/srt/managers/tokenizer_manager.py#L146](python/sglang/srt/managers/tokenizer_manager.py#L146)
- Scheduler: [python/sglang/srt/managers/scheduler.py#L280](python/sglang/srt/managers/scheduler.py#L280)
- Detokenizer: [python/sglang/srt/managers/detokenizer_manager.py#L72](python/sglang/srt/managers/detokenizer_manager.py#L72)
- Model runner: [python/sglang/srt/model_executor/model_runner.py#L236](python/sglang/srt/model_executor/model_runner.py#L236)
- Server arguments: [python/sglang/srt/server_args.py#L178](python/sglang/srt/server_args.py#L178)
