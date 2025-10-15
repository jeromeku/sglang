# Low‑Level Traces: Offline Generation (sgl.Engine)

This document deconstructs the `sgl.Engine` high‑level API into the internal components and functions that actually run the request. It provides precise call stacks and file:line links for synchronous, streaming, and asynchronous usage, and shows how to drive the low‑level APIs directly for power users.

Conventions
- Paths are repository‑relative with VS Code‑style line anchors (e.g., `file.py#L123`).
- “TM” = TokenizerManager; “Sched” = Scheduler; “Detok” = DetokenizerManager.
- IPC is via ZeroMQ using sockets in `PortArgs`.


## Process Topology

- Main process: `Engine` + TM (+ TemplateManager)
- Subprocesses: one or more Schedulers (TP/PP/DP) + one Detokenizer
- DP controller process if `dp_size > 1`.

Initialization entry points
- Engine init: [python/sglang/srt/entrypoints/engine.py#L93](python/sglang/srt/entrypoints/engine.py#L93)
- Launch subprocesses: [python/sglang/srt/entrypoints/engine.py#L754](python/sglang/srt/entrypoints/engine.py#L754)
- TM init: [python/sglang/srt/managers/tokenizer_manager.py#L146](python/sglang/srt/managers/tokenizer_manager.py#L146)
- Scheduler run loop(s): [python/sglang/srt/managers/scheduler.py#L1018](python/sglang/srt/managers/scheduler.py#L1018), [python/sglang/srt/managers/scheduler.py#L1037](python/sglang/srt/managers/scheduler.py#L1037), [python/sglang/srt/managers/scheduler.py#L1065](python/sglang/srt/managers/scheduler.py#L1065)
- Detok event loop: [python/sglang/srt/managers/detokenizer_manager.py#L114-L120](python/sglang/srt/managers/detokenizer_manager.py#L114-L120)


## Core Data Structures

- Client request (untokenized): `GenerateReqInput` in [python/sglang/srt/managers/io_struct.py#L120](python/sglang/srt/managers/io_struct.py#L120)
- Tokenized request (TM → Sched): `TokenizedGenerateReqInput` in [python/sglang/srt/managers/io_struct.py#L579](python/sglang/srt/managers/io_struct.py#L579)
- Batched tokenized request: `BatchTokenizedGenerateReqInput` in [python/sglang/srt/managers/io_struct.py#L645](python/sglang/srt/managers/io_struct.py#L645)
- Scheduler → Detok (token IDs, stats): `BatchTokenIDOutput` in [python/sglang/srt/managers/io_struct.py#L807](python/sglang/srt/managers/io_struct.py#L807)
- Detok → TM (text chunks): `BatchStrOutput` in [python/sglang/srt/managers/io_struct.py#L887](python/sglang/srt/managers/io_struct.py#L887)


## Synchronous Non‑Streaming Trace

High‑level call
1) `Engine.generate(..., stream=False)` → [python/sglang/srt/entrypoints/engine.py#L150-L213](python/sglang/srt/entrypoints/engine.py#L150-L213)
   - Builds `GenerateReqInput` ([engine.py#L193](python/sglang/srt/entrypoints/engine.py#L193)).
   - Calls `TM.generate_request(...)` to get an async generator ([engine.py#L214](python/sglang/srt/entrypoints/engine.py#L214)).
   - Drives event loop to get first (and only) item ([engine.py#L227-L229](python/sglang/srt/entrypoints/engine.py#L227-L229)).

TokenizerManager
2) `TokenizerManager.generate_request(...)` → [python/sglang/srt/managers/tokenizer_manager.py#L370-L414](python/sglang/srt/managers/tokenizer_manager.py#L370-L414)
   - Normalize inputs/batch/sampling params (`obj.normalize_batch_and_arguments`) [io_struct.py#L312](python/sglang/srt/managers/io_struct.py#L312).
   - Single request path: `_tokenize_one_request` → builds `TokenizedGenerateReqInput` ([tokenizer_manager.py#L554-L610](python/sglang/srt/managers/tokenizer_manager.py#L554-L610)).
   - Send to Scheduler via ZMQ (`_send_one_request`) [tokenizer_manager.py#L812](python/sglang/srt/managers/tokenizer_manager.py#L812) → `send_pyobj` at [#L820](python/sglang/srt/managers/tokenizer_manager.py#L820).
   - Create `ReqState` to accumulate outputs and wait (`:821`).
   - Async wait loop yields output as Detok replies arrive (`_wait_one_response`) [python/sglang/srt/managers/tokenizer_manager.py#L850](python/sglang/srt/managers/tokenizer_manager.py#L850).

Scheduler ingress and batching
3) Receive on TP0/PP0; broadcast to other TP ranks if needed: [python/sglang/srt/managers/scheduler.py#L1200](python/sglang/srt/managers/scheduler.py#L1200).
   - Requests are demuxed into work/control and broadcasted across TP/DP groups (`:1242-1272`).
   - Dispatch to handler: `handle_generate_request(...)` `:1335` creates a runtime `Req` and enqueues it, with multimodal expansion, logprob start logic, grammar setup, etc. (`:1366-1479`).
4) Pick next batch to run: `get_next_batch_to_run()` `:2400` builds prefill/decoding batches, applies tree cache, mixed chunking, preemption for priority, etc.
5) Execute: `run_batch(...)` [#L2180](python/sglang/srt/managers/scheduler.py#L2180) forwards to GPU via `TpModelWorker.forward_batch_generation(...)` [python/sglang/srt/managers/tp_worker.py#L214](python/sglang/srt/managers/tp_worker.py#L214).
   - Ultimately calls `ModelRunner.forward(...)` and then `ModelRunner.sample(...)` [python/sglang/srt/model_executor/model_runner.py#L292-L299](python/sglang/srt/model_executor/model_runner.py#L292-L299).

Return path (tokens → text)
6) Scheduler packages token ids + stats to Detok: `send_to_detokenizer.send_pyobj(BatchTokenIDOutput(...))` [python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899).
7) Detok incrementally decodes and sends `BatchStrOutput` to TM: [python/sglang/srt/managers/detokenizer_manager.py#L227](python/sglang/srt/managers/detokenizer_manager.py#L227) → `event_loop()` [#L114-L120](python/sglang/srt/managers/detokenizer_manager.py#L114-L120).
8) TM aggregates, fills `meta_info`, streams chunks or final output: `_handle_batch_output(...)` [python/sglang/srt/managers/tokenizer_manager.py#L1306](python/sglang/srt/managers/tokenizer_manager.py#L1306).
9) Engine collects the single result and returns it to the caller: [engine.py#L227-L229](python/sglang/srt/entrypoints/engine.py#L227-L229).


## Synchronous Streaming Trace

Differences from non‑streaming are marked.

1) `Engine.generate(..., stream=True)` returns a Python generator ([engine.py#L216-L226](python/sglang/srt/entrypoints/engine.py#L216-L226)), whose `next()` repeatedly pulls TM’s async generator ([engine.py#L221-L223](python/sglang/srt/entrypoints/engine.py#L221-L223)).
2) TM sets `stream=True` in `TokenizedGenerateReqInput` and sends to Sched ([tokenizer_manager.py#L738](python/sglang/srt/managers/tokenizer_manager.py#L738)).
3) Sched produces partial outputs and streams token ids/logprobs to Detok as they are available (see `stream_output_tokens` path within [scheduler_output_processor_mixin.py#L899+](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899)).
4) Detok incremental decoding: maintains per‑RID `DecodeStatus` and returns only deltas per chunk ([detokenizer_manager.py#L136-L189](python/sglang/srt/managers/detokenizer_manager.py#L136-L189)).
5) TM collects deltas, appends state, and yields interim dicts until `finished_reason` is non‑None ([tokenizer_manager.py#L1360-L1399](python/sglang/srt/managers/tokenizer_manager.py#L1360-L1399)).
6) Engine’s wrapper yields each chunk to the user ([engine.py#L221-L224](python/sglang/srt/entrypoints/engine.py#L221-L224)).


## Asynchronous Trace

1) `await Engine.async_generate(..., stream=False)` → [python/sglang/srt/entrypoints/engine.py#L231-L301](python/sglang/srt/entrypoints/engine.py#L231-L301).
   - Builds `GenerateReqInput` ([#L276](python/sglang/srt/entrypoints/engine.py#L276)) and calls `TM.generate_request(...)` to get an async generator ([#L296](python/sglang/srt/entrypoints/engine.py#L296)).
   - For non‑stream: `await generator.__anext__()` returns final dict ([#L300-L301](python/sglang/srt/entrypoints/engine.py#L300-L301)).

2) `agen = await Engine.async_generate(..., stream=True)` returns the async generator unwrapped ([#L298-L299](python/sglang/srt/entrypoints/engine.py#L298-L299)).
   - Consume via `async for chunk in agen:` to receive incremental `BatchStrOutput` chunks as dicts.

Internals in TM/Sched/Detok are identical to the synchronous case.


## Driving Low‑Level APIs Directly

The `Engine` convenience wrapper does three things: builds `ServerArgs`, allocates `PortArgs`, and calls `_launch_subprocesses(...)` to get a TM instance and start Sched+Detok. You can do this yourself to gain tighter control over the system.

Minimal low‑level flow (single process main + subprocess Sched/Detok):

```python
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.managers.io_struct import GenerateReqInput
import asyncio

server_args = ServerArgs(model_path="meta-llama/Llama-3.1-8B-Instruct", log_level="error")
port_args = PortArgs.init_new(server_args)

# Launch TM (main process), Scheduler(s) and Detok (subprocesses)
tokenizer_manager, template_manager, sched_info = _launch_subprocesses(server_args, port_args)

# Build a low-level request
obj = GenerateReqInput(text="Hello, my name is", sampling_params={"temperature":0.7, "max_new_tokens":64}, stream=False)

# Drive the async generator directly
gen = tokenizer_manager.generate_request(obj, request=None)
result = asyncio.get_event_loop().run_until_complete(gen.__anext__())
print(result["text"])  # { text, output_ids, meta_info }
```

Streaming variant (pull chunks):

```python
obj = GenerateReqInput(text="Write a limerick.", sampling_params={"max_new_tokens":64}, stream=True)
agen = tokenizer_manager.generate_request(obj, request=None)
while True:
    try:
        chunk = asyncio.get_event_loop().run_until_complete(agen.__anext__())
        print(chunk.get("text", ""), end="", flush=True)
    except StopAsyncIteration:
        break
```

Asynchronous variant:

```python
obj = GenerateReqInput(text="Three uses of graphs:", sampling_params={"temperature":0.2}, stream=True)
agen = await tokenizer_manager.generate_request(obj, request=None)
async for chunk in agen:
    ...
```

Notes
- `_launch_subprocesses` is an internal helper in `engine.py`. It is stable within this repo but not considered a public API.
- To use DP/PP/TP, set `ServerArgs` accordingly; `_launch_subprocesses` handles process layout.
- For pre‑tokenized input, set `GenerateReqInput.input_ids` instead of `text`.
- For embeddings or multimodal, use `EmbeddingReqInput` or set `image_data`/`audio_data`/`video_data` fields.


## Internal Extension Points

Below are practical hooks for customizing behavior without forking high‑level APIs.

Tokenizer / Preprocessing
- Override `ServerArgs` tokenizer settings (`tokenizer_mode`, `skip_tokenizer_init`, `enable_tokenizer_batch_encode`).
- Extend or replace tokenization by wrapping TM methods like `_tokenize_texts` / `_tokenize_one_request` ([python/sglang/srt/managers/tokenizer_manager.py#L474](python/sglang/srt/managers/tokenizer_manager.py#L474), [python/sglang/srt/managers/tokenizer_manager.py#L554](python/sglang/srt/managers/tokenizer_manager.py#L554)), or feed `input_ids` directly.

Scheduling
- Choose a policy via `ServerArgs.schedule_policy` (fcfs/lpm/dfs-weight/lof/random). See [python/sglang/srt/managers/schedule_policy.py#L1](python/sglang/srt/managers/schedule_policy.py#L1).
- Priority scheduling knobs: `enable_priority_scheduling`, `schedule_low_priority_values_first`, and `priority_scheduling_preemption_threshold`.
- Custom policy logic lives in `SchedulePolicy.calc_priority(...)` and friends.

Sampling and logits
- Provide a `custom_logit_processor` (serialized) per request when `--enable-custom-logit-processor` is set.
  - Base interface: [python/sglang/srt/sampling/custom_logit_processor.py#L20](python/sglang/srt/sampling/custom_logit_processor.py#L20).
  - Processor is applied in `sampler.apply_custom_logit_processor(...)` [python/sglang/srt/layers/sampler.py#L468](python/sglang/srt/layers/sampler.py#L468).

Model / loaders / quantization
- Attention backends and deterministic knobs via `ServerArgs` (see `server_args.py`, section “Kernel backend” and “Deterministic ...”).
- To introduce a new load format/loader, implement a `BaseModelLoader` and pass a type into `LoadConfig.load_format`, or register choices and map them in `get_model_loader(...)` ([python/sglang/srt/model_loader/loader.py#L1917](python/sglang/srt/model_loader/loader.py#L1917)).

LoRA
- Enable with `enable_lora`; dynamically load/unload via TM/Engine methods that proxy to `ModelRunner.LoRAManager` ([python/sglang/srt/managers/tokenizer_manager.py#L300-L309](python/sglang/srt/managers/tokenizer_manager.py#L300-L309), [python/sglang/srt/model_executor/model_runner.py#L1290](python/sglang/srt/model_executor/model_runner.py#L1290)).

Detokenization
- Customize incremental decode in `DetokenizerManager` ([python/sglang/srt/managers/detokenizer_manager.py#L120-L189](python/sglang/srt/managers/detokenizer_manager.py#L120-L189)).

Tracing / Metrics
- Enable OpenTelemetry traces: `ServerArgs.enable_trace` + `oltp_traces_endpoint`.
- Turn on tokenizer/scheduler metrics and histograms via `enable_metrics` and bucket tunables.


## Component Internals (Deep Dive)

This section breaks down each component into subcomponents with precise hooks to tweak or extend behavior.

### TokenizerManager (TM)

- Construction and Sockets
  - Class: [python/sglang/srt/managers/tokenizer_manager.py#L146](python/sglang/srt/managers/tokenizer_manager.py#L146)
  - Sockets: `recv_from_detokenizer`, `send_to_scheduler` initialized at [#L255-L269](python/sglang/srt/managers/tokenizer_manager.py#L255-L269)
  - Multi-tokenizer routing switches socket choice ([#L260-L268](python/sglang/srt/managers/tokenizer_manager.py#L260-L268))

- Request Normalization (batch shape, parallel sampling, rid management)
  - Normalization pipeline: `GenerateReqInput.normalize_batch_and_arguments()` [python/sglang/srt/managers/io_struct.py#L286-L359](python/sglang/srt/managers/io_struct.py#L286-L359)
  - Key helpers: `_validate_inputs` [#L341](python/sglang/srt/managers/io_struct.py#L341), `_determine_batch_size` [#L355](python/sglang/srt/managers/io_struct.py#L355), `_handle_parallel_sampling` [#L377](python/sglang/srt/managers/io_struct.py#L377)
  - TM-side orchestration: `generate_request(...)` [python/sglang/srt/managers/tokenizer_manager.py#L370-L414](python/sglang/srt/managers/tokenizer_manager.py#L370-L414)

- Tokenization & SamplingParams
  - Batch and single tokenize: `_tokenize_texts` [#L474](python/sglang/srt/managers/tokenizer_manager.py#L474), `_tokenize_one_request` [#L554-L610](python/sglang/srt/managers/tokenizer_manager.py#L554-L610)
  - Build tokenized object (sets `SamplingParams`, validates): `_create_tokenized_object` [#L706-L740](python/sglang/srt/managers/tokenizer_manager.py#L706-L740)
  - Sampling params normalization/verify: `SamplingParams.normalize/verify` [python/sglang/srt/sampling/sampling_params.py]

- Multimodal Preprocessing
  - MM processor pipeline: [#L589-L605](python/sglang/srt/managers/tokenizer_manager.py#L589-L605) using `get_mm_processor`

- LoRA & Model Update Coordination
  - LoRA registry and acquire/release: [#L300-L309](python/sglang/srt/managers/tokenizer_manager.py#L300-L309), release in `_handle_batch_output` [#L1390-L1411](python/sglang/srt/managers/tokenizer_manager.py#L1390-L1411)
  - Update lock & pause: `model_update_lock`, `is_pause_cond` at [#L291-L299](python/sglang/srt/managers/tokenizer_manager.py#L291-L299)

- Batching Modes
  - Single send: `_send_one_request` [#L812-L824](python/sglang/srt/managers/tokenizer_manager.py#L812-L824)
  - Batched send: `_send_batch_request` (uses `BatchTokenizedGenerateReqInput`) [#L826-L849](python/sglang/srt/managers/tokenizer_manager.py#L826-L849)
  - Batch orchestration: `_handle_batch_request` [#L900-L1040](python/sglang/srt/managers/tokenizer_manager.py#L900-L1040)

- Result Dispatch & Streaming
  - Dispatcher and handle loop: [#L342-L368](python/sglang/srt/managers/tokenizer_manager.py#L342-L368), `handle_loop` [#L1318-L1325](python/sglang/srt/managers/tokenizer_manager.py#L1318-L1325)
  - Output assembly: `_handle_batch_output` [#L1306-L1399](python/sglang/srt/managers/tokenizer_manager.py#L1306-L1399)
  - Per-chunk streaming yield: `_wait_one_response` [#L850-L899](python/sglang/srt/managers/tokenizer_manager.py#L850-L899)

Tweak ideas
- Replace `_tokenize_texts` for custom tokenization, or set `input_ids` to bypass it.
- Inject per-request logic in `_create_tokenized_object` (e.g., set default `SamplingParams`).
- Wrap `_handle_batch_output` to augment `meta_info` or add metrics.

### Scheduler (Sched)

- Construction and Sockets
  - Class: [python/sglang/srt/managers/scheduler.py#L280](python/sglang/srt/managers/scheduler.py#L280)
  - ZMQ sockets setup and rank role: [#L340-L373](python/sglang/srt/managers/scheduler.py#L340-L373)
  - Tracing setup within subprocess: [#L3031-L3036](python/sglang/srt/managers/scheduler.py#L3031-L3036)

- Initialization: Model, Tokenizer, Grammar, Policy
  - `ModelConfig` from args, tokenizer init: [#L374-L438](python/sglang/srt/managers/scheduler.py#L374-L438)
  - Grammar backend: [#L648-L671](python/sglang/srt/managers/scheduler.py#L648-L671)
  - Schedule policy creation: [#L672-L693](python/sglang/srt/managers/scheduler.py#L672-L693) and policy impl [schedule_policy.py](python/sglang/srt/managers/schedule_policy.py#L1)

- Memory Pool & Caches
  - KV cache pools chosen based on arch/hybrid/mamba: `ModelRunner.init_memory_pool_and_cache()` (see [python/sglang/srt/model_executor/model_runner.py#L1600+](python/sglang/srt/model_executor/model_runner.py#L1600))
  - Request/token pool allocators selected in `ModelRunner` [#L1700+](python/sglang/srt/model_executor/model_runner.py#L1700)

- Ingress & Broadcast
  - `recv_requests()` pulls from tokenizer, splits work/control, broadcasts across TP/DP [#L1200-L1298](python/sglang/srt/managers/scheduler.py#L1200-L1298)
  - `process_input_requests()` dispatches to request handlers [#L1299-L1332](python/sglang/srt/managers/scheduler.py#L1299-L1332)

- Request Handling
  - `handle_generate_request(...)` builds runtime `Req`, validates, sets logprob regions, handles multimodal expansion [#L1335-L1479](python/sglang/srt/managers/scheduler.py#L1335-L1479)
  - Grammar queue readiness and prefill-only cases are staged before batching.

- Batch Assembly and Flow
  - `get_next_batch_to_run()` merges prefill with running decode batches, handles mixed chunked prefill, DP attention prep [#L2400-L2492](python/sglang/srt/managers/scheduler.py#L2400-L2492)
  - `update_running_batch(...)` retracts when out of KV memory, updates ratios [#L2140-L2199](python/sglang/srt/managers/scheduler.py#L2140-L2199)
  - Chunked prefill specifics and HiCache integration appear along this path.

- GPU Execution
  - Orchestrated by `run_batch(...)` [#L2180](python/sglang/srt/managers/scheduler.py#L2180), which invokes `TpModelWorker.forward_batch_generation(...)` [python/sglang/srt/managers/tp_worker.py#L214](python/sglang/srt/managers/tp_worker.py#L214)
  - Prefill-only fast path computes logprobs without full sampling.

- Output Processing & Streaming
  - Token ids/logits packaged to detok: `send_to_detokenizer(BatchTokenIDOutput)` [python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899](python/sglang/srt/managers/scheduler_output_processor_mixin.py#L899)

Tweak ideas
- Add a new scheduling policy branch in `SchedulePolicy` or adjust thresholds (e.g., in-batch prefix caching constants).
- Modify retract/merge rules in `update_running_batch` to tune decode packing.
- Enable deterministic inference or swap attention backend to aid reproducibility.

### TpModelWorker and ModelRunner

- TpModelWorker wrapper
  - Creates `ModelRunner`, holds tokenizer (for some models), exposes forward helpers [python/sglang/srt/managers/tp_worker.py#L95-L214](python/sglang/srt/managers/tp_worker.py#L95-L214)

- ModelRunner lifecycle
  - Init and device/distributed setup: [python/sglang/srt/model_executor/model_runner.py#L236-L353](python/sglang/srt/model_executor/model_runner.py#L236-L353)
  - Torch distributed + model parallel init: [#L700-L770](python/sglang/srt/model_executor/model_runner.py#L700-L770)
  - Weight loading via loader registry: [#L1004-L1200](python/sglang/srt/model_executor/model_runner.py#L1004-L1200), registry [python/sglang/srt/model_loader/loader.py#L1917](python/sglang/srt/model_loader/loader.py#L1917)
  - Attention/backend selection & CUDA graphs: [#L1200-L1540](python/sglang/srt/model_executor/model_runner.py#L1200-L1540)
  - KV memory pools and allocators: [#L1560-L1860](python/sglang/srt/model_executor/model_runner.py#L1560-L1860)
  - Forward & sampling: `forward(...)`, `sample(...)` [#L1860+](python/sglang/srt/model_executor/model_runner.py#L1860)
  - LoRA manager and dynamic load: [#L1290-L1340](python/sglang/srt/model_executor/model_runner.py#L1290-L1340)

Tweak ideas
- Insert hooks in `forward` to log intermediate tensors (rank-0 only) or enable `CUDA_LAUNCH_BLOCKING=1` for kernel-error clarity.
- Force `attention_backend="torch_native"` for simpler step-by-step debugging.
- Use `bench_one_batch.py` to drive `ModelRunner` standalone for iterative debugging.

### DetokenizerManager (Detok)

- Class and event loop
  - Class: [python/sglang/srt/managers/detokenizer_manager.py#L72](python/sglang/srt/managers/detokenizer_manager.py#L72)
  - Event loop: [#L114-L120](python/sglang/srt/managers/detokenizer_manager.py#L114-L120)

- Incremental Decoding Internals
  - `DecodeStatus` ring with `surr_offset`, `read_offset`, `sent_offset` [#L60-L70](python/sglang/srt/managers/detokenizer_manager.py#L60-L70)
  - Batch decode → strings; compute deltas; stop trimming: [#L138-L189](python/sglang/srt/managers/detokenizer_manager.py#L138-L189)

Tweak ideas
- Adjust `trim_matched_stop` and stop-trim options to change output shaping.
- Increase `SGLANG_DETOKENIZER_MAX_STATES` for large concurrent streaming workloads.
