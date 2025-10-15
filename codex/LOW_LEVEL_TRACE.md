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
