# Extending Offline Engine Components

This guide shows concrete ways to customize and extend internals used by `sgl.Engine` for offline generation: tokenization, scheduling, sampling, loaders, and detokenization. Each section references source locations and offers minimal code examples.


## Custom Tokenization / Preprocessing

Goals
- Provide your own text → ids conversion, or augment/validate inputs.
- Offload heavy preprocessing when using multimodal inputs.

Approaches
- Use `input_ids` directly in `GenerateReqInput` to bypass tokenizer.
- Wrap/monkey‑patch `TokenizerManager._tokenize_texts` / `_tokenize_one_request` for custom behavior.

References
- TM init and tokenization: [python/sglang/srt/managers/tokenizer_manager.py#L146](python/sglang/srt/managers/tokenizer_manager.py#L146), [python/sglang/srt/managers/tokenizer_manager.py#L474](python/sglang/srt/managers/tokenizer_manager.py#L474), [python/sglang/srt/managers/tokenizer_manager.py#L554](python/sglang/srt/managers/tokenizer_manager.py#L554)


## Scheduling Policy

Goals
- Customize fairness or KV cache locality (prefix caching) heuristics.

Approaches
- Choose built‑in policies via `ServerArgs.schedule_policy` (e.g., `fcfs`, `lpm`, `dfs-weight`, `lof`, `random`).
- Enable priority scheduling with `enable_priority_scheduling`, use `priority` per request, and set `schedule_low_priority_values_first`.
- Extend `SchedulePolicy` in-place (fork the module) or inject a new policy string and branch in `calc_priority`.

References
- Policy implementation: [python/sglang/srt/managers/schedule_policy.py#L1](python/sglang/srt/managers/schedule_policy.py#L1)
- Batch assembly and retract decode: [python/sglang/srt/managers/scheduler.py#L2180](python/sglang/srt/managers/scheduler.py#L2180), [python/sglang/srt/managers/scheduler.py#L2140](python/sglang/srt/managers/scheduler.py#L2140)


## Custom Logit Processor (Per‑request Sampling Hook)

Goals
- Modify logits (e.g., disallow tokens, temperature warping) before sampling.

Mechanism
- Implement a `CustomLogitProcessor` and pass it as a serialized string via `GenerateReqInput.custom_logit_processor`.
- Enable feature with `ServerArgs.enable_custom_logit_processor=True`.

Example
```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class ForceHello(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        # toy example: make token id 42 very likely
        logits[..., 42] = logits[..., 42] + 5.0
        return logits

processor_str = ForceHello.to_str()

output = llm.generate(
    "Say something",
    {"max_new_tokens": 32},
    custom_logit_processor=processor_str,
)
```

References
- Base class: [python/sglang/srt/sampling/custom_logit_processor.py#L20](python/sglang/srt/sampling/custom_logit_processor.py#L20)
- Application point: [python/sglang/srt/layers/sampler.py#L468](python/sglang/srt/layers/sampler.py#L468)


## Model Loaders & Quantization

Goals
- Introduce a new checkpoint format or loading scheme.

Approaches
- Implement a subclass of `BaseModelLoader` and make `LoadConfig.load_format` your type in a custom driver, or add a string choice wired in `get_model_loader(...)`.
- Use `ServerArgs.quantization`, ModelOpt (`modelopt_quant`) for vendor quantization.

References
- Base loader and registry: [python/sglang/srt/model_loader/loader.py#L253](python/sglang/srt/model_loader/loader.py#L253), [python/sglang/srt/model_loader/loader.py#L1917](python/sglang/srt/model_loader/loader.py#L1917)
- Model instantiation: [python/sglang/srt/model_loader/__init__.py#L20](python/sglang/srt/model_loader/__init__.py#L20)


## Attention Backends, CUDA Graphs, Determinism

Goals
- Change kernels and performance characteristics.

Approaches
- `ServerArgs.attention_backend` and friends (prefill/decode‑specific) choose kernels.
- `disable_cuda_graph`, `cuda_graph_max_bs` shape CUDA graph capture.
- `enable_deterministic_inference` + `deterministic_attention_backend` ensure repeatability.

References
- Selection logic: [python/sglang/srt/model_executor/model_runner.py#L540](python/sglang/srt/model_executor/model_runner.py#L540)
- CUDA graph runners: [python/sglang/srt/model_executor/cuda_graph_runner.py#L1](python/sglang/srt/model_executor/cuda_graph_runner.py#L1)


## LoRA Adapters

Goals
- Layer‑wise low‑rank adapters that can be hot‑loaded.

Approaches
- Start engine with `enable_lora=True`. Load/unload via `Engine.load_lora_adapter(...)` and `Engine.unload_lora_adapter(...)`.
- Schedulers validate lora batches against `max_loras_per_batch`.

References
- LoRA manager: [python/sglang/srt/model_executor/model_runner.py#L1290](python/sglang/srt/model_executor/model_runner.py#L1290)
- Engine surface: [python/sglang/srt/entrypoints/engine.py#L526-L548](python/sglang/srt/entrypoints/engine.py#L526-L548)


## Detokenizer

Goals
- Change how token→text streaming happens; trim stops differently; control `DecodeStatus` capacity.

Approaches
- Edit `trim_matched_stop` and streaming deltas in `DetokenizerManager`.
- Configure `SGLANG_DETOKENIZER_MAX_STATES` for high throughput streaming cases.

References
- Detok core: [python/sglang/srt/managers/detokenizer_manager.py#L46](python/sglang/srt/managers/detokenizer_manager.py#L46), [python/sglang/srt/managers/detokenizer_manager.py#L120-L189](python/sglang/srt/managers/detokenizer_manager.py#L120-L189)


## Tracing and Metrics

Goals
- Inspect per‑request latency, token timings, and system health.

Approaches
- Enable `enable_trace` and set `oltp_traces_endpoint`. Use built‑in Prometheus metrics via `enable_metrics`.

References
- Tracing setup: [python/sglang/srt/entrypoints/engine.py#L144-L148](python/sglang/srt/entrypoints/engine.py#L144-L148), [python/sglang/srt/managers/scheduler.py#L3031-L3036](python/sglang/srt/managers/scheduler.py#L3031-L3036)
- Metrics: [python/sglang/srt/managers/tokenizer_manager.py#L320-L336](python/sglang/srt/managers/tokenizer_manager.py#L320-L336), python/sglang/srt/managers/scheduler_metrics_mixin.py
