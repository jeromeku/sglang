# Configuration Guide (ServerArgs) for Offline Engine

This guide summarizes the most impactful `ServerArgs` knobs for `sgl.Engine` offline inference and how they affect process layout, scheduling, memory, and performance. It is a companion to ARCHITECTURE.md.

Source: [python/sglang/srt/server_args.py#L178](python/sglang/srt/server_args.py#L178)


## Topology & Parallelism

- `tp_size`, `pp_size`, `dp_size`
  - Tensor, pipeline, and data parallel sizes. Influence number of scheduler processes and interconnect groups.
  - DP spawns a `DataParallelController` and multiple TP×PP groups.
- `nnodes`, `node_rank`, `dist_init_addr`
  - Multi‑node setup. Non‑zero nodes skip local TM/Detok and wait for schedulers.
- `tokenizer_worker_num`
  - >1 enables `MultiTokenizerRouter` instead of a single TM.
- `device`
  - `cuda`, `cpu`, `xpu`, `npu`, `hpu` supported. Sets distributed backend and attention choices.
- `enable_dp_attention`
  - Uses DP attention groups that change how work/control messages broadcast across ranks.


## Memory & Scheduling

- `max_total_tokens`, `max_running_requests`, `max_prefill_tokens`
  - Upper bounds for KV cache and concurrency; computed with model capacity at startup.
- `mem_fraction_static`
  - Fraction of total memory reserved for model weights and fixed buffers (auto‑inferred if None).
- `page_size`
  - KV cache paging granularity; 1 uses token‑granular allocator, otherwise paged.
- `schedule_policy`
  - `fcfs` (default), `lpm`, `dfs-weight`, `lof`, `random` (see `schedule_policy.py`).
- `enable_priority_scheduling`, `schedule_low_priority_values_first`, `priority_scheduling_preemption_threshold`
  - Enable per‑request priorities and allow preemption.
- `chunked_prefill_size`, `enable_mixed_chunk`
  - Prefill chunking and mixed prefill/decode batching.
- Hierarchical cache
  - `enable_hierarchical_cache`, `hicache_*` knobs to offload KV to host with different layouts and backends.


## Kernel / Attention / Graphs

- `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`
  - Choose per‑phase kernel backend (triton, flashinfer, fa3/fa4, flashmla, cutlass_mla, intel_amx, ascend, etc.).
- CUDA graphs
  - `disable_cuda_graph`, `cuda_graph_max_bs`, `cuda_graph_bs`, `piecewise_cuda_graph_max_tokens`, `piecewise_cuda_graph_tokens`.
- Deterministic inference
  - `enable_deterministic_inference`, `deterministic_attention_backend`.


## Model, Tokenizer, Quantization

- `model_path`, `revision`, `load_format`
  - Automatic or explicit loader selection (safetensors, pt, gguf, bitsandbytes, layered, remote, etc.).
- `tokenizer_path`, `tokenizer_mode`, `skip_tokenizer_init`
  - Hugging Face fast/slow modes; skip to reduce startup time if you provide `input_ids`.
- `quantization`, `kv_cache_dtype`, ModelOpt `modelopt_quant` and checkpoint (see loader code).


## Speculative / MoE / LoRA

- Spec decoding: `speculative_algorithm`, `speculative_draft_model_path`, `speculative_num_steps`, `speculative_accept_*`.
- MoE: `ep_size`, `moe_runner_backend`, `moe_a2a_backend`, DeepEP/Mooncake settings.
- LoRA: `enable_lora`, `lora_backend`, `lora_paths`, `max_loras_per_batch`, `max_loaded_loras`.


## Observability

- Logging: `log_level`, `log_requests`, `log_requests_level`.
- Metrics: `enable_metrics`, bucket settings, `tokenizer_metrics_allowed_custom_labels`.
- Tracing: `enable_trace`, `oltp_traces_endpoint`.


## Disaggregation & Bootstrap

- `disaggregation_mode`: `null` (default), `prefill`, `decode`.
- `disaggregation_transfer_backend`: `mooncake`, `nixl`, `ascend`, `fake`.
- Bootstrap ports / rooms are set automatically; can be overridden in calls.


## Safety Tips

- Many knobs interact (e.g., DP attention and graphs). Favor defaults, then change one axis at a time.
- On low HBM GPUs, reduce `cuda_graph_max_bs` and/or set `disable_cuda_graph=True` to avoid OOM.
- Enable `skip_tokenizer_init` when driving the engine with `input_ids` only.
