# SGLang Architecture Walkthrough

This directory contains a comprehensive technical walkthrough of the SGLang Runtime (SRT) architecture, inspired by [Aleksa Gordić's vLLM article](https://www.aleksagordic.com/blog/vllm).

## Documents

### 1. [SGLang Architecture Deep Dive](./sglang_architecture_deep_dive.md)

The main walkthrough document covering:
- **Architecture Overview**: Multi-process design, ZMQ communication, request flow
- **Core Components**: Engine, Scheduler, ModelRunner, TpModelWorker
- **RadixCache**: Radix tree-based prefix caching with hierarchical storage
- **Request Scheduling**: LPM, DFS-Weight, FCFS, and in-batch prefix caching
- **Model Execution**: Forward pass pipeline, batch transformations
- **Attention Mechanisms**: RadixAttention, FlashInfer, paged attention
- **Multi-GPU**: Tensor/Pipeline/MoE/Data parallelism
- **Advanced Features**: Chunked prefill, speculative decoding, disaggregation, multimodal
- **Performance**: CUDA graphs, kernel optimizations, roofline analysis

### 2. [Code Examples & Implementation Details](./code_examples_detailed.md)

Detailed code walkthroughs including:
- **Batch Processing**: Request → ScheduleBatch → ModelWorkerBatch → ForwardBatch
- **Memory Management**: ReqToTokenPool, TokenToKVPool, block allocation
- **Attention Implementation**: FlashInfer backend, attention metadata
- **Scheduling Algorithms**: LPM implementation, in-batch optimization
- **CUDA Graph Capture**: Graph capture and replay process
- **Distributed Communication**: All-reduce, pipeline send/recv
- **Complete Request Trace**: End-to-end trace with actual data

### 3. [Quick Reference Guide](./quick_reference.md) *(this file)*

Quick reference for common operations and concepts.

---

## Key Architectural Concepts

### RadixCache: The Core Innovation

SGLang's main differentiator is its **RadixCache** - a radix tree that automatically reuses KV cache blocks across requests with shared prefixes.

```
Example:
Request 1: "What is the capital of France?"
Request 2: "What is the capital of Spain?"

RadixCache:
    root
     |
    "What is the capital"  ← Shared prefix (cached once)
     ├──── "of France?"
     └──── "of Spain?"
```

**Benefits:**
- Automatic prefix detection (no manual API calls needed)
- 50-80% memory reduction for shared-prefix workloads
- 1.5-3x throughput improvement
- Support for hierarchical storage (GPU → Host → Disk)

### Multi-Process Architecture

```
┌─────────────┐
│ HTTP Server │ (Main process)
└──────┬──────┘
       │ ZMQ
┌──────┴──────┐
│   Engine    │ (Main process)
└──┬──┬──┬────┘
   │  │  │ ZMQ sockets
   │  │  │
   ▼  ▼  ▼
┌───┐┌───┐┌────┐
│Tok││Sch││Detk│ (Subprocesses)
└───┘└─┬─┘└────┘
       │
       ▼
   ┌────────┐
   │GPU Exec│
   └────────┘
```

### Data Flow: Request Lifecycle

```
1. HTTP Request
2. Tokenization (TokenizerManager)
3. Scheduling (Scheduler picks from queue)
4. Prefix Matching (RadixCache finds reusable blocks)
5. KV Allocation (Allocate blocks for new tokens)
6. GPU Forward (ModelRunner executes)
7. Sampling (Select next token)
8. Detokenization (DetokenizerManager)
9. Response (Stream back to client)
```

### Batch Transformation Pipeline

```
ScheduleBatch (CPU, high-level)
    ↓
ModelWorkerBatch (CPU→GPU transition)
    ↓
ForwardBatch (GPU, low-level tensors)
```

---

## Key File Locations

### Entry Points
- [`engine.py`](../../python/sglang/srt/entrypoints/engine.py) - Main engine orchestration
- [`http_server.py`](../../python/sglang/srt/entrypoints/http_server.py) - FastAPI server

### Core Managers
- [`scheduler.py`](../../python/sglang/srt/managers/scheduler.py) - Request scheduling (2,696 lines)
- [`tokenizer_manager.py`](../../python/sglang/srt/managers/tokenizer_manager.py) - Async tokenization
- [`detokenizer_manager.py`](../../python/sglang/srt/managers/detokenizer_manager.py) - Async detokenization
- [`tp_worker.py`](../../python/sglang/srt/managers/tp_worker.py) - Tensor parallel worker

### Memory & Caching
- [`radix_cache.py`](../../python/sglang/srt/mem_cache/radix_cache.py) - Radix tree cache
- [`memory_pool.py`](../../python/sglang/srt/mem_cache/memory_pool.py) - KV cache pools
- [`allocator.py`](../../python/sglang/srt/mem_cache/allocator.py) - Memory allocators

### Model Execution
- [`model_runner.py`](../../python/sglang/srt/model_executor/model_runner.py) - Forward pass execution (2,856 lines)
- [`forward_batch_info.py`](../../python/sglang/srt/model_executor/forward_batch_info.py) - Batch data structures

### Attention
- [`radix_attention.py`](../../python/sglang/srt/layers/radix_attention.py) - Attention layer interface
- [`attention/`](../../python/sglang/srt/layers/attention/) - Backend implementations
  - `flashinfer_backend.py` - FlashInfer (default)
  - `flashattention_backend.py` - FlashAttention-2
  - `triton_backend.py` - Triton kernels

### Request Data
- [`schedule_batch.py`](../../python/sglang/srt/managers/schedule_batch.py) - Req, ScheduleBatch classes
- [`io_struct.py`](../../python/sglang/srt/managers/io_struct.py) - Input/output structures

### Scheduling
- [`schedule_policy.py`](../../python/sglang/srt/managers/schedule_policy.py) - Scheduling algorithms

### CUDA Kernels
- [`sgl-kernel/csrc/`](../../sgl-kernel/csrc/) - Custom CUDA kernels
  - `attention/` - Attention optimizations
  - `gemm/` - Matrix multiplication
  - `quantization/` - Quantization ops

---

## SGLang vs vLLM: Key Differences

| Feature | SGLang | vLLM |
|---------|--------|------|
| **Prefix Caching** | Automatic RadixCache with tree structure | PagedAttention with manual API |
| **Cache Eviction** | 6 policies (LRU, LFU, FIFO, MRU, FILO, Priority) | Limited eviction support |
| **Hierarchical Cache** | GPU → Host → Disk | GPU only |
| **Scheduling** | 5 policies (LPM, DFS-Weight, FCFS, LOF, Random) | FCFS primarily |
| **In-Batch Prefix** | Yes (automatic within waiting queue) | No |
| **Architecture** | Multi-process with ZMQ | Single process / multi-process |
| **Multimodal** | First-class (vision, audio, video) | Growing support |
| **Disaggregation** | Full P/D disaggregation with KV transfer | Limited |
| **LoRA Support** | Dynamic loading with batched inference | Static loading |
| **Grammar/FSM** | Integrated (xgrammar, outlines) | External tools |

### Performance Characteristics

**SGLang Advantages:**
- 1.5-3x higher throughput for shared-prefix workloads
- 50-80% memory reduction with prefix caching
- Better cache hit rates with LPM scheduling
- Superior multimodal performance

**vLLM Advantages:**
- More mature ecosystem and wider adoption
- Better documentation for beginners
- Simpler architecture (easier to understand)
- Broader model support out-of-the-box

---

## Quick Code Examples

### Starting the Server

```bash
# Basic usage
python -m sglang.launch_server --model-path meta-llama/Llama-3-8b --port 30000

# With prefix caching (default)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --schedule-policy lpm

# Multi-GPU (tensor parallelism)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-70b \
    --tp-size 8 \
    --port 30000

# Disaggregated prefill/decode
python -m sglang.launch_server \
    --model-path meta-llama/Llama-70b \
    --disagg-mode decode \
    --nccl-init-addr 192.168.1.1:50000
```

### Client Usage

```python
import openai

client = openai.Client(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Basic completion
response = client.completions.create(
    model="meta-llama/Llama-3-8b",
    prompt="What is the capital of France?",
    max_tokens=50,
)
print(response.choices[0].text)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.completions.create(
    model="meta-llama/Llama-3-8b",
    prompt="Tell me a story",
    max_tokens=200,
    stream=True,
):
    print(chunk.choices[0].text, end="", flush=True)

# Structured output (grammar)
response = client.completions.create(
    model="meta-llama/Llama-3-8b",
    prompt="Generate a JSON with name and age",
    max_tokens=100,
    extra_body={
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
)
```

### Direct Python API

```python
from sglang import Engine

# Initialize engine
engine = Engine(
    model_path="meta-llama/Llama-3-8b",
    tp_size=1,
    log_level="info",
)

# Generate
response = engine.generate(
    prompt="What is the capital of France?",
    sampling_params={
        "max_new_tokens": 50,
        "temperature": 0.7,
    }
)
print(response["text"])

# Batch generate
responses = engine.generate_batch([
    {"prompt": "What is 2+2?"},
    {"prompt": "What is the capital of Spain?"},
])

# Shutdown
engine.shutdown()
```

---

## Performance Tuning

### Memory Configuration

```bash
# Conservative (default)
--mem-fraction-static 0.8

# Aggressive (more cache, risk OOM)
--mem-fraction-static 0.95

# With hierarchical cache
--enable-hicache \
--hicache-cpu-gb 100 \
--hicache-disk-gb 500
```

### Scheduling Tuning

```bash
# Best for shared prefixes
--schedule-policy lpm

# Best for diverse requests
--schedule-policy fcfs

# Enable priority scheduling
--enable-priority-scheduling

# In-batch prefix caching threshold
export IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD=64
```

### Chunked Prefill

```bash
# Enable chunked prefill (default: 8192)
--chunked-prefill-size 4096

# Disable (better for throughput, worse for latency)
--chunked-prefill-size -1
```

### CUDA Graph

```bash
# Capture more batch sizes (slower startup, better runtime)
--cuda-graph-batch-sizes "1,2,4,8,16,32,64,128,256"

# Disable CUDA graph (for debugging)
--disable-cuda-graph
```

---

## Monitoring & Observability

### Metrics Endpoint

```bash
# Prometheus metrics
curl http://localhost:30000/metrics

# Key metrics:
# - sglang_request_latency_seconds
# - sglang_request_throughput
# - sglang_cache_hit_rate
# - sglang_kv_cache_usage_ratio
```

### Health Check

```bash
curl http://localhost:30000/health
```

### Get Model Info

```bash
curl http://localhost:30000/v1/models
```

### Internal State (Debug)

```bash
curl http://localhost:30000/get_internal_state
```

---

## Common Debugging Tips

### Enable Debug Logging

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --log-level debug
```

### Trace Request Flow

```bash
# Enable tracing
--enable-trace \
--otlp-traces-endpoint http://localhost:4318

# View with Jaeger or other OTLP-compatible tools
```

### Profile Performance

```bash
# Enable profiler
--enable-profiler

# Results saved to profiler_results/
```

### Check CUDA Memory

```python
import torch
print(torch.cuda.memory_allocated() / 1e9)  # GB
print(torch.cuda.memory_reserved() / 1e9)   # GB
```

---

## Advanced Topics

### Custom Scheduling Policy

Implement in [`schedule_policy.py`](../../python/sglang/srt/managers/schedule_policy.py):

```python
class MyCustomPolicy(CacheAwarePolicy):
    MY_POLICY = "my-policy"

    def calc_priority(self, waiting_queue):
        # Your logic here
        pass
```

### Custom Attention Backend

Implement in [`layers/attention/`](../../python/sglang/srt/layers/attention/):

```python
class MyAttnBackend(AttentionBackend):
    def forward(self, q, k, v, layer, forward_batch, save_kv_cache):
        # Your implementation
        pass
```

### LoRA Adapter Loading

```python
# Load adapter dynamically
response = client.completions.create(
    model="meta-llama/Llama-3-8b",
    prompt="Hello",
    extra_body={
        "lora_id": "my-adapter",
        "lora_path": "/path/to/adapter",
    }
)
```

---

## Further Reading

### Official Documentation
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Documentation](https://sgl-project.github.io/)
- [Examples](https://github.com/sgl-project/sglang/tree/main/examples)

### Papers
- [RadixAttention: Efficient LLM Serving with Prefix Caching](https://arxiv.org/abs/2312.07104)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)

### Related Systems
- [vLLM](https://github.com/vllm-project/vllm)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [LightLLM](https://github.com/ModelTC/lightllm)

---

## Contributing

See the main [SGLang repository](https://github.com/sgl-project/sglang) for contribution guidelines.

---

## Document Metadata

- **Version**: 1.0
- **Last Updated**: 2025-11-30
- **Maintainer**: Generated via Claude Code walkthrough
- **License**: Apache 2.0

---

## Questions?

For questions about SGLang architecture or implementation:
1. Check the detailed walkthroughs in this directory
2. Review the inline code documentation
3. Open an issue on GitHub
4. Join the SGLang community discussions
