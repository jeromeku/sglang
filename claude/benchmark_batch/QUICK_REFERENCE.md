# SGLang Benchmark Quick Reference

## When to Use Which Benchmark?

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK SELECTION GUIDE                     │
└─────────────────────────────────────────────────────────────────┘

Need maximum throughput?
└─> bench_offline_throughput.py
    • Batch all requests together
    • No HTTP overhead
    • Best for: Batch processing, throughput benchmarking

Need realistic serving metrics?
└─> bench_serving.py
    • Poisson request arrival
    • Measures TTFT, ITL, latency percentiles
    • Best for: Production simulation, load testing

Need per-stage profiling (no server)?
└─> bench_one_batch.py
    • Direct ModelRunner API
    • Separate prefill/decode measurements
    • Best for: Model performance analysis, debugging

Need per-stage profiling (with server)?
└─> bench_one_batch_server.py
    • HTTP server with controlled batches
    • Server-side profiling
    • Best for: Server overhead analysis, production profiling
```

## Quick Command Reference

### 1. bench_offline_throughput.py
```bash
# Maximum throughput test
python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input 1024 \
    --random-output 256

# With Engine backend (default)
--backend engine

# With Runtime backend (HTTP client)
--backend runtime --base-url http://localhost:30000
```

### 2. bench_serving.py
```bash
# Realistic serving test
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input 1024 \
    --random-output 256 \
    --request-rate 10  # 10 req/s

# Burst traffic (all at once)
--request-rate inf

# With ShareGPT dataset
--dataset-name sharegpt \
--dataset-path sharegpt.json
```

### 3. bench_one_batch.py
```bash
# Latency profiling
python -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --batch-size 1 16 64 \
    --input-len 256 1024 \
    --output-len 32 256

# With profiling
--profile \
--profile-stage all \
--profile-activities CPU GPU

# Correctness test
--correctness-test
```

### 4. bench_one_batch_server.py
```bash
# Launch server and benchmark
python -m sglang.bench_one_batch_server \
    --model meta-llama/Meta-Llama-3.1-8B \
    --batch-size 1 16 64 \
    --input-len 1024 \
    --output-len 256

# Use existing server
--model None \
--base-url http://localhost:30000

# With profiling
--profile \
--profile-by-stage \
--show-report
```

## Key Metrics Explained

### Throughput Metrics
```
Request Throughput  = successful_requests / total_time (req/s)
Input Throughput    = total_input_tokens / total_time (tok/s)
Output Throughput   = total_output_tokens / total_time (tok/s)
Total Throughput    = (input + output tokens) / total_time (tok/s)
```

### Latency Metrics
```
TTFT (Time To First Token) = First token arrival - Request sent
ITL (Inter-Token Latency)  = Token_N arrival - Token_(N-1) arrival
E2E Latency                = Last token arrival - Request sent
```

### Prefill vs Decode
```
Prefill Phase:
  • Process all input tokens at once
  • Parallel computation (attention over all input)
  • Throughput = batch_size × input_len / prefill_time

Decode Phase:
  • Generate one token at a time
  • Sequential process
  • Throughput = batch_size / per_token_time
```

## Data Flow Summary

### Offline Mode (Engine)
```
Python API
    ↓
Engine.generate()
    ↓
TokenizerManager (main process)
    ↓ ZMQ
Scheduler (subprocess)
    ↓
ModelRunner.forward()
    ↓ GPU
Model computation
    ↓ ZMQ
DetokenizerManager (subprocess)
    ↓ ZMQ
TokenizerManager
    ↓
Python API (result)
```

### Server Mode (HTTP)
```
HTTP Request
    ↓
FastAPI Server
    ↓
Engine.async_generate()
    ↓
TokenizerManager (main process)
    ↓ ZMQ
Scheduler (subprocess)
    ↓
ModelRunner.forward()
    ↓ GPU
Model computation
    ↓ ZMQ
DetokenizerManager (subprocess)
    ↓ ZMQ
TokenizerManager
    ↓ async stream
FastAPI Server
    ↓ SSE
HTTP Response (streaming)
```

## Tensor Shape Reference

### Prefill Phase
```python
# Example: batch_size=2, seq_lens=[10, 15]

# Input
input_ids: [25]              # Concatenated: 10 + 15
positions: [25]              # [0..9, 0..14]
seq_lens: [2] = [10, 15]

# Model forward
logits: [25, vocab_size]     # All token positions

# Extract last token per request
next_token_logits: [2, vocab_size]

# Sample
next_token_ids: [2]
```

### Decode Phase
```python
# Example: batch_size=2, iteration 5

# Input
input_ids: [2]               # Last token from each request
positions: [2] = [14, 19]    # Current position
seq_lens: [2] = [15, 20]     # Updated lengths

# Model forward
logits: [2, vocab_size]

# Sample
next_token_ids: [2]
```

## Memory Allocation

```
GPU Memory Layout:
├── Model Weights (TP-sharded)
│   └── ~7GB for Llama-7B (fp16)
│
├── KV Cache Pool
│   ├── num_layers × 2 (K+V) × num_heads × head_dim
│   └── Dynamic allocation per request
│   └── Example: 32 layers × 2 × 32 heads × 128 dim = 262K per token
│
└── Activation Memory
    └── Temporary tensors during forward pass
```

## Common Issues & Solutions

### Issue: OOM (Out of Memory)
```bash
# Reduce batch size
--batch-size 1

# Reduce KV cache
--mem-fraction-static 0.7

# Use quantization
--load-format awq
```

### Issue: Low Throughput
```bash
# Increase batch size
--batch-size 64

# Enable CUDA graph
--disable-cuda-graph false

# Use tensor parallelism
--tp-size 2
```

### Issue: High Latency
```bash
# Reduce batch size
--batch-size 1

# Use chunked prefill
--chunked-prefill-size 512

# Check request rate
--request-rate 1  # Lower rate
```

## Performance Tuning Checklist

### For Maximum Throughput
- ✓ Use bench_offline_throughput.py
- ✓ Large batch sizes (64+)
- ✓ Disable streaming
- ✓ Enable CUDA graphs
- ✓ Use FP8/AWQ quantization

### For Minimum Latency
- ✓ Small batch sizes (1-8)
- ✓ Use tensor parallelism
- ✓ Enable chunked prefill
- ✓ Low request rate

### For Production Serving
- ✓ Use bench_serving.py
- ✓ Test with realistic request patterns
- ✓ Measure TTFT and ITL
- ✓ Monitor P99 latency
- ✓ Enable streaming

## File Locations

```
sglang/
├── python/sglang/
│   ├── bench_offline_throughput.py    # Offline throughput
│   ├── bench_serving.py               # Server with dynamic requests
│   ├── bench_one_batch.py             # Direct API latency
│   └── bench_one_batch_server.py      # Server latency
│
├── python/sglang/srt/
│   ├── entrypoints/
│   │   ├── engine.py                  # Engine implementation
│   │   └── http_server.py             # FastAPI server
│   │
│   ├── managers/
│   │   ├── tokenizer_manager.py       # Request tokenization
│   │   ├── scheduler.py               # Batch scheduling
│   │   └── detokenizer_manager.py     # Token to text
│   │
│   └── model_executor/
│       └── model_runner.py            # Model execution
│
└── claude/benchmark_batch/
    ├── README.md                      # Overview
    ├── 01_bench_offline_throughput.md # Detailed docs
    ├── 02_bench_serving.md
    ├── 03_bench_one_batch.md
    └── 04_bench_one_batch_server.md
```

## Result Files

### JSONL Format (append mode)
```json
{"run_name": "test", "batch_size": 16, "input_len": 1024, "prefill_latency": 0.234, "median_decode_latency": 0.012, ...}
{"run_name": "test", "batch_size": 32, "input_len": 1024, "prefill_latency": 0.345, ...}
```

### CSV Export
```bash
# Convert JSONL to CSV
cat result.jsonl | jq -r '[.batch_size, .input_len, .output_len, .prefill_latency, .median_decode_latency, .overall_throughput] | @csv' > results.csv
```

## Profiling

### Torch Profiler
```bash
# Set output directory
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles

# Run with profiling
python -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3-8B \
    --batch-size 1 \
    --input-len 256 \
    --profile \
    --profile-stage all

# View in Chrome
# 1. Open chrome://tracing
# 2. Load /tmp/profiles/profile_*.trace.json.gz
```

### CUDA Profiler (nsys)
```bash
nsys profile \
    --force-overwrite=true \
    -o bench_trace \
    python -m sglang.bench_one_batch \
        --model-path meta-llama/Meta-Llama-3-8B \
        --batch-size 1 \
        --input-len 256 \
        --profile \
        --profile-activities CUDA_PROFILER

# View with nsys-ui
nsys-ui bench_trace.nsys-rep
```

## Additional Resources

- [Main README](README.md) - Detailed overview
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - SGLang architecture
- [Server Args](../../python/sglang/srt/server_args.py) - Configuration options
- [Benchmark Utils](../../python/sglang/srt/utils/bench_utils.py) - Shared utilities
