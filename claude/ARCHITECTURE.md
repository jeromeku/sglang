# SGLang Offline Batch Inference Architecture

This document describes the architecture of SGLang's offline batch inference engine (`sgl.Engine`), tracing the complete initialization and execution path for model inference requests.

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Engine Initialization](#engine-initialization)
  - [Entry Point](#entry-point)
  - [Configuration Options](#configuration-options)
  - [Subprocess Launch Flow](#subprocess-launch-flow)
- [Process Architecture](#process-architecture)
  - [Main Process Components](#main-process-components)
  - [Scheduler Subprocess](#scheduler-subprocess)
  - [Detokenizer Subprocess](#detokenizer-subprocess)
  - [Inter-Process Communication](#inter-process-communication)
- [Request Execution Flow](#request-execution-flow)
  - [The generate() Call Stack](#the-generate-call-stack)
  - [Synchronous vs Asynchronous Execution](#synchronous-vs-asynchronous-execution)
  - [Data Flow Through Components](#data-flow-through-components)
- [Model Instantiation and Execution](#model-instantiation-and-execution)
  - [Model Loading](#model-loading)
  - [Forward Pass Execution](#forward-pass-execution)
  - [Accessing the Model](#accessing-the-model)
- [Key Data Structures](#key-data-structures)
- [Code Examples](#code-examples)

---

## Overview

SGLang's offline batch inference engine is designed as a **multi-process system** optimized for high-throughput LLM inference. The architecture separates concerns across three main components:

1. **TokenizerManager** (main process): Handles tokenization, request management, and response coordination
2. **Scheduler** (subprocess): Manages batching, scheduling, and model execution
3. **DetokenizerManager** (subprocess): Handles detokenization of output tokens

This separation enables efficient parallel processing while isolating the GPU-bound model execution in dedicated subprocesses.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Process                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    sgl.Engine                             │  │
│  │  - Provides synchronous API (generate(), encode())        │  │
│  │  - Wraps async operations with event loop                 │  │
│  └────────────────────┬──────────────────────────────────────┘  │
│                       │                                          │
│  ┌────────────────────▼──────────────────────────────────────┐  │
│  │              TokenizerManager                             │  │
│  │  - Tokenizes requests                                     │  │
│  │  - Manages request/response state                         │  │
│  │  - Handles multimodal preprocessing                       │  │
│  │  - Async request handling via asyncio                     │  │
│  └───────────────┬──────────────────────┬────────────────────┘  │
│                  │ (ZMQ)                │ (ZMQ)                  │
└──────────────────┼──────────────────────┼────────────────────────┘
                   │                      │
        ┌──────────▼──────────┐  ┌────────▼─────────────┐
        │  Scheduler Process  │  │ Detokenizer Process  │
        │  ┌──────────────┐   │  │  ┌───────────────┐   │
        │  │  Scheduler   │   │  │  │ Detokenizer   │   │
        │  │  - Batching  │   │  │  │  Manager      │   │
        │  │  - Scheduling│   │  │  │ - Decodes     │   │
        │  │  - KV Cache  │   │  │  │   token IDs   │   │
        │  └──────┬───────┘   │  │  │ - Incremental │   │
        │         │           │  │  │   decoding    │   │
        │  ┌──────▼───────┐   │  │  └───────────────┘   │
        │  │ ModelRunner  │   │  │                      │
        │  │ - GPU Model  │   │  └──────────────────────┘
        │  │ - Forward    │   │
        │  │ - Sampling   │   │
        │  └──────────────┘   │
        │                     │
        └─────────────────────┘
```

---

## Engine Initialization

### Entry Point

The engine is instantiated via the `sgl.Engine` class, which is a lazy import pointing to [`sglang.srt.entrypoints.engine.Engine`](python/sglang/srt/entrypoints/engine.py#L93).

**Location**: [python/sglang/\_\_init\_\_.py#L47](python/sglang/__init__.py#L47)

```python
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")
```

**Implementation**: [python/sglang/srt/entrypoints/engine.py#L93-L149](python/sglang/srt/entrypoints/engine.py#L93-L149)

```python
class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes requests and sends to scheduler
        2. Scheduler (subprocess): Receives requests, schedules batches, forwards,
           and sends output tokens to Detokenizer
        3. DetokenizerManager (subprocess): Detokenizes output tokens

    Note:
    1. HTTP server, Engine, and TokenizerManager all run in main process
    2. IPC is handled via ZMQ library, each process uses different port
    """
```

### Configuration Options

The engine accepts all configuration options from [`ServerArgs`](python/sglang/srt/server_args.py#L177-L299). Key configuration categories:

| Category | Key Options | Purpose |
|----------|-------------|---------|
| **Model & Tokenizer** | `model_path`, `tokenizer_path`, `load_format`, `trust_remote_code`, `context_length` | Model loading and tokenizer configuration |
| **Quantization** | `dtype`, `quantization`, `kv_cache_dtype` | Precision and quantization settings |
| **Memory & Scheduling** | `mem_fraction_static`, `max_running_requests`, `max_total_tokens`, `chunked_prefill_size` | Memory allocation and request scheduling |
| **Runtime** | `device`, `tp_size`, `pp_size`, `stream_interval` | Hardware configuration and parallelism |
| **Data Parallelism** | `dp_size`, `load_balance_method` | Data parallel configuration |
| **Logging** | `log_level`, `enable_metrics`, `show_time_cost` | Observability settings |

**Configuration Flow**:

1. **User provides kwargs** → [engine.py#L107-L120](python/sglang/srt/entrypoints/engine.py#L107-L120)
   ```python
   def __init__(self, **kwargs):
       if "server_args" in kwargs:
           server_args = kwargs["server_args"]
       else:
           if "log_level" not in kwargs:
               kwargs["log_level"] = "error"  # Default: suppress logs
           server_args = ServerArgs(**kwargs)
   ```

2. **Allocate IPC ports** → [engine.py#L126](python/sglang/srt/entrypoints/engine.py#L126)
   ```python
   self.port_args = PortArgs.init_new(server_args)
   ```

3. **Launch subprocesses** → [engine.py#L130-L133](python/sglang/srt/entrypoints/engine.py#L130-L133)
   ```python
   tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
       server_args=server_args,
       port_args=self.port_args,
   )
   ```

### Subprocess Launch Flow

The initialization follows this sequence:

**Step 1: Environment Setup** → [engine.py#L671-L734](python/sglang/srt/entrypoints/engine.py#L671-L734)

```python
def _set_envs_and_config(server_args: ServerArgs):
    # Set CUDA/NCCL environment variables
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"

    # Configure multiprocessing
    mp.set_start_method("spawn", force=True)

    # Register signal handlers for cleanup
    signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
```

**Step 2: Launch Scheduler Subprocesses** → [engine.py#L776-L821](python/sglang/srt/entrypoints/engine.py#L776-L821)

For single data parallel rank (`dp_size == 1`), launches one scheduler per TP/PP rank:

```python
for pp_rank in pp_rank_range:
    for tp_rank in tp_rank_range:
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = (server_args.base_gpu_id +
                  ((pp_rank % pp_size_per_node) * tp_size_per_node) +
                  (tp_rank % tp_size_per_node) * server_args.gpu_id_step)

        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank,
                  moe_ep_rank, pp_rank, None, writer),
        )
        proc.start()
        scheduler_procs.append(proc)
```

**Step 3: Launch Detokenizer Subprocess** → [engine.py#L856-L864](python/sglang/srt/entrypoints/engine.py#L856-L864)

```python
detoken_proc = mp.Process(
    target=run_detokenizer_process,
    args=(server_args, port_args),
)
detoken_proc.start()
```

**Step 4: Initialize TokenizerManager** → [engine.py#L866-L874](python/sglang/srt/entrypoints/engine.py#L866-L874)

```python
if server_args.tokenizer_worker_num > 1:
    tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
else:
    tokenizer_manager, template_manager = _init_tokenizer_manager(
        server_args, port_args
    )
```

**Step 5: Wait for Scheduler Ready Signal** → [engine.py#L876-L893](python/sglang/srt/entrypoints/engine.py#L876-L893)

```python
for i in range(len(scheduler_pipe_readers)):
    data = scheduler_pipe_readers[i].recv()  # Blocks until scheduler ready
    if data["status"] != "ready":
        raise RuntimeError("Initialization failed")
    scheduler_infos.append(data)

scheduler_info = scheduler_infos[0]
tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
```

---

## Process Architecture

### Main Process Components

**Engine** → [engine.py#L93](python/sglang/srt/entrypoints/engine.py#L93)
- Provides synchronous API wrapper
- Manages subprocess lifecycle
- Coordinates ZMQ communication

**TokenizerManager** → [tokenizer_manager.py#L145](python/sglang/srt/managers/tokenizer_manager.py#L145)
- Runs asyncio event loop
- Tokenizes input text
- Preprocesses multimodal data
- Manages request/response state
- Coordinates with scheduler and detokenizer

**Key Properties**:
- **Process Type**: Main process
- **Concurrency**: Asyncio-based (single-threaded async)
- **Communication**: ZMQ sockets to subprocesses

### Scheduler Subprocess

Launched via `run_scheduler_process()` → [scheduler.py#L2989-L3084](python/sglang/srt/managers/scheduler.py#L2989-L3084)

**Initialization Sequence**:

1. **Process configuration** → [scheduler.py#L3014-L3036](python/sglang/srt/managers/scheduler.py#L3014-L3036)
   ```python
   setproctitle.setproctitle(f"sglang::scheduler{prefix}")
   configure_logger(server_args, prefix=prefix)
   set_gpu_proc_affinity(...)
   ```

2. **Create Scheduler instance** → [scheduler.py#L3040-L3048](python/sglang/srt/managers/scheduler.py#L3040-L3048)
   ```python
   scheduler = Scheduler(
       server_args, port_args, gpu_id, tp_rank,
       moe_ep_rank, pp_rank, dp_rank
   )
   ```

3. **Send ready signal** → [scheduler.py#L3049-L3055](python/sglang/srt/managers/scheduler.py#L3049-L3055)
   ```python
   pipe_writer.send({
       "status": "ready",
       "max_total_num_tokens": scheduler.max_total_num_tokens,
       "max_req_input_len": scheduler.max_req_input_len,
   })
   ```

4. **Run event loop** → [scheduler.py#L3057-L3078](python/sglang/srt/managers/scheduler.py#L3057-L3078)
   ```python
   if disaggregation_mode == DisaggregationMode.NULL:
       if server_args.pp_size > 1:
           scheduler.event_loop_pp()
       elif scheduler.enable_overlap:
           scheduler.event_loop_overlap()
       else:
           scheduler.event_loop_normal()  # Most common path
   ```

**Scheduler Components**:

| Component | Purpose | Location |
|-----------|---------|----------|
| `Scheduler` | Request scheduling, batching, KV cache management | [scheduler.py](python/sglang/srt/managers/scheduler.py) |
| `ModelRunner` | Model loading, forward pass execution, sampling | [model_runner.py](python/sglang/srt/model_executor/model_runner.py) |
| `RadixCache` | Prefix cache for reusing KV cache across requests | [radix_cache.py](python/sglang/srt/mem_cache/radix_cache.py) |
| `ScheduleBatch` | Batch state and request management | [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) |

**Key Properties**:
- **Process Type**: Separate subprocess per TP/PP rank
- **Concurrency**: Synchronous event loop
- **GPU Binding**: Each subprocess bound to specific GPU
- **Model Location**: Model instantiated and runs in this process

### Detokenizer Subprocess

Launched via `run_detokenizer_process()` → [detokenizer_manager.py#L71](python/sglang/srt/managers/detokenizer_manager.py#L71)

**Purpose**: Detokenizes output token IDs incrementally and sends results back to TokenizerManager

**Key Properties**:
- **Process Type**: Single subprocess
- **Concurrency**: Synchronous polling loop
- **Communication**: Receives from Scheduler via ZMQ, sends to TokenizerManager

### Inter-Process Communication

**ZMQ Socket Configuration**:

```
┌─────────────────────┐
│ TokenizerManager    │
│  ┌──────────────┐   │
│  │ send_to_     │───┼──────┐
│  │ scheduler    │   │      │ PUSH → PULL
│  └──────────────┘   │      │
│                     │      ▼
│  ┌──────────────┐   │  ┌─────────────────┐
│  │ recv_from_   │◄──┼──│  Scheduler      │
│  │ detokenizer  │   │  │  ┌───────────┐  │
│  └──────────────┘   │  │  │ recv_from_│  │
└─────────────────────┘  │  │ tokenizer │  │
                         │  └───────────┘  │
         ▲               │  ┌───────────┐  │
         │               │  │ send_to_  │──┼──┐
         │ PUSH → PULL   │  │ detokeniz.│  │  │
         │               │  └───────────┘  │  │
         │               └─────────────────┘  │
    ┌────┴──────────────┐                    │
    │ DetokenizerManager│◄────────────────────┘
    │  ┌─────────────┐  │
    │  │ recv_from_  │  │
    │  │ scheduler   │  │
    │  └─────────────┘  │
    │  ┌─────────────┐  │
    │  │ send_to_    │  │
    │  │ tokenizer   │  │
    │  └─────────────┘  │
    └───────────────────┘
```

**Port Allocation** → [server_args.py](python/sglang/srt/server_args.py):
- IPC uses Unix domain sockets or TCP ports
- Allocated dynamically in `PortArgs.init_new()`
- Each subprocess gets unique communication channels

---

## Request Execution Flow

### The generate() Call Stack

**Complete execution path from user call to response:**

#### 1. User calls `engine.generate()`

**Location**: [engine.py#L150-L229](python/sglang/srt/entrypoints/engine.py#L150-L229)

```python
def generate(
    self,
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    # ... more parameters
) -> Union[Dict, Iterator[Dict]]:
    """Synchronous wrapper around async generate"""

    # Create GenerateReqInput object
    obj = GenerateReqInput(
        text=prompt,
        input_ids=input_ids,
        sampling_params=sampling_params,
        # ... more fields
    )

    # Get async generator from TokenizerManager
    loop = asyncio.get_event_loop()
    generator = self.tokenizer_manager.generate_request(obj, None)

    # Handle streaming vs non-streaming
    if stream:
        def generator_wrapper():
            while True:
                try:
                    chunk = loop.run_until_complete(generator.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        return generator_wrapper()
    else:
        ret = loop.run_until_complete(generator.__anext__())
        return ret
```

**Key Points**:
- `generate()` is **synchronous** wrapper
- Uses `asyncio.get_event_loop().run_until_complete()` to bridge sync/async
- Delegates to `TokenizerManager.generate_request()`

#### 2. TokenizerManager processes request

**Location**: [tokenizer_manager.py#L397-L414](python/sglang/srt/managers/tokenizer_manager.py#L397-L414)

```python
async def generate_request(
    self, obj: Union[GenerateReqInput, EmbeddingReqInput], request
):
    """Main async request handler"""
    created_time = time.time()

    async with self.model_update_lock.reader_lock:
        # Handle LoRA if needed
        if self.server_args.enable_lora and obj.lora_path:
            obj.lora_id = await self.lora_registry.acquire(obj.lora_path)

        if obj.is_single:
            # Single request path
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
        else:
            # Batch request path
            async for response in self._handle_batch_request(
                obj, request, created_time
            ):
                yield response
```

**Processing Steps**:

a. **Tokenization** (async) → [tokenizer_manager.py#L474-L549](python/sglang/srt/managers/tokenizer_manager.py#L474-L549)
   ```python
   async def _tokenize_texts(self, texts, is_cross_encoder=False):
       # Use async tokenizer if available for single requests
       if self.async_dynamic_batch_tokenizer and input_format == "single_string":
           result = await self.async_dynamic_batch_tokenizer.encode(text)
       else:
           # Fallback to synchronous tokenizer
           encoded = self.tokenizer(texts)
       return input_ids, token_type_ids
   ```

b. **Send to Scheduler** (sync ZMQ send)
   ```python
   def _send_one_request(self, obj, tokenized_obj, created_time):
       # Create request state
       state = ReqState(
           out_list=[], finished=False, event=asyncio.Event(),
           obj=obj, created_time=created_time
       )

       # Send tokenized request to scheduler via ZMQ
       self.send_to_scheduler.send_pyobj(tokenized_obj)

       # Track request state
       self.rid_to_state[tokenized_obj.rid] = state
       return state
   ```

c. **Wait for Response** (async) → [tokenizer_manager.py#L650-L750](python/sglang/srt/managers/tokenizer_manager.py#L650-L750)
   ```python
   async def _wait_one_response(self, obj, state, request):
       while True:
           # Wait for event from detokenizer
           await state.event.wait()
           state.event.clear()

           if state.finished:
               # Process final response
               if obj.stream:
                   yield self._prepare_stream_response(state, obj)
               yield self._prepare_final_response(state, obj)
               break
           elif obj.stream:
               # Yield intermediate streaming response
               yield self._prepare_stream_response(state, obj)
   ```

#### 3. Scheduler processes batch

**Event Loop** → [scheduler.py (event_loop_normal)](python/sglang/srt/managers/scheduler.py)

```python
def event_loop_normal(self):
    """Main scheduler event loop (synchronous)"""
    while True:
        # 1. Receive new requests from TokenizerManager
        recv_reqs = self.recv_requests()

        # 2. Process requests
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                self.handle_generate_request(recv_req)

        # 3. Schedule a batch for execution
        batch = self.get_next_batch_to_run()

        if batch:
            # 4. Execute forward pass via ModelRunner
            result = self.model_runner.forward(batch)

            # 5. Process model outputs
            self.process_batch_result(batch, result)

            # 6. Send outputs to Detokenizer
            self.send_to_detokenizer.send_pyobj(batch_output)
```

**Batching Strategy**:
- Combines pending requests into efficient batch
- Manages KV cache allocation
- Handles continuous batching (prefill + decode)

#### 4. ModelRunner executes forward pass

**Location**: [model_runner.py](python/sglang/srt/model_executor/model_runner.py)

```python
def forward(self, batch: ScheduleBatch) -> ForwardBatch:
    """Execute model forward pass"""

    # 1. Prepare input tensors
    forward_batch = ForwardBatch(
        input_ids=batch.input_ids,
        req_pool_indices=batch.req_pool_indices,
        # ... more batch metadata
    )

    # 2. Run model forward (GPU execution)
    if self.is_generation:
        logits = self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch
        )

    # 3. Sample next tokens
    next_token_ids = self.sampler(logits, forward_batch)

    return next_token_ids
```

**Model Location**: Model is instantiated in the **Scheduler subprocess**, bound to specific GPU.

#### 5. Detokenizer processes output

**Location**: [detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py)

```python
def event_loop(self):
    """Detokenizer event loop"""
    while True:
        # Receive output tokens from scheduler
        recv_obj = self.recv_from_scheduler.recv_pyobj()

        # Incrementally decode tokens
        for rid, new_tokens in recv_obj.items():
            state = self.decode_status[rid]
            state.decode_ids.extend(new_tokens)

            # Decode to text
            decoded_text = self.tokenizer.decode(
                state.decode_ids[state.read_offset:],
                skip_special_tokens=True
            )
            state.decoded_text += decoded_text
            state.read_offset = len(state.decode_ids)

        # Send decoded text to TokenizerManager
        self.send_to_tokenizer.send_pyobj(decoded_outputs)
```

#### 6. TokenizerManager returns response

Response flows back through the async generator to the user's synchronous `generate()` call.

### Synchronous vs Asynchronous Execution

| Component | Execution Model | Concurrency |
|-----------|----------------|-------------|
| `Engine.generate()` | **Synchronous** wrapper | Blocks on async operations via `run_until_complete()` |
| `TokenizerManager` | **Asynchronous** (asyncio) | Single-threaded async event loop |
| Scheduler subprocess | **Synchronous** event loop | Sequential batch processing |
| ModelRunner | **Synchronous** | GPU-bound operations |
| Detokenizer subprocess | **Synchronous** polling | Sequential token decoding |

**Key Insight**: Only TokenizerManager uses asyncio. Scheduler and Detokenizer use traditional synchronous event loops with ZMQ polling.

### Data Flow Through Components

```
User Request
    │
    ├─→ Engine.generate() [SYNC]
    │       │
    │       ├─→ TokenizerManager.generate_request() [ASYNC]
    │       │       │
    │       │       ├─→ Tokenize input [ASYNC]
    │       │       │
    │       │       ├─→ Send to Scheduler [SYNC ZMQ]
    │       │       │       │
    │       │       │       └──→ Scheduler subprocess [SYNC]
    │       │       │                   │
    │       │       │                   ├─→ Batch requests
    │       │       │                   │
    │       │       │                   ├─→ ModelRunner.forward()
    │       │       │                   │       │
    │       │       │                   │       └─→ Model (GPU) [SYNC]
    │       │       │                   │
    │       │       │                   ├─→ Send tokens to Detokenizer [ZMQ]
    │       │       │                   │
    │       │       │       ┌───────────┘
    │       │       │       │
    │       │       │       └──→ Detokenizer subprocess [SYNC]
    │       │       │                   │
    │       │       │                   ├─→ Decode tokens to text
    │       │       │                   │
    │       │       ├─→ Receive from Detokenizer [ZMQ]
    │       │       │
    │       │       └─→ Yield response [ASYNC]
    │       │
    │       └─→ return response [SYNC]
    │
    └─→ User receives response
```

---

## Model Instantiation and Execution

### Model Loading

The model is loaded in the **Scheduler subprocess** during `ModelRunner` initialization.

**Location**: [model_runner.py#L238-L299](python/sglang/srt/model_executor/model_runner.py#L238-L299)

```python
class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        # ... more params
    ):
        # Set GPU device
        self.device = server_args.device
        self.gpu_id = gpu_id

        # Initialize distributed environment
        self.init_torch_distributed()

        # Load model
        self.load_model()

        # Initialize memory pools and KV cache
        self.init_memory_pool()
```

**Model Loading Sequence**:

1. **Initialize distributed environment** → [model_runner.py#L676-L750](python/sglang/srt/model_executor/model_runner.py#L676-L750)
   ```python
   def init_torch_distributed(self):
       init_distributed_environment(
           world_size=self.tp_size * self.pp_size,
           rank=self.tp_rank + self.pp_rank * self.tp_size,
           distributed_init_method=f"tcp://{self.dist_init_addr}:{self.dist_port}",
           backend="nccl",
       )

       initialize_model_parallel(
           tensor_model_parallel_size=self.tp_size,
           pipeline_model_parallel_size=self.pp_size,
       )
   ```

2. **Load model from disk** → [model_runner.py (load_model)](python/sglang/srt/model_executor/model_runner.py)
   ```python
   def load_model(self):
       with set_default_torch_dtype(self.dtype):
           # Get model architecture
           model = get_model(
               model_config=self.model_config,
               load_config=self.load_config,
               # ...
           )

           # Load weights
           model_loader = get_model_loader(self.load_config)
           model_loader.load_model(
               model=model,
               model_config=self.model_config,
               # ...
           )

       self.model = model.eval()
   ```

3. **Initialize GPU memory pools** → [model_runner.py (init_memory_pool)](python/sglang/srt/model_executor/model_runner.py)
   ```python
   def init_memory_pool(self):
       # Calculate available GPU memory
       available_gpu_memory = get_available_gpu_memory(
           self.gpu_id,
           self.mem_fraction_static
       )

       # Allocate KV cache memory pool
       self.req_to_token_pool = ReqToTokenPool(...)
       self.token_to_kv_pool = MHATokenToKVPool(...)
   ```

### Forward Pass Execution

**Call Path**: `Scheduler.event_loop_normal()` → `ModelRunner.forward()` → `model.forward()`

**Location**: [model_runner.py (forward method)](python/sglang/srt/model_executor/model_runner.py)

```python
def forward(self, batch: ScheduleBatch) -> LogitsProcessorOutput:
    """Execute model forward pass on batch"""

    # 1. Prepare forward batch metadata
    forward_batch = ForwardBatch(
        input_ids=batch.input_ids,
        positions=batch.positions,
        # ... KV cache pointers, attention metadata
    )

    # 2. Run model forward (GPU computation)
    if self.is_generation:
        logits = self.model.forward(
            input_ids=forward_batch.input_ids,
            positions=forward_batch.positions,
            forward_batch=forward_batch,
        )

    # 3. Apply logits processors (temperature, top-p, etc.)
    logits_output = self.logits_processor(logits, forward_batch)

    # 4. Sample next tokens
    next_token_ids = self.sampler(
        logits_output.next_token_logits,
        forward_batch.sampling_info
    )

    return LogitsProcessorOutput(
        next_token_ids=next_token_ids,
        logprobs=logits_output.logprobs,
        # ...
    )
```

**Key Points**:
- Model runs on **GPU** in Scheduler subprocess
- Forward pass is **synchronous** (blocks until GPU completes)
- Batching enables efficient GPU utilization
- KV cache reduces recomputation of prefix tokens

### Accessing the Model

The model instance lives in the **Scheduler subprocess**, which is **separate from the main Engine process**.

**Direct Access**: ❌ **Not possible** - Model is in different process

**Indirect Access via RPC**: ✅ **Possible** - Engine provides RPC mechanism

**Location**: [engine.py#L584-L595](python/sglang/srt/entrypoints/engine.py#L584-L595)

```python
def collective_rpc(self, method: str, **kwargs):
    """Execute RPC call on all scheduler processes"""
    obj = RpcReqInput(method=method, parameters=kwargs)
    self.send_to_rpc.send_pyobj(obj)
    recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
    assert isinstance(recv_req, RpcReqOutput)
    assert recv_req.success, recv_req.message

# Example: Save model
def save_sharded_model(self, **kwargs):
    self.collective_rpc("save_sharded_model", **kwargs)
```

**Available RPC Methods**:
- `save_remote_model()` - Save model to disk
- `save_sharded_model()` - Save model with sharding
- `get_weights_by_name(name)` - Retrieve specific model weights
- `update_weights_from_tensor()` - Update model weights
- `update_weights_from_disk()` - Load new weights from disk

**Example: Get Model Weights**:

```python
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B")

# Get weights for a specific layer
weights = engine.get_weights_by_name("model.layers.0.self_attn.q_proj.weight")

# Update model weights without restarting
new_weights = [("model.layers.0.mlp.gate_proj.weight", new_tensor)]
engine.update_weights_from_tensor(new_weights)
```

**Process Architecture Diagram**:

```
┌────────────────────────────────────────────┐
│         Main Process                       │
│  ┌──────────────────────────────────────┐  │
│  │  sgl.Engine                          │  │
│  │  - No direct model access            │  │
│  │  - Use RPC for model operations      │  │
│  └──────────────┬───────────────────────┘  │
│                 │ RPC via ZMQ               │
└─────────────────┼───────────────────────────┘
                  │
    ┌─────────────▼──────────────┐
    │  Scheduler Subprocess      │
    │  ┌──────────────────────┐  │
    │  │  ModelRunner         │  │
    │  │  ┌────────────────┐  │  │
    │  │  │  self.model    │◄─┼──┼─── Model lives here!
    │  │  │  (GPU bound)   │  │  │
    │  │  └────────────────┘  │  │
    │  └──────────────────────┘  │
    └────────────────────────────┘
```

---

## Key Data Structures

### GenerateReqInput

**Location**: [io_struct.py#L89-L149](python/sglang/srt/managers/io_struct.py#L89-L149)

User-facing request object containing all generation parameters:

```python
@dataclass
class GenerateReqInput(BaseReq):
    # Input
    text: Optional[Union[List[str], str]] = None
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    image_data: Optional[MultimodalDataInputFormat] = None

    # Sampling parameters
    sampling_params: Optional[Union[List[Dict], Dict]] = None

    # Output control
    return_logprob: Optional[Union[List[bool], bool]] = None
    stream: bool = False

    # Advanced options
    lora_path: Optional[str] = None
    custom_logit_processor: Optional[str] = None
```

### TokenizedGenerateReqInput

**Location**: [io_struct.py](python/sglang/srt/managers/io_struct.py)

Internal tokenized request sent to Scheduler:

```python
@dataclass
class TokenizedGenerateReqInput:
    rid: str  # Request ID
    input_ids: List[int]  # Tokenized input
    sampling_params: SamplingParams
    image_inputs: Optional[ImageInputs]
    lora_id: Optional[str]
    # ... more fields
```

### ScheduleBatch

**Location**: [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py)

Batch of requests scheduled for model execution:

```python
class ScheduleBatch:
    reqs: List[Req]  # Requests in batch
    req_to_token_pool: ReqToTokenPool  # Memory pool for tokens
    token_to_kv_pool: TokenToKVPool  # KV cache pool

    # Batch tensors
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    # ... more batch metadata
```

### ForwardBatch

**Location**: [forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py)

Model-level batch representation for forward pass:

```python
@dataclass
class ForwardBatch:
    input_ids: torch.Tensor  # [batch_size, seq_len]
    positions: torch.Tensor  # [batch_size, seq_len]
    req_pool_indices: torch.Tensor

    # Attention metadata
    attn_metadata: AttentionMetadata

    # Sampling info
    sampling_info: SamplingBatchInfo
```

---

## Code Examples

### Example 1: Basic Generation

```python
import sglang as sgl

# Initialize engine
engine = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B",
    tp_size=1,  # Tensor parallelism
    mem_fraction_static=0.9,  # GPU memory fraction
    log_level="info"
)

# Generate
response = engine.generate(
    prompt="What is the capital of France?",
    sampling_params={"temperature": 0.7, "max_new_tokens": 100}
)

print(response["text"])

# Cleanup
engine.shutdown()
```

**Execution Flow**:
1. `Engine.__init__()` spawns Scheduler and Detokenizer subprocesses
2. `engine.generate()` creates `GenerateReqInput`
3. Event loop runs `TokenizerManager.generate_request()` (async)
4. Request tokenized and sent to Scheduler via ZMQ
5. Scheduler batches request and runs `ModelRunner.forward()`
6. Model (GPU) generates tokens
7. Detokenizer decodes tokens to text
8. Response returns to user synchronously

### Example 2: Batch Generation

```python
# Batch prompts
prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye",
    "Translate to French: Thank you"
]

responses = engine.generate(
    prompt=prompts,
    sampling_params={"temperature": 0.0, "max_new_tokens": 50}
)

for resp in responses:
    print(resp["text"])
```

**Batching Behavior**:
- All prompts tokenized in parallel (async)
- Scheduler combines into single batch
- Single model forward pass processes all prompts
- Responses detokenized and returned together

### Example 3: Streaming Generation

```python
# Stream tokens as they're generated
stream = engine.generate(
    prompt="Write a story about a dragon:",
    sampling_params={"temperature": 0.8, "max_new_tokens": 200},
    stream=True
)

for chunk in stream:
    print(chunk["text"], end="", flush=True)
print()
```

**Streaming Flow**:
1. `stream=True` returns generator instead of final response
2. Each decoded token chunk yielded as soon as available
3. Detokenizer performs incremental decoding
4. TokenizerManager yields intermediate results via async generator

### Example 4: Accessing Model Weights

```python
# Get specific layer weights
weights = engine.get_weights_by_name(
    name="model.layers.0.self_attn.q_proj.weight",
    truncate_size=100  # Only show first 100 values
)
print(f"Weight shape: {weights['shape']}")
print(f"Weight values: {weights['weights'][:10]}")

# Update weights from new checkpoint
engine.update_weights_from_disk(
    model_path="/path/to/new/checkpoint",
    load_format="safetensors"
)
```

**RPC Mechanism**:
- `get_weights_by_name()` sends RPC request to Scheduler subprocess
- Scheduler accesses `ModelRunner.model` and extracts weights
- Weights serialized and returned via ZMQ
- Engine blocks until response received

### Example 5: Context Manager Usage

```python
# Automatic cleanup with context manager
with sgl.Engine(model_path="meta-llama/Llama-3.1-8B") as engine:
    response = engine.generate(
        prompt="Explain quantum computing",
        sampling_params={"max_new_tokens": 150}
    )
    print(response["text"])

# Subprocesses automatically killed on exit
```

**Cleanup Mechanism**:
- `__exit__()` calls `engine.shutdown()`
- `shutdown()` invokes `kill_process_tree()`
- All child processes (Scheduler, Detokenizer) terminated

---

## Summary

SGLang's offline batch inference engine is architected as a **multi-process system** with clear separation of concerns:

1. **Main Process** (Engine + TokenizerManager): Handles user API, tokenization, and request coordination via asyncio
2. **Scheduler Subprocess**: Manages batching, scheduling, KV cache, and executes the model on GPU
3. **Detokenizer Subprocess**: Performs incremental token decoding

**Key Architectural Decisions**:

- ✅ **Multi-process isolation**: Model in separate subprocess prevents GIL contention and enables clean GPU binding
- ✅ **Async tokenization**: Non-blocking tokenization in main process while scheduler runs forward passes
- ✅ **ZMQ IPC**: Efficient zero-copy inter-process communication
- ✅ **Continuous batching**: Dynamically batches prefill and decode requests for high throughput
- ✅ **RPC for model access**: Controlled model operations without direct memory sharing

This architecture achieves high throughput for batch inference while maintaining a simple synchronous API for users.

---

**References**:
- Main Engine: [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py)
- TokenizerManager: [python/sglang/srt/managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)
- Scheduler: [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)
- ModelRunner: [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)
- DetokenizerManager: [python/sglang/srt/managers/detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py)
- ServerArgs: [python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)
