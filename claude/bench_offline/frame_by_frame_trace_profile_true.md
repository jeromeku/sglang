# Frame-by-Frame Execution Trace: bench_offline_throughput.py with profile=True

This document provides a comprehensive, frame-by-frame trace of the `throughput_test_once` function when `profile=True`, tracing every nested function call to understand the complete execution flow.

## Table of Contents
1. [Entry Point: throughput_test_once](#entry-point-throughput_test_once)
2. [Profile Start Flow](#profile-start-flow)
3. [Generate Flow](#generate-flow)
4. [Profile Stop Flow](#profile-stop-flow)
5. [Monitor Trace File Flow](#monitor-trace-file-flow)
6. [Get Server Info Flow](#get-server-info-flow)
7. [Complete Call Graph](#complete-call-graph)

---

## Entry Point: throughput_test_once

**Location**: [python/sglang/bench_offline_throughput.py:210-295](../python/sglang/bench_offline_throughput.py#L210-L295)

### Frame 1: Function Entry
```python
def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[DatasetRow],
    ignore_eos: bool,
    extra_request_body: Dict,
    profile: bool,  # <- True in our trace
    return_logprob: bool = False,
    logprob_start_len: int = -1,
):
```

**Parameters**:
- `backend_name`: String identifier (typically "engine" or "runtime")
- `backend`: Instance of `Engine` or `Runtime` class
- `reqs`: List of `DatasetRow` objects with prompts and expected output lengths
- `profile`: **True** (enables profiling)

### Frame 2: Initialize Measurement Results
**Line**: [220-230](../python/sglang/bench_offline_throughput.py#L220-L230)

```python
measurement_results = {
    "backend": backend_name,
    "successful_requests": len(reqs),
    "total_latency": -1,
    "total_input_tokens": sum(r.prompt_len for r in reqs),
    "total_output_tokens": -1,
    "request_throughput": -1,
    "input_throughput": -1,
    "output_throughput": -1,
    "total_throughput": -1,
}
```

### Frame 3: Prepare Request Data
**Lines**: [232-241](../python/sglang/bench_offline_throughput.py#L232-L241)

```python
prompt = [r.prompt for r in reqs]
sampling_params = [
    {
        "temperature": 0,
        "max_new_tokens": r.output_len,
        "ignore_eos": ignore_eos,
        **extra_request_body,
    }
    for r in reqs
]
```

**Action**: Extracts prompts and creates sampling parameters for each request

---

## Profile Start Flow

### Frame 4: Profile Condition Check
**Lines**: [243-248](../python/sglang/bench_offline_throughput.py#L243-L248)

```python
if profile:
    assert (
        "SGLANG_TORCH_PROFILER_DIR" in os.environ
    ), "Please set SGLANG_TORCH_PROFILER_DIR."
    os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
    backend.start_profile()
```

**Actions**:
1. Check that `SGLANG_TORCH_PROFILER_DIR` environment variable is set
2. Create output directory if it doesn't exist
3. Call `backend.start_profile()`

### Frame 5: backend.start_profile() - Engine Entry
**Location**: [python/sglang/srt/entrypoints/engine.py:396-397](../python/sglang/srt/entrypoints/engine.py#L396-L397)

```python
def start_profile(self, **kwargs):
    self.loop.run_until_complete(self.tokenizer_manager.start_profile(**kwargs))
```

**Action**:
- Uses asyncio event loop to call `tokenizer_manager.start_profile()`
- This is a synchronous wrapper around an async function

### Frame 6: TokenizerManager.start_profile() - Communicator Setup
**Location**: [python/sglang/srt/managers/tokenizer_communicator_mixin.py:308-340](../python/sglang/srt/managers/tokenizer_communicator_mixin.py#L308-L340)

```python
async def start_profile(
    self: TokenizerManager,
    output_dir: Optional[str] = None,
    start_step: Optional[int] = None,
    num_steps: Optional[int] = None,
    activities: Optional[List[str]] = None,
    with_stack: Optional[bool] = None,
    record_shapes: Optional[bool] = None,
    profile_by_stage: bool = False,
    merge_profiles: bool = False,
    profile_prefix: Optional[str] = None,
):
    self.auto_create_handle_loop()
    env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
    with_stack = False if with_stack is False or env_with_stack is False else True
    env_record_shapes: bool = get_bool_env_var("SGLANG_PROFILE_RECORD_SHAPES", "true")
    record_shapes = (record_shapes is not False) and env_record_shapes
    req = ProfileReq(
        type=ProfileReqType.START_PROFILE,
        output_dir=output_dir,
        start_step=start_step,
        num_steps=num_steps,
        activities=activities,
        with_stack=with_stack,
        record_shapes=record_shapes,
        profile_by_stage=profile_by_stage,
        profile_id=str(time.time()),  # Unique timestamp-based ID
        merge_profiles=merge_profiles,
        profile_prefix=profile_prefix,
    )
    return await self._execute_profile(req)
```

**Actions**:
1. Creates async handle loop if needed
2. Reads environment variables for profiling settings:
   - `SGLANG_PROFILE_WITH_STACK` (default: true)
   - `SGLANG_PROFILE_RECORD_SHAPES` (default: true)
3. Creates `ProfileReq` object with type `START_PROFILE`
4. Generates unique `profile_id` using current timestamp
5. Calls `_execute_profile()`

### Frame 7: TokenizerManager._execute_profile()
**Location**: [python/sglang/srt/managers/tokenizer_communicator_mixin.py:347-351](../python/sglang/srt/managers/tokenizer_communicator_mixin.py#L347-L351)

```python
async def _execute_profile(self: TokenizerManager, req: ProfileReq):
    result = (await self.profile_communicator(req))[0]
    if not result.success:
        raise RuntimeError(result.message)
    return result
```

**Actions**:
1. Calls `profile_communicator` (a `_Communicator` instance) with the request
2. Waits for response from scheduler process
3. Checks success status and raises error if failed
4. Returns `ProfileReqOutput` result

### Frame 8: _Communicator.__call__() - IPC Send
**Location**: [python/sglang/srt/managers/tokenizer_communicator_mixin.py:83-116](../python/sglang/srt/managers/tokenizer_communicator_mixin.py#L83-L116)

```python
class _Communicator(Generic[T]):
    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Future] = deque()

    async def queueing_call(self, obj: T):
        # Queue management for sequential requests
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            self._sender.send_pyobj(obj)  # ZMQ IPC send to scheduler

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()  # Wait for response
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values
```

**Actions**:
1. Queues the request if another is in flight
2. Serializes `ProfileReq` object using pickle
3. Sends via ZeroMQ socket to scheduler subprocess
4. Creates event and waits for response
5. Returns result when received

**Key Points**:
- Uses ZeroMQ for inter-process communication
- Only 1 in-flight request at a time (queueing mode)
- Async/await for non-blocking operation

### Frame 9: Scheduler Process - Receive and Dispatch
**Location**: [python/sglang/srt/managers/scheduler.py:565](../python/sglang/srt/managers/scheduler.py#L565)

The scheduler process runs in a separate subprocess and has a dispatch table:

```python
# In Scheduler.__init__()
self.req_type_to_processor = dict(
    [
        # ... other handlers ...
        (ProfileReq, self.profile),
        # ... other handlers ...
    ]
)
```

When the `ProfileReq` is received, it's dispatched to `self.profile()`.

### Frame 10: Scheduler.profile() - Route to Mixin
**Location**: [python/sglang/srt/managers/scheduler_profiler_mixin.py:319-350](../python/sglang/srt/managers/scheduler_profiler_mixin.py#L319-L350)

```python
def profile(self, recv_req: ProfileReq):
    if recv_req.type == ProfileReqType.START_PROFILE:
        if recv_req.profile_by_stage or recv_req.start_step:
            return self.init_profile(
                recv_req.output_dir,
                recv_req.start_step,
                recv_req.num_steps,
                recv_req.activities,
                recv_req.with_stack,
                recv_req.record_shapes,
                recv_req.profile_by_stage,
                recv_req.profile_id,
                recv_req.merge_profiles,
                recv_req.profile_prefix,
            )
        else:
            self.init_profile(
                recv_req.output_dir,
                recv_req.start_step,
                recv_req.num_steps,
                recv_req.activities,
                recv_req.with_stack,
                recv_req.record_shapes,
                recv_req.profile_by_stage,
                recv_req.profile_id,
                recv_req.merge_profiles,
                recv_req.profile_prefix,
            )
            return self.start_profile()
    else:  # STOP_PROFILE
        return self.stop_profile()
```

**Actions**:
1. Checks `ProfileReqType` (START_PROFILE in this case)
2. Routes to `init_profile()`
3. If no delayed start, immediately calls `start_profile()`
4. Returns result to tokenizer manager

### Frame 11: SchedulerProfilerMixin.init_profile()
**Location**: [python/sglang/srt/managers/scheduler_profiler_mixin.py:46-100](../python/sglang/srt/managers/scheduler_profiler_mixin.py#L46-L100)

```python
def init_profile(
    self,
    output_dir: Optional[str],
    start_step: Optional[int],
    num_steps: Optional[int],
    activities: Optional[List[str]],
    with_stack: Optional[bool],
    record_shapes: Optional[bool],
    profile_by_stage: bool,
    profile_id: str,
    merge_profiles: bool = False,
    profile_prefix: str = "",
) -> ProfileReqOutput:
    if self.profile_in_progress:
        return ProfileReqOutput(
            success=False,
            message="Profiling is already in progress. Call /stop_profile first.",
        )

    self.profile_by_stage = profile_by_stage
    self.merge_profiles = merge_profiles

    if output_dir is None:
        output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
    if activities is None:
        activities = ["CPU", "GPU"]

    self.torch_profiler_output_dir = Path(output_dir).expanduser()
    self.torch_profiler_with_stack = with_stack
    self.torch_profiler_record_shapes = record_shapes
    self.profiler_activities = activities
    self.profile_id = profile_id
    self.profile_prefix = profile_prefix

    if start_step:
        self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

    if num_steps:
        self.profile_steps = num_steps
        if self.profile_by_stage:
            self.profiler_target_prefill_ct = num_steps
            self.profiler_target_decode_ct = num_steps
            self.profiler_prefill_ct = 0
            self.profiler_decode_ct = 0
        elif start_step:
            self.profiler_target_forward_ct = (
                self.profiler_start_forward_ct + num_steps
            )
        else:
            self.profiler_target_forward_ct = self.forward_ct + num_steps
    else:
        self.profiler_target_forward_ct = None

    return ProfileReqOutput(success=True, message="Succeeded")
```

**Actions**:
1. Check if profiling already in progress (error if true)
2. Set default output dir from environment or `/tmp`
3. Set default activities to `["CPU", "GPU"]`
4. Store profiling configuration in scheduler state:
   - `torch_profiler_output_dir`: Where traces will be saved
   - `profiler_activities`: What to profile (CPU, GPU, MEM, CUDA_PROFILER, RPD)
   - `profile_id`: Unique ID for this profile session
   - `with_stack`: Whether to capture Python stack traces
   - `record_shapes`: Whether to record tensor shapes
5. Configure step-based triggering if specified
6. Return success

### Frame 12: SchedulerProfilerMixin.start_profile()
**Location**: [python/sglang/srt/managers/scheduler_profiler_mixin.py:102-175](../python/sglang/srt/managers/scheduler_profiler_mixin.py#L102-L175)

```python
def start_profile(
    self, stage: Optional[ForwardMode] = None
) -> ProfileReqOutput | None:
    stage_str = f" for {stage.name}" if stage else ""
    logger.info(
        f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
    )

    activities = self.profiler_activities
    with_stack = self.torch_profiler_with_stack
    record_shapes = self.torch_profiler_record_shapes

    activity_map = {
        "CPU": torch.profiler.ProfilerActivity.CPU,
        "GPU": torch.profiler.ProfilerActivity.CUDA,
    }
    torchprof_activities = [
        activity_map[a] for a in activities if a in activity_map
    ]

    if "RPD" in activities:
        # ROCm Profiling Data (AMD GPUs)
        from rpdTracerControl import rpdTracerControl
        rpdTracerControl.skipCreate()
        self.rpd_profile_path = os.path.join(
            self.torch_profiler_output_dir,
            "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
        )
        # Setup RPD profiler...
        self.rpd_profiler = rpdTracerControl()
        self.rpd_profiler.setPythonTrace(True)
        self.rpd_profiler.start()
        self.rpd_profiler.rangePush("", "rpd profile range", "")
        self.profile_in_progress = True

    elif torchprof_activities:
        # Standard PyTorch profiler
        self.torch_profiler = torch.profiler.profile(
            activities=torchprof_activities,
            with_stack=with_stack if with_stack is not None else True,
            record_shapes=record_shapes if record_shapes is not None else False,
            on_trace_ready=(
                None
                if not _is_npu
                else torch_npu.profiler.tensorboard_trace_handler(
                    self.torch_profiler_output_dir
                )
            ),
        )
        self.torch_profiler.start()
        self.profile_in_progress = True

    if "MEM" in activities:
        # Memory profiling
        torch.cuda.memory._record_memory_history(max_entries=100000)
        self.profile_in_progress = True

    if "CUDA_PROFILER" in activities:
        # CUDA profiler (for use with nsys)
        torch.cuda.cudart().cudaProfilerStart()
        self.profile_in_progress = True

    return ProfileReqOutput(success=True, message="Succeeded")
```

**Actions**:
1. Log profiling start message
2. Map activity names to PyTorch profiler activities:
   - "CPU" → `torch.profiler.ProfilerActivity.CPU`
   - "GPU" → `torch.profiler.ProfilerActivity.CUDA`
3. Initialize profilers based on activities:
   - **RPD**: ROCm profiling for AMD GPUs
   - **Standard**: PyTorch profiler for CPU/GPU
   - **MEM**: CUDA memory profiler
   - **CUDA_PROFILER**: CUDA profiler (nsys integration)
4. Create `torch.profiler.profile` context:
   ```python
   torch.profiler.profile(
       activities=[CPU, CUDA],
       with_stack=True,
       record_shapes=False,
   )
   ```
5. Call `.start()` on profiler
6. Set `self.profile_in_progress = True`
7. Return success

**Key Points**:
- Profiler captures CPU and GPU activity
- Stack traces are recorded for attribution
- Profiler runs in background, capturing all operations
- This is a LOW-LEVEL system profiler, not application-level

### Frame 13: Response Flow Back to TokenizerManager
The `ProfileReqOutput` travels back through:
1. Scheduler sends response via ZMQ
2. `_Communicator.handle_recv()` receives it (called from TokenizerManager event loop)
3. Sets `_result_values` and signals `_result_event`
4. `_Communicator.queueing_call()` returns result
5. `_execute_profile()` checks success
6. `start_profile()` returns to Engine
7. Engine returns to benchmark code

---

## Generate Flow

### Frame 14: Start Timer and Call backend.generate()
**Lines**: [250-256](../python/sglang/bench_offline_throughput.py#L250-L256)

```python
st = time.perf_counter()
gen_out = backend.generate(
    prompt=prompt,
    sampling_params=sampling_params,
    return_logprob=return_logprob,
    logprob_start_len=logprob_start_len,
)
latency = time.perf_counter() - st
```

**Actions**:
1. Start high-precision timer
2. Call `backend.generate()` with all prompts
3. **PROFILER IS NOW RECORDING ALL OPERATIONS**
4. Measure total latency

### Frame 15: Engine.generate() - Entry
**Location**: [python/sglang/srt/entrypoints/engine.py:162-242](../python/sglang/srt/entrypoints/engine.py#L162-L242)

```python
def generate(
    self,
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    # ... many other parameters ...
) -> Union[Dict, Iterator[Dict]]:
    # Validate data parallel rank if enabled
    if self.server_args.enable_dp_attention:
        if data_parallel_rank is None:
            logger.debug("data_parallel_rank not provided, using default dispatch")
        # ... validation ...

    # Create GenerateReqInput object
    obj = GenerateReqInput(
        text=prompt,
        input_ids=input_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
        return_logprob=return_logprob,
        logprob_start_len=logprob_start_len,
        top_logprobs_num=top_logprobs_num,
        token_ids_logprob=token_ids_logprob,
        lora_path=lora_path,
        custom_logit_processor=custom_logit_processor,
        return_hidden_states=return_hidden_states,
        stream=stream,
        bootstrap_host=bootstrap_host,
        bootstrap_port=bootstrap_port,
        bootstrap_room=bootstrap_room,
        data_parallel_rank=data_parallel_rank,
        rid=rid,
    )
    generator = self.tokenizer_manager.generate_request(obj, None)

    if stream:
        def generator_wrapper():
            while True:
                try:
                    chunk = self.loop.run_until_complete(generator.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        return generator_wrapper()
    else:
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret
```

**Actions**:
1. Create `GenerateReqInput` data structure
2. Call `tokenizer_manager.generate_request()`
3. Since `stream=False` (default in benchmark), run async generator once
4. Return complete result

### Frame 16: TokenizerManager.generate_request()
**Location**: [python/sglang/srt/managers/tokenizer_manager.py:426-505](../python/sglang/srt/managers/tokenizer_manager.py#L426-L505)

This is a LARGE async function. Key steps:

```python
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    created_time = time.time()
    self.auto_create_handle_loop()
    obj.normalize_batch_and_arguments()

    # Tracing support
    external_trace_header = None
    if request:
        if "trace_context" in request.headers:
            trace_set_remote_propagate_context(request.headers["trace_context"])
        else:
            external_trace_header = extract_trace_headers(request.headers)

    # Multi-worker support
    if self.server_args.tokenizer_worker_num > 1:
        self._attach_multi_http_worker_info(obj)

    # Tracing
    if self.enable_trace:
        self._trace_request_start(obj, created_time, external_trace_header)

    # Logging
    if self.log_requests:
        max_length, skip_names, _ = self.log_request_metadata
        logger.info(
            f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
        )

    # Pause handling
    async with self.is_pause_cond:
        await self.is_pause_cond.wait_for(lambda: not self.is_pause)

    # Model update lock
    async with self.model_update_lock.reader_lock:
        # LoRA adapter handling (if enabled)
        if self.server_args.enable_lora and obj.lora_path:
            # ... complex LoRA loading logic ...
            obj.lora_id = await self.lora_registry.acquire(obj.lora_path)
        else:
            obj.lora_id = None

        # Token size validation and LoRA support checks
        # ... validation logic ...

        # Tokenization
        if isinstance(obj, GenerateReqInput):
            if obj.input_ids is None:
                # Tokenize text prompts
                if self.server_args.skip_tokenizer_init:
                    # ... handle pre-tokenized ...
                else:
                    obj.input_ids = self.tokenizer.encode_batch(obj.text)

            # Prepare tokenized request
            tokenized_obj = self._prepare_tokenized_generate_req(obj)
        else:
            # Embedding request
            tokenized_obj = self._prepare_tokenized_embedding_req(obj)

        # MAIN CALL: Send to scheduler and yield results
        is_single = obj.is_single
        async for out in self._generate_request_impl(
            obj, tokenized_obj, request, created_time
        ):
            if is_single and isinstance(out, list):
                yield out[0]
            else:
                yield out
```

**Actions**:
1. Record creation time
2. Normalize batch arguments
3. Handle tracing context
4. Wait if system is paused
5. Acquire model update read lock
6. Handle LoRA adapters if needed
7. **Tokenize prompts** (CPU-bound operation)
8. Call `_generate_request_impl()` which sends to scheduler
9. Yield results as they come back

**Key Points**:
- This is async generator function
- Tokenization happens HERE in tokenizer manager process
- Scheduler only receives token IDs, not text

### Frame 17-20: Request Processing in Scheduler
The tokenized requests are sent to the scheduler subprocess which:

1. **Receives TokenizedGenerateReqInput**
2. **Batches requests** into forward batches
3. **Schedules batching** based on:
   - Available KV cache memory
   - Batch size limits
   - Priority/fairness
4. **Executes forward passes** through model:
   - Prefill phase: Process input tokens
   - Decode phase: Generate output tokens
5. **Sends output tokens** back to detokenizer

**PROFILER CAPTURES**:
- Model forward passes
- Attention computations
- KV cache operations
- Tensor operations
- CUDA kernels
- Memory allocations

This is the CORE of what profiling captures!

### Frame 21: Results Return
Results flow back:
1. Scheduler sends tokens to detokenizer subprocess
2. Detokenizer converts tokens to text
3. Detokenizer sends to tokenizer manager
4. TokenizerManager yields result
5. Engine returns result
6. Benchmark receives `gen_out`

---

## Profile Stop Flow

### Frame 22: Profile Stop Trigger
**Lines**: [259-263](../python/sglang/bench_offline_throughput.py#L259-L263)

```python
if profile:
    dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
    known_files = set(os.listdir(dir))
    backend.stop_profile()
    monitor_trace_file(known_files, dir)
```

**Actions**:
1. Get profiler output directory
2. List existing files (to detect new trace files)
3. Call `backend.stop_profile()`
4. Monitor for trace file completion

### Frame 23: Engine.stop_profile()
**Location**: [python/sglang/srt/entrypoints/engine.py:399-400](../python/sglang/srt/entrypoints/engine.py#L399-L400)

```python
def stop_profile(self):
    self.loop.run_until_complete(self.tokenizer_manager.stop_profile())
```

### Frame 24: TokenizerManager.stop_profile()
**Location**: [python/sglang/srt/managers/tokenizer_communicator_mixin.py:342-345](../python/sglang/srt/managers/tokenizer_communicator_mixin.py#L342-L345)

```python
async def stop_profile(self: TokenizerManager):
    self.auto_create_handle_loop()
    req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
    return await self._execute_profile(req)
```

**Actions**:
1. Create `ProfileReq` with type `STOP_PROFILE`
2. Send via `_execute_profile()` → `profile_communicator`
3. Same IPC flow as start

### Frame 25: Scheduler.profile() - Stop Branch
**Location**: [python/sglang/srt/managers/scheduler_profiler_mixin.py:319](../python/sglang/srt/managers/scheduler_profiler_mixin.py#L319)

```python
def profile(self, recv_req: ProfileReq):
    if recv_req.type == ProfileReqType.START_PROFILE:
        # ... start logic ...
    else:  # STOP_PROFILE
        return self.stop_profile()
```

### Frame 26: SchedulerProfilerMixin.stop_profile()
**Location**: [python/sglang/srt/managers/scheduler_profiler_mixin.py:209-285](../python/sglang/srt/managers/scheduler_profiler_mixin.py#L209-L285)

```python
def stop_profile(
    self, stage: Optional[ForwardMode] = None
) -> ProfileReqOutput | None:
    if not self.profile_in_progress:
        return ProfileReqOutput(
            success=False,
            message="Profiling is not in progress. Call /start_profile first.",
        )

    self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)

    stage_suffix = f"-{stage.name}" if stage else ""
    logger.info("Stop profiling" + stage_suffix + "...")

    # Stop PyTorch profiler
    if self.torch_profiler is not None:
        self.torch_profiler.stop()
        if not _is_npu:
            # Build filename with rank information
            filename_parts = [self.profile_id, f"TP-{self.tp_rank}"]

            # Add other parallelism ranks if > 1
            if getattr(self, "dp_size", 1) > 1:
                filename_parts.append(f"DP-{getattr(self, 'dp_rank', 0)}")
            if getattr(self, "pp_size", 1) > 1:
                filename_parts.append(f"PP-{getattr(self, 'pp_rank', 0)}")
            if getattr(self, "moe_ep_size", 1) > 1:
                filename_parts.append(f"EP-{getattr(self, 'moe_ep_rank', 0)}")

            filename = "-".join(filename_parts) + stage_suffix + ".trace.json.gz"

            # Export Chrome trace format
            self.torch_profiler.export_chrome_trace(
                os.path.join(self.torch_profiler_output_dir, filename)
            )
        torch.distributed.barrier(self.cpu_group)  # Sync across ranks

    # Stop RPD profiler (AMD)
    if self.rpd_profiler is not None:
        self.rpd_profiler.rangePop()
        self.rpd_profiler.stop()
        self.rpd_profiler.flush()
        torch.distributed.barrier(self.cpu_group)
        if self.tp_rank == 0:
            from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace
            rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
        self.rpd_profiler = None
        self.rpd_profiler_path = None

    # Stop memory profiler
    if self.profiler_activities is not None and "MEM" in self.profiler_activities:
        memory_profile_path = os.path.join(
            self.torch_profiler_output_dir,
            str(time.time())
            + f"-TP-{self.tp_rank}-memory"
            + stage_suffix
            + ".pickle",
        )
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    # Stop CUDA profiler
    if "CUDA_PROFILER" in self.profiler_activities:
        torch.cuda.cudart().cudaProfilerStop()

    # Merge traces from all ranks if requested
    merge_message = self._merge_profile_traces()

    logger.info(
        "Profiling done. Traces are saved to: %s%s",
        self.torch_profiler_output_dir,
        merge_message,
    )

    # Clean up state
    self.torch_profiler = None
    self.profile_in_progress = False
    self.profiler_start_forward_ct = None

    return ProfileReqOutput(success=True, message="Succeeded")
```

**Actions**:
1. Verify profiling was in progress
2. Create output directory
3. Stop PyTorch profiler
4. Export trace in Chrome Trace Format:
   - Filename pattern: `{profile_id}-TP-{rank}[-DP-{rank}][-PP-{rank}].trace.json.gz`
   - Format: JSON with gzip compression
   - Can be opened in `chrome://tracing`
5. Barrier synchronization across distributed ranks
6. Stop AMD RPD profiler if active
7. Dump memory snapshot if memory profiling enabled
8. Stop CUDA profiler if enabled
9. Optionally merge traces from multiple ranks
10. Log completion message
11. Clean up state
12. Return success

**Output Files**:
- `{timestamp}-TP-0.trace.json.gz`: Main trace file
- `{timestamp}-TP-0-memory.pickle`: Memory snapshot (if enabled)
- Merged trace (if merge_profiles=True)

---

## Monitor Trace File Flow

### Frame 27: monitor_trace_file() - Wait for Write Completion
**Location**: [python/sglang/bench_offline_throughput.py:298-328](../python/sglang/bench_offline_throughput.py#L298-L328)

```python
def monitor_trace_file(known_files, directory, interval=1):
    print(f"Monitoring {directory} for new trace files...")

    while True:
        flag = False
        time.sleep(interval)  # Wait 1 second
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    # File still growing, keep waiting
                    previous_size = current_size
                else:
                    # File size stable, writing complete
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break
```

**Actions**:
1. Poll directory every 1 second
2. Detect new files (not in `known_files` set)
3. For each new file:
   - Check file size
   - Wait if size is still growing
   - Break when size is stable (write complete)
4. Return when all new files are complete

**Purpose**:
Ensure trace files are fully written before benchmark continues. Export is asynchronous and may take time for large traces.

---

## Get Server Info Flow

### Frame 28: Parse Output and Get Server Info
**Lines**: [265-268](../python/sglang/bench_offline_throughput.py#L265-L268)

```python
if backend_name == "runtime":
    gen_out = json.loads(gen_out)

server_info = backend.get_server_info()
```

### Frame 29: Engine.get_server_info()
**Location**: [python/sglang/srt/entrypoints/engine.py:417-426](../python/sglang/srt/entrypoints/engine.py#L417-L426)

```python
def get_server_info(self):
    internal_states = self.loop.run_until_complete(
        self.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(self.tokenizer_manager.server_args),
        **self.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }
```

**Actions**:
1. Get internal state from scheduler (async call)
2. Merge:
   - Server arguments
   - Scheduler info (set during init)
   - Internal states (from scheduler)
   - Version
3. Return combined dictionary

### Frame 30: TokenizerManager.get_internal_state()
**Location**: [python/sglang/srt/managers/tokenizer_communicator_mixin.py:667-673](../python/sglang/srt/managers/tokenizer_communicator_mixin.py#L667-L673)

```python
async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
    req = GetInternalStateReq()
    responses: List[GetInternalStateReqOutput] = (
        await self.get_internal_state_communicator(req)
    )
    # Many DP ranks
    return [res.internal_state for res in responses]
```

**Actions**:
1. Create `GetInternalStateReq`
2. Send via communicator to scheduler(s)
3. Receive responses (one per data-parallel rank)
4. Extract internal_state dicts
5. Return list

### Frame 31: Scheduler.get_internal_state()
**Location**: [python/sglang/srt/managers/scheduler.py:2358-2367](../python/sglang/srt/managers/scheduler.py#L2358-L2367)

```python
def get_internal_state(self, recv_req: GetInternalStateReq):
    ret = vars(get_global_server_args())
    ret["last_gen_throughput"] = self.last_gen_throughput
    ret["memory_usage"] = {
        "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
        "kvcache": round(
            self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
        ),
        "token_capacity": int(self.max_total_num_tokens),
    }
    return ret
```

**Actions**:
1. Get server args
2. Add `last_gen_throughput`: Most recent measured throughput (tokens/sec)
3. Add `memory_usage`:
   - `weight`: Model weight memory in GB
   - `kvcache`: KV cache memory usage in GB
   - `token_capacity`: Total token slots available
4. Return dict

### Frame 32: Extract Throughput Metric
**Lines**: [288-293](../python/sglang/bench_offline_throughput.py#L288-L293)

```python
if inspect.isawaitable(server_info):
    server_info = asyncio.run(server_info)

measurement_results["last_gen_throughput"] = server_info["internal_states"][0][
    "last_gen_throughput"
]
```

**Actions**:
1. Handle async server_info if needed
2. Extract `last_gen_throughput` from first internal state
3. Add to measurement results

**Note**: `last_gen_throughput` is scheduler's measured throughput during actual generation, distinct from benchmark's E2E throughput measurement.

---

## Frame 33: Calculate Final Metrics
**Lines**: [270-286](../python/sglang/bench_offline_throughput.py#L270-L286)

```python
measurement_results["total_latency"] = latency
measurement_results["total_output_tokens"] = sum(
    o["meta_info"]["completion_tokens"] for o in gen_out
)
measurement_results["request_throughput"] = (
    measurement_results["successful_requests"] / latency
)
measurement_results["input_throughput"] = (
    measurement_results["total_input_tokens"] / latency
)
measurement_results["output_throughput"] = (
    measurement_results["total_output_tokens"] / latency
)
measurement_results["total_throughput"] = (
    measurement_results["total_input_tokens"]
    + measurement_results["total_output_tokens"]
) / latency
```

**Metrics**:
- `total_latency`: Total wall-clock time (seconds)
- `total_output_tokens`: Sum of generated tokens across all requests
- `request_throughput`: Requests per second
- `input_throughput`: Input tokens per second
- `output_throughput`: Output tokens per second
- `total_throughput`: Total tokens (input + output) per second

### Frame 34: Return Results
**Line**: [295](../python/sglang/bench_offline_throughput.py#L295)

```python
return measurement_results
```

Benchmark complete! Profiling data saved to disk.

---

## Complete Call Graph

```
throughput_test_once()
├─ [PROFILING ENABLED CHECK]
├─ backend.start_profile()
│  └─ Engine.start_profile()
│     └─ loop.run_until_complete()
│        └─ TokenizerManager.start_profile()
│           ├─ auto_create_handle_loop()
│           ├─ get_bool_env_var("SGLANG_PROFILE_WITH_STACK")
│           ├─ get_bool_env_var("SGLANG_PROFILE_RECORD_SHAPES")
│           ├─ ProfileReq(type=START_PROFILE, profile_id=timestamp, ...)
│           └─ _execute_profile(req)
│              └─ profile_communicator(req)
│                 └─ _Communicator.queueing_call()
│                    ├─ _sender.send_pyobj(ProfileReq)  [ZMQ IPC]
│                    └─ await _result_event.wait()
│                       │
│                       │  [Scheduler Subprocess]
│                       ├─ Scheduler receives ProfileReq
│                       ├─ Scheduler.profile(recv_req)
│                       │  └─ SchedulerProfilerMixin.profile()
│                       │     ├─ init_profile(...)
│                       │     │  ├─ Set output_dir
│                       │     │  ├─ Set activities ["CPU", "GPU"]
│                       │     │  ├─ Store profile_id
│                       │     │  └─ Return ProfileReqOutput(success=True)
│                       │     └─ start_profile()
│                       │        ├─ Map activities to ProfilerActivity
│                       │        ├─ torch.profiler.profile(activities, with_stack, record_shapes)
│                       │        ├─ torch_profiler.start()
│                       │        ├─ torch.cuda.memory._record_memory_history()  [if MEM]
│                       │        ├─ torch.cuda.cudart().cudaProfilerStart()  [if CUDA_PROFILER]
│                       │        ├─ profile_in_progress = True
│                       │        └─ Return ProfileReqOutput(success=True)
│                       │
│                       └─ Scheduler sends ProfileReqOutput  [ZMQ IPC]
│                          └─ _Communicator receives response
│                             └─ _result_event.set()
├─ time.perf_counter()  [START TIMER]
├─ backend.generate(prompt, sampling_params, ...)
│  └─ Engine.generate()
│     ├─ GenerateReqInput(text=prompt, sampling_params=...)
│     └─ tokenizer_manager.generate_request(obj, None)
│        └─ loop.run_until_complete(generator.__anext__())
│           └─ TokenizerManager.generate_request()
│              ├─ auto_create_handle_loop()
│              ├─ obj.normalize_batch_and_arguments()
│              ├─ is_pause_cond.wait_for(lambda: not is_pause)
│              ├─ model_update_lock.reader_lock
│              ├─ [LoRA adapter loading if needed]
│              ├─ tokenizer.encode_batch(obj.text)  [TOKENIZATION]
│              ├─ _prepare_tokenized_generate_req(obj)
│              └─ _generate_request_impl(obj, tokenized_obj, ...)
│                 └─ [Send TokenizedGenerateReqInput to Scheduler]
│                    │
│                    │  [Scheduler Subprocess - PROFILER RECORDING]
│                    ├─ Scheduler.event_loop_normal()
│                    ├─ get_next_batch_to_run()
│                    ├─ run_batch()
│                    │  └─ tp_worker.forward_batch()
│                    │     └─ model_runner.forward()
│                    │        ├─ model.forward()  [Model computation]
│                    │        │  ├─ Attention layers [PROFILED]
│                    │        │  ├─ FFN layers [PROFILED]
│                    │        │  ├─ KV cache ops [PROFILED]
│                    │        │  └─ Sampling [PROFILED]
│                    │        └─ Return logits/tokens
│                    ├─ process_batch_result()
│                    └─ Send output tokens to Detokenizer
│                       └─ DetokenizerManager receives tokens
│                          └─ Detokenize to text
│                             └─ Send to TokenizerManager
│                                └─ Yield result
├─ time.perf_counter()  [END TIMER]
├─ latency = end - start
├─ backend.stop_profile()
│  └─ Engine.stop_profile()
│     └─ loop.run_until_complete()
│        └─ TokenizerManager.stop_profile()
│           ├─ ProfileReq(type=STOP_PROFILE)
│           └─ _execute_profile(req)
│              └─ profile_communicator(req)
│                 └─ [ZMQ IPC to Scheduler]
│                    │
│                    │  [Scheduler Subprocess]
│                    ├─ Scheduler.profile(recv_req)
│                    │  └─ SchedulerProfilerMixin.stop_profile()
│                    │     ├─ torch_profiler.stop()
│                    │     ├─ torch_profiler.export_chrome_trace(output_path)
│                    │     │  └─ Write: {profile_id}-TP-{rank}.trace.json.gz
│                    │     ├─ torch.distributed.barrier()
│                    │     ├─ torch.cuda.memory._dump_snapshot()  [if MEM]
│                    │     ├─ torch.cuda.cudart().cudaProfilerStop()  [if CUDA_PROFILER]
│                    │     ├─ _merge_profile_traces()  [if merge_profiles]
│                    │     ├─ profile_in_progress = False
│                    │     └─ Return ProfileReqOutput(success=True)
│                    │
│                    └─ [ZMQ IPC response]
├─ monitor_trace_file(known_files, dir)
│  ├─ time.sleep(interval)
│  ├─ os.listdir(dir) - known_files  [Detect new files]
│  ├─ For each new file:
│  │  ├─ os.path.getsize(file)  [Check size]
│  │  ├─ Wait if size growing
│  │  └─ Break when stable
│  └─ Return when all files complete
├─ backend.get_server_info()
│  └─ Engine.get_server_info()
│     ├─ tokenizer_manager.get_internal_state()
│     │  └─ get_internal_state_communicator(req)
│     │     └─ [ZMQ IPC to Scheduler]
│     │        ├─ Scheduler.get_internal_state()
│     │        │  ├─ Get server args
│     │        │  ├─ last_gen_throughput
│     │        │  ├─ memory_usage {weight, kvcache, token_capacity}
│     │        │  └─ Return dict
│     │        └─ [ZMQ IPC response]
│     └─ Merge server_args + scheduler_info + internal_states + version
├─ Calculate metrics
│  ├─ total_output_tokens
│  ├─ request_throughput
│  ├─ input_throughput
│  ├─ output_throughput
│  ├─ total_throughput
│  └─ last_gen_throughput
└─ Return measurement_results

OUTPUT FILES:
  {SGLANG_TORCH_PROFILER_DIR}/{timestamp}-TP-{rank}.trace.json.gz
```

---

## Summary: What Profiling Captures

When `profile=True`, the PyTorch profiler records:

### 1. **CPU Activities**
- Python function calls (if `with_stack=True`)
- C++ operations
- Memory allocations/deallocations
- Data transfers

### 2. **GPU Activities** (CUDA events)
- Kernel launches
- Kernel execution times
- Memory transfers (H2D, D2H, D2D)
- Synchronization events
- Stream operations

### 3. **Model Operations**
- Forward pass timing
- Attention computation (FlashAttention, etc.)
- Matrix multiplications (GEMM)
- Layer normalization
- Activation functions
- KV cache read/write
- Sampling operations

### 4. **Memory Events** (if `MEM` activity)
- CUDA memory allocations
- Memory frees
- Peak memory usage
- Memory fragmentation

### 5. **Distributed Operations**
- NCCL collectives (all-reduce, all-gather, etc.)
- P2P communication
- Barrier synchronizations

### Output Format
- **Chrome Trace Format** (`.trace.json.gz`)
- Can be visualized in `chrome://tracing`
- Shows timeline of all operations
- Includes stack traces for attribution
- Shows concurrency and bottlenecks

---

## Key Architectural Points

1. **Multi-Process Architecture**:
   - TokenizerManager (main process)
   - Scheduler (subprocess, runs model)
   - Detokenizer (subprocess)
   - Communication via ZeroMQ IPC

2. **Profiler Runs in Scheduler Process**:
   - Captures model forward passes
   - All CUDA operations
   - Critical path is in scheduler

3. **Async/Sync Boundary**:
   - Engine provides sync API
   - Wraps async TokenizerManager
   - Uses `loop.run_until_complete()`

4. **Profiler State**:
   - Stored in Scheduler instance
   - `profile_in_progress` flag
   - Unique `profile_id` per session

5. **Trace File Export**:
   - Asynchronous write
   - `monitor_trace_file()` waits for completion
   - Important for reproducibility

This completes the frame-by-frame trace of profiling in SGLang's offline throughput benchmark!
