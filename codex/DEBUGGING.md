# Debugging Guide: Subprocess Model Execution (Scheduler) and Engine

The model runs in a scheduler subprocess launched with Python’s `spawn` method. This guide shows practical ways to debug code paths inside the child process, set breakpoints, and reduce complexity to diagnose issues quickly.

## Quick Wins

- Run minimal config: `tp_size=1`, `pp_size=1`, `dp_size=1`, `device=cuda` (or `cpu`) to reduce concurrency.
- Force simpler kernels: set `attention_backend="torch_native"` or `disable_cuda_graph=True` for simpler stepping.
- Enable strict error surfacing: `CUDA_LAUNCH_BLOCKING=1` and `TORCH_SHOW_CPP_STACKTRACES=1`.
- Turn on traces/metrics: `enable_trace=True`, `enable_metrics=True`, `log_level="debug"`.

## Option A: Remote PDB in the Scheduler Subprocess

Because `Engine` uses `multiprocessing.set_start_method("spawn", force=True)`, direct `pdb.set_trace()` in the child may fight with the main process’ standard input. Use a socket‑based debugger.

1) Add a guarded remote‑pdb hook near the start of the scheduler process (temporary debugging change):

```python
# File: python/sglang/srt/managers/scheduler.py (inside run_scheduler_process)
import os
if os.getenv("SGLANG_REMOTE_PDB") == "1":
    import remote_pdb
    remote_pdb.RemotePdb("0.0.0.0", 4444).set_trace()
```

2) Launch your script with the env var:

```bash
SGLANG_REMOTE_PDB=1 python examples/runtime/engine/offline_batch_inference.py --model-path <model>
```

3) In another terminal, connect to the waiting child:

```bash
nc 127.0.0.1 4444
```

Now you can `n`, `s`, `c` through scheduler code (including `ModelRunner.forward`). Remove the hook after debugging.

## Option B: debugpy Attach in Scheduler

1) Add a guarded debugpy block inside `run_scheduler_process` similarly:

```python
import os
if os.getenv("SGLANG_DEBUGPY") == "1":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("[scheduler] waiting for debugger attach at :5678")
    debugpy.wait_for_client()
```

2) Start the engine with `SGLANG_DEBUGPY=1 ...`, then attach from VS Code/PyCharm to port 5678.

This preserves TTY in the parent process while giving a rich debugger UI for the child.

## Option C: Drive ModelRunner Standalone

Use a single‑process harness to avoid subprocesses altogether and debug step‑by‑step.

- Script: `python/sglang/bench_one_batch.py` loads a model and runs prefill/decode using `ModelRunner` directly.
- Or adapt the low‑level example: `codex/examples/offline_generate_lowlevel.py` and place `pdb.set_trace()` where needed.

Pros: no subprocess I/O complications; easier to step into `forward()` and sampler.

## Breakpoints that Work Reliably

- In scheduler path (child): Add breakpoints inside:
  - `run_scheduler_process` [python/sglang/srt/managers/scheduler.py#L2989](python/sglang/srt/managers/scheduler.py#L2989)
  - `Scheduler.run_batch` [python/sglang/srt/managers/scheduler.py#L2180](python/sglang/srt/managers/scheduler.py#L2180)
  - `TpModelWorker.forward_batch_generation` [python/sglang/srt/managers/tp_worker.py#L214](python/sglang/srt/managers/tp_worker.py#L214)
  - `ModelRunner.forward` / `sample` [python/sglang/srt/model_executor/model_runner.py#L1860](python/sglang/srt/model_executor/model_runner.py#L1860)

- In tokenizer path (parent):
  - `TokenizerManager.generate_request` [python/sglang/srt/managers/tokenizer_manager.py#L370](python/sglang/srt/managers/tokenizer_manager.py#L370)
  - `TokenizerManager._handle_batch_output` [python/sglang/srt/managers/tokenizer_manager.py#L1306](python/sglang/srt/managers/tokenizer_manager.py#L1306)

## Logging and Crash Dumps

- Increase verbosity: set `log_level="debug"` in `ServerArgs` or call `configure_logger` with a debug level.
- Enable crash dumps in TM: set `crash_dump_folder`; on SIGTERM/SIGQUIT TM will dump unfinished requests [python/sglang/srt/managers/tokenizer_manager.py#L1427-L1512](python/sglang/srt/managers/tokenizer_manager.py#L1427-L1512)
- KV events: configure `kv_events_config` for detailed KV cache timelines.
- Tracing: `enable_trace=True` and point `oltp_traces_endpoint` to your collector to trace cross‑process slices.

## Kernel / CUDA Debugging Tips

- Set `CUDA_LAUNCH_BLOCKING=1` to force synchronous kernel launches (meaningful stack traces).
- Temporarily switch `attention_backend` to `torch_native` or `triton` and disable CUDA graphs for simpler call stacks.
- Use `deterministic_attention_backend` with `enable_deterministic_inference=True` to make issues reproducible.

