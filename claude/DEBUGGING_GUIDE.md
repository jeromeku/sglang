# Debugging SGLang's Multi-Process Architecture

This guide explains how to debug SGLang's offline inference engine, which runs the model in a separate subprocess. We'll cover multiple debugging strategies, from simple logging to advanced debugger attachment.

## Table of Contents

- [The Challenge](#the-challenge)
- [Debugging Strategies](#debugging-strategies)
  - [1. Logging-Based Debugging](#1-logging-based-debugging)
  - [2. Print Debugging](#2-print-debugging)
  - [3. File-Based Debugging](#3-file-based-debugging)
  - [4. Remote Debugger (debugpy)](#4-remote-debugger-debugpy)
  - [5. PDB with Subprocess Attachment](#5-pdb-with-subprocess-attachment)
  - [6. Single Process Mode](#6-single-process-mode)
- [Common Debugging Scenarios](#common-debugging-scenarios)
- [Tips and Best Practices](#tips-and-best-practices)

---

## The Challenge

SGLang's architecture creates a fundamental debugging challenge:

```
┌─────────────────────────┐
│   Main Process          │
│   - You start here      │
│   - Your IDE/debugger   │
│     is attached here    │
└────────┬────────────────┘
         │ spawns
         │
         ▼
┌─────────────────────────┐
│   Scheduler Subprocess  │
│   - Model lives here    │
│   - Forward pass here   │
│   - Your breakpoint     │
│     WON'T work here!    │
└─────────────────────────┘
```

**Why standard debugging fails**:
1. Python debuggers (pdb, ipdb, VSCode) attach to the **main process**
2. The model runs in a **child subprocess** created via `multiprocessing.spawn`
3. Breakpoints in model code are **ignored** because the debugger doesn't control the subprocess
4. The subprocess has its own Python interpreter instance

**Location of subprocess creation**: [engine.py#L804-L820](python/sglang/srt/entrypoints/engine.py#L804-L820)

```python
proc = mp.Process(
    target=run_scheduler_process,  # This runs in a NEW process
    args=(server_args, port_args, gpu_id, tp_rank, ...),
)
proc.start()  # Debugger loses control here
```

---

## Debugging Strategies

### 1. Logging-Based Debugging

**Best for**: Understanding execution flow, tracking values, production debugging

The simplest and most reliable approach. SGLang already has comprehensive logging infrastructure.

#### Basic Logging

**Change log level**:

```python
import sglang as sgl

engine = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B",
    log_level="debug",  # Options: debug, info, warning, error
)
```

**Add custom logging in model code**:

```python
# In python/sglang/srt/model_executor/model_runner.py
import logging
logger = logging.getLogger(__name__)

def forward(self, batch: ScheduleBatch):
    logger.info(f"Forward pass: batch_size={len(batch.reqs)}")
    logger.debug(f"Input IDs shape: {batch.input_ids.shape}")
    logger.debug(f"Positions: {batch.positions}")

    # Your debugging code
    logits = self.model.forward(...)

    logger.info(f"Output logits shape: {logits.shape}")
    return logits
```

#### Structured Logging with Context

```python
# Add structured context to logs
logger.info(
    "Model forward",
    extra={
        "batch_size": len(batch.reqs),
        "input_shape": batch.input_ids.shape,
        "tp_rank": self.tp_rank,
        "gpu_id": self.gpu_id,
    }
)
```

#### Logging Configuration

SGLang configures logging in [utils.py (configure_logger)](python/sglang/srt/utils.py):

```python
def configure_logger(server_args: ServerArgs, prefix: str = ""):
    """Configure logger with appropriate handlers and formatters"""
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=f"%(asctime)s{prefix} [%(levelname)s] %(name)s: %(message)s",
    )
```

---

### 2. Print Debugging

**Best for**: Quick value inspection, temporary debugging

Works in subprocesses because print statements go to the **parent's stdout/stderr**.

```python
# In model_runner.py
def forward(self, batch: ScheduleBatch):
    print(f"[SCHEDULER PID={os.getpid()}] Forward batch_size={len(batch.reqs)}")
    print(f"[SCHEDULER] Input IDs: {batch.input_ids[:5]}")  # First 5 tokens

    logits = self.model.forward(...)

    print(f"[SCHEDULER] Output logits min/max: {logits.min()}/{logits.max()}")
    return logits
```

**Pro tip**: Add process ID and rank to identify output source:

```python
import os
prefix = f"[PID={os.getpid()} TP{self.tp_rank}]"
print(f"{prefix} Debug message")
```

---

### 3. File-Based Debugging

**Best for**: Inspecting tensors, saving intermediate values, crash dumps

Write debug information to files that you can inspect after execution.

#### Save Tensors

```python
import torch
import os

def forward(self, batch: ScheduleBatch):
    # Create debug directory
    debug_dir = "/tmp/sglang_debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Save input tensors
    torch.save(
        {
            "input_ids": batch.input_ids,
            "positions": batch.positions,
            "batch_size": len(batch.reqs),
        },
        f"{debug_dir}/forward_input_{self.forward_pass_id}.pt"
    )

    logits = self.model.forward(...)

    # Save output tensors
    torch.save(
        {"logits": logits},
        f"{debug_dir}/forward_output_{self.forward_pass_id}.pt"
    )

    self.forward_pass_id += 1
    return logits
```

#### Inspect Saved Tensors

```python
# In a separate script or notebook
import torch

data = torch.load("/tmp/sglang_debug/forward_input_0.pt")
print(f"Input shape: {data['input_ids'].shape}")
print(f"First tokens: {data['input_ids'][0][:10]}")
```

#### Crash Dumps

SGLang has built-in crash dump support:

```python
engine = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B",
    crash_dump_folder="/tmp/sglang_crashes",  # Enable crash dumps
)
```

When a crash occurs, SGLang saves request state and stack traces to the dump folder.

---

### 4. Remote Debugger (debugpy)

**Best for**: Full debugging experience (breakpoints, variable inspection, stepping)

Use `debugpy` to attach a remote debugger to the subprocess.

#### Step 1: Install debugpy

```bash
pip install debugpy
```

#### Step 2: Add debugpy to Scheduler Process

Edit [scheduler.py#L2989-L3084](python/sglang/srt/managers/scheduler.py#L2989-L3084):

```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # === ADD THIS BLOCK ===
    import debugpy
    import os

    # Only debug TP rank 0 to avoid conflicts
    if tp_rank == 0:
        # Each subprocess needs a unique port
        debug_port = 5678 + gpu_id
        debugpy.listen(("0.0.0.0", debug_port))
        print(f"[SCHEDULER TP{tp_rank}] Waiting for debugger on port {debug_port}...")
        debugpy.wait_for_client()  # Blocks until debugger attaches
        print(f"[SCHEDULER TP{tp_rank}] Debugger attached!")
    # === END BLOCK ===

    # Rest of initialization...
    configure_logger(server_args, prefix=prefix)
    scheduler = Scheduler(...)
    # ...
```

#### Step 3: Attach Debugger

**VSCode Launch Configuration** (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Scheduler",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "${workspaceFolder}"
                }
            ]
        }
    ]
}
```

**PyCharm**: Run → Attach to Process → Select port 5678

#### Step 4: Debug Flow

1. **Start your script**:
   ```python
   import sglang as sgl
   engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B")
   ```

2. **Script pauses** with message: `Waiting for debugger on port 5678...`

3. **Attach debugger** in your IDE (Run → Start Debugging → "Attach to Scheduler")

4. **Set breakpoints** in model/scheduler code:
   ```python
   # model_runner.py
   def forward(self, batch):
       breakpoint()  # Or set breakpoint in IDE
       logits = self.model.forward(...)
   ```

5. **Continue execution** - breakpoints now work!

#### Multi-GPU Debugging

If using `tp_size > 1`, each rank needs a unique port:

```python
if tp_rank == 0:  # Only debug rank 0
    debug_port = 5678
    debugpy.listen(("0.0.0.0", debug_port))
    debugpy.wait_for_client()
```

---

### 5. PDB with Subprocess Attachment

**Best for**: Terminal-based debugging, quick inspection without IDE

Python's built-in `pdb` can work in subprocesses with proper stdin redirection.

#### Method A: Using pdb.set_trace()

**Modify subprocess creation** to inherit stdin:

Edit [engine.py#L804-L820](python/sglang/srt/entrypoints/engine.py#L804-L820):

```python
import sys

proc = mp.Process(
    target=run_scheduler_process,
    args=(server_args, port_args, gpu_id, tp_rank, ...),
)

# BEFORE starting, redirect stdin
import multiprocessing
if hasattr(multiprocessing, 'set_start_method'):
    # Ensure subprocess can access terminal
    pass

proc.start()
```

**Better approach**: Use remote_pdb (subprocess-friendly pdb):

```bash
pip install remote-pdb
```

**In scheduler code**:

```python
def forward(self, batch):
    # Drop into debugger on port 4444
    from remote_pdb import set_trace
    set_trace(port=4444)  # Listens on localhost:4444

    logits = self.model.forward(...)
    return logits
```

**Connect to debugger**:

```bash
# In another terminal
telnet localhost 4444
```

You now have a pdb session in the subprocess!

#### Method B: rpdb (Remote PDB)

```bash
pip install rpdb
```

```python
import rpdb

def forward(self, batch):
    rpdb.set_trace()  # Default port 4444
    logits = self.model.forward(...)
```

Connect with: `telnet localhost 4444`

---

### 6. Single Process Mode (Advanced)

**Best for**: Development, deep debugging sessions

Run everything in a single process to enable normal debugging.

#### Approach: Disable Multiprocessing

This requires **modifying SGLang's code** to bypass subprocess creation.

**Option 1**: Set environment variable before importing:

```python
import os
os.environ['SGLANG_SINGLE_PROCESS_MODE'] = '1'  # Hypothetical flag

import sglang as sgl
```

**Option 2**: Directly call scheduler in main process:

```python
# Create a custom wrapper script
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs, PortArgs

# Initialize in main process (no subprocess)
server_args = ServerArgs(model_path="meta-llama/Llama-3.1-8B")
port_args = PortArgs.init_new(server_args)

scheduler = Scheduler(
    server_args=server_args,
    port_args=port_args,
    gpu_id=0,
    tp_rank=0,
    moe_ep_rank=0,
    pp_rank=0,
    dp_rank=None,
)

# Now you can debug directly
logits = scheduler.model_runner.forward(batch)
```

**⚠️ Limitations**:
- Only works for `tp_size=1, pp_size=1, dp_size=1`
- Requires manual request handling
- IPC (ZMQ) won't work
- Not suitable for production

---

## Common Debugging Scenarios

### Scenario 1: Model Forward Pass Returns NaN

**Goal**: Find where NaN originates

```python
# In model_runner.py
def forward(self, batch: ScheduleBatch):
    import torch

    # Check inputs
    assert not torch.isnan(batch.input_ids).any(), "NaN in input_ids"

    # Forward with checkpoints
    logits = self.model.forward(batch.input_ids, batch.positions, batch)

    # Check output
    if torch.isnan(logits).any():
        print(f"[ERROR] NaN detected in logits!")
        print(f"Batch size: {len(batch.reqs)}")
        print(f"Input IDs: {batch.input_ids}")
        print(f"Logits stats: min={logits.min()}, max={logits.max()}")

        # Save for inspection
        torch.save(
            {"input_ids": batch.input_ids, "logits": logits},
            "/tmp/nan_debug.pt"
        )

        # Option: raise exception to trigger crash dump
        raise ValueError("NaN in logits")

    return logits
```

### Scenario 2: Debugging Specific Request

**Goal**: Debug only when processing a specific prompt

```python
# In tokenizer_manager.py
async def _tokenize_one_request(self, obj: GenerateReqInput):
    # Set a condition
    if "dragon" in obj.text:  # Debug requests about dragons
        print(f"[DEBUG] Dragon request detected: {obj.text}")
        # Add detailed logging or save state

    input_ids = await self._tokenize_texts(obj.text)
    return TokenizedGenerateReqInput(...)
```

### Scenario 3: Hanging Request

**Goal**: Find where request is stuck

**Add timeout logging**:

```python
import time

# In scheduler.py event_loop_normal
def event_loop_normal(self):
    while True:
        start = time.time()

        # Receive requests
        recv_reqs = self.recv_requests()
        elapsed = time.time() - start
        if elapsed > 1.0:
            print(f"[WARNING] recv_requests took {elapsed:.2f}s")

        # Process batch
        start = time.time()
        batch = self.get_next_batch_to_run()
        elapsed = time.time() - start
        if elapsed > 1.0:
            print(f"[WARNING] get_next_batch took {elapsed:.2f}s")

        # Forward pass
        if batch:
            start = time.time()
            result = self.model_runner.forward(batch)
            elapsed = time.time() - start
            print(f"[TIMING] Forward took {elapsed:.3f}s for batch_size={len(batch.reqs)}")
```

### Scenario 4: Memory Leak Investigation

**Goal**: Track memory usage over time

```python
import torch
import os
import psutil

# In model_runner.py
def forward(self, batch: ScheduleBatch):
    # Log GPU memory
    if self.forward_pass_id % 10 == 0:  # Every 10 forward passes
        allocated = torch.cuda.memory_allocated(self.gpu_id) / 1e9
        reserved = torch.cuda.memory_reserved(self.gpu_id) / 1e9

        # Log CPU memory
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1e9

        print(f"[MEMORY] Forward #{self.forward_pass_id}")
        print(f"  GPU allocated: {allocated:.2f} GB")
        print(f"  GPU reserved: {reserved:.2f} GB")
        print(f"  CPU RSS: {cpu_mem:.2f} GB")
        print(f"  KV cache usage: {self.token_to_kv_pool.available / self.token_to_kv_pool.size:.1%}")

    logits = self.model.forward(...)
    self.forward_pass_id += 1
    return logits
```

---

## Tips and Best Practices

### 1. Use Conditional Debugging

Don't debug every request - add conditions:

```python
DEBUG_REQUESTS = os.environ.get("SGLANG_DEBUG_REQUESTS", "").split(",")

def forward(self, batch):
    should_debug = any(req.rid in DEBUG_REQUESTS for req in batch.reqs)

    if should_debug:
        print(f"[DEBUG] Processing request {batch.reqs[0].rid}")
        # Detailed logging only for specific requests
```

Run with:
```bash
SGLANG_DEBUG_REQUESTS="req_12345,req_67890" python your_script.py
```

### 2. Structured Debug Output

Create a debug context manager:

```python
from contextlib import contextmanager
import time

@contextmanager
def debug_timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name}: {elapsed:.3f}s")

# Usage
with debug_timer("Model forward"):
    logits = self.model.forward(...)
```

### 3. Subprocess PID Tracking

Print subprocess PIDs at startup:

```python
# In run_scheduler_process
def run_scheduler_process(...):
    print(f"[SCHEDULER] Started with PID={os.getpid()}, TP={tp_rank}, GPU={gpu_id}")
    # Rest of code...
```

Then you can:
- Attach `gdb` or other debuggers by PID
- Send signals: `kill -USR1 <PID>` for custom signal handlers
- Monitor with `top -p <PID>`

### 4. Signal-Based Debugging

Add custom signal handlers:

```python
import signal

def debug_handler(signum, frame):
    """Triggered by: kill -USR1 <PID>"""
    print(f"[DEBUG] Signal received!")
    print(f"[DEBUG] Current batch size: {len(self.current_batch.reqs)}")
    print(f"[DEBUG] KV cache usage: {self.token_to_kv_pool.available}")
    # Print anything useful

signal.signal(signal.SIGUSR1, debug_handler)
```

Trigger from another terminal:
```bash
kill -USR1 $(pgrep -f "sglang::scheduler")
```

### 5. Environment Variable Flags

Control debug behavior with environment variables:

```python
DEBUG_FORWARD = os.environ.get("SGLANG_DEBUG_FORWARD", "0") == "1"
DEBUG_SAVE_TENSORS = os.environ.get("SGLANG_DEBUG_SAVE_TENSORS", "0") == "1"

def forward(self, batch):
    if DEBUG_FORWARD:
        print(f"[DEBUG] Forward batch: {len(batch.reqs)} requests")

    logits = self.model.forward(...)

    if DEBUG_SAVE_TENSORS:
        torch.save({"logits": logits}, "/tmp/debug_logits.pt")

    return logits
```

### 6. Trace Execution with sys.settrace

For deep debugging, trace function calls:

```python
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        print(f"[TRACE] {code.co_filename}:{code.co_name}")
    return trace_calls

# Enable tracing in scheduler
sys.settrace(trace_calls)
```

**⚠️ Warning**: Very verbose and slow!

### 7. Post-Mortem Debugging

If subprocess crashes, enable post-mortem debugging:

```python
import sys
import pdb

def run_scheduler_process(...):
    try:
        scheduler = Scheduler(...)
        scheduler.event_loop_normal()
    except Exception:
        # Drop into debugger on crash
        import traceback
        traceback.print_exc()
        pdb.post_mortem()
```

---

## Recommended Workflow

**For quick debugging**:
1. Start with **logging** (`log_level="debug"`)
2. Add **print statements** with PID/rank prefix
3. Save tensors to files if needed

**For deep debugging**:
1. Use **debugpy remote debugger** for full IDE experience
2. Set breakpoints in model code
3. Inspect variables, step through code

**For production issues**:
1. Enable **crash dumps** (`crash_dump_folder="..."`)
2. Add **structured logging** with context
3. Use **signal handlers** for runtime inspection

**Development setup**:
```python
# debug_engine.py
import sglang as sgl
import os

# Enable all debugging features
os.environ['SGLANG_DEBUG_FORWARD'] = '1'

engine = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B",
    log_level="debug",
    crash_dump_folder="/tmp/sglang_debug",
    show_time_cost=True,  # Show timing info
)

# Add debugpy if needed
# See "Remote Debugger" section above
```

---

## Summary

Debugging multi-process systems requires different strategies than single-process debugging:

| Strategy | Complexity | IDE Support | Best For |
|----------|-----------|-------------|----------|
| **Logging** | ⭐ Low | ✅ Yes | Production, understanding flow |
| **Print debugging** | ⭐ Low | ✅ Yes | Quick value inspection |
| **File-based** | ⭐⭐ Medium | ✅ Yes | Tensor inspection, crashes |
| **Remote debugger (debugpy)** | ⭐⭐⭐ High | ✅ Yes | Full debugging experience |
| **PDB/remote_pdb** | ⭐⭐ Medium | ❌ Terminal only | Command-line debugging |
| **Single process mode** | ⭐⭐⭐ High | ✅ Yes | Development only |

**Recommendation**: Start with logging and progress to remote debugging as needed. The combination of `debugpy` + logging + file-based tensor saving covers 95% of debugging scenarios.

---

**Key Files for Debugging**:
- Scheduler: [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)
- ModelRunner: [python/sglang/srt/model_executor/model_runner.py](python/sglang/srt/model_executor/model_runner.py)
- Engine: [python/sglang/srt/entrypoints/engine.py](python/sglang/srt/entrypoints/engine.py)
