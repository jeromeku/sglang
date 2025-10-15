# ZMQ and IPC Primer for SGLang

This document provides a practical introduction to **ZeroMQ (ZMQ)** and **Inter-Process Communication (IPC)** as used in SGLang's architecture.

## Table of Contents

- [What is Inter-Process Communication (IPC)?](#what-is-inter-process-communication-ipc)
- [What is ZeroMQ (ZMQ)?](#what-is-zeromq-zmq)
- [ZMQ Socket Patterns](#zmq-socket-patterns)
- [How SGLang Uses ZMQ](#how-sglang-uses-zmq)
- [ZMQ in Action: Code Walkthrough](#zmq-in-action-code-walkthrough)
- [ZMQ vs Other IPC Methods](#zmq-vs-other-ipc-methods)
- [Practical Examples](#practical-examples)
- [Common Patterns and Pitfalls](#common-patterns-and-pitfalls)

---

## What is Inter-Process Communication (IPC)?

### The Problem

When you have multiple processes (not threads) that need to exchange data, they face a challenge:

```
Process A                Process B
┌─────────────┐         ┌─────────────┐
│ Memory: 0x1 │         │ Memory: 0x1 │
│   data = 5  │    ?    │   data = ?  │
└─────────────┘         └─────────────┘
    Separate memory spaces!
    Cannot directly access each other's memory
```

**Key constraint**: Unlike threads (which share memory), processes have **isolated memory spaces** for security and stability.

### IPC Solutions

**Inter-Process Communication (IPC)** refers to mechanisms that allow processes to exchange data:

| Method | How It Works | Example |
|--------|--------------|---------|
| **Pipes** | One-way data flow between parent/child | `os.pipe()`, shell pipes `\|` |
| **Named Pipes (FIFO)** | Persistent pipes accessible by path | `mkfifo /tmp/mypipe` |
| **Sockets** | Network-style communication (TCP/UDP) | `socket.socket()` |
| **Unix Domain Sockets** | Sockets for same-machine communication | Fast, filesystem-based |
| **Shared Memory** | Direct memory sharing | Fast but complex, needs synchronization |
| **Message Queues** | FIFO message buffers | `queue.Queue()` (multiprocessing) |
| **Memory-Mapped Files** | File-backed shared memory | `mmap` |

### Why Not Just Use TCP Sockets?

You *could* use raw TCP sockets for IPC:

```python
# Process A (server)
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 9999))
s.listen()
conn, addr = s.accept()
conn.send(b"Hello from Process A")

# Process B (client)
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 9999))
data = s.recv(1024)
```

**Problems**:
- ❌ Requires port management
- ❌ Lots of boilerplate (bind, listen, accept, connect)
- ❌ Need to handle serialization manually
- ❌ No built-in messaging patterns (request-reply, pub-sub, etc.)
- ❌ Connection management complexity

**This is where ZMQ comes in!**

---

## What is ZeroMQ (ZMQ)?

### The Elevator Pitch

> **ZeroMQ is like sockets on steroids** - it provides high-level messaging patterns while handling all the low-level networking complexity.

Think of it as:
- 📬 A **mailbox system** for processes (or machines)
- 🔌 Sockets that "just work" without connection management
- 📦 Built-in message serialization
- 🚀 High-performance, low-latency

### Key Concepts

#### 1. Sockets with Superpowers

ZMQ sockets are **NOT** the same as TCP sockets:

```python
import zmq

context = zmq.Context()

# Create a ZMQ socket (not a TCP socket!)
socket = context.socket(zmq.PUSH)

# Connect to an endpoint
socket.connect("tcp://localhost:5555")

# Send a Python object - ZMQ handles serialization!
socket.send_pyobj({"message": "Hello", "data": [1, 2, 3]})
```

**What ZMQ does for you**:
- ✅ Automatic reconnection on failure
- ✅ Message queuing (buffers messages if receiver is slow)
- ✅ Built-in patterns (PUSH/PULL, PUB/SUB, REQ/REP)
- ✅ Serialization via `send_pyobj()` / `recv_pyobj()`
- ✅ Works over TCP, Unix sockets, or in-process

#### 2. Transport Agnostic

ZMQ supports multiple transport protocols with the **same API**:

```python
# TCP (network)
socket.connect("tcp://192.168.1.100:5555")

# Unix domain socket (local, fast)
socket.connect("ipc:///tmp/my_socket")

# In-process (threads)
socket.connect("inproc://my_channel")
```

SGLang primarily uses **`ipc://`** (Unix domain sockets) for local inter-process communication.

#### 3. No Connection Management

Unlike TCP, ZMQ sockets are **connection-less** from your perspective:

```python
# Traditional TCP
server.bind(...)
server.listen()
conn, addr = server.accept()  # Wait for connection
conn.send(data)

# ZMQ
socket.bind("tcp://*:5555")
socket.send(data)  # No accept() needed!
```

ZMQ handles connections in the background. You just send/receive messages.

---

## ZMQ Socket Patterns

ZMQ provides **messaging patterns** that solve common communication needs.

### 1. PUSH/PULL (Pipeline)

**Pattern**: One-way data flow from producers to consumers

```
Producer(s)  ───PUSH──→  Consumer(s)
               (load balanced)
```

**Use case**: Distributing tasks to workers

```python
# Producer (sends tasks)
context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.bind("tcp://*:5555")

for i in range(10):
    sender.send_pyobj({"task_id": i, "data": f"Task {i}"})

# Consumer (receives tasks)
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://localhost:5555")

while True:
    task = receiver.recv_pyobj()
    print(f"Processing {task}")
```

**Key properties**:
- ✅ Load balancing: Messages distributed round-robin to multiple consumers
- ✅ One-way: PUSH sends, PULL receives (no replies)
- ✅ Asynchronous: Sender doesn't wait for receiver

**SGLang uses PUSH/PULL** for TokenizerManager → Scheduler → DetokenizerManager communication.

### 2. REQ/REP (Request-Reply)

**Pattern**: Synchronous request-response

```
Client  ───REQ──→  Server
        ←──REP───
```

**Use case**: RPC-style communication

```python
# Server
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    request = socket.recv_pyobj()
    print(f"Received: {request}")
    response = {"result": request["value"] * 2}
    socket.send_pyobj(response)

# Client
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

socket.send_pyobj({"value": 21})
response = socket.recv_pyobj()  # Blocks until response
print(response)  # {"result": 42}
```

**Key properties**:
- ✅ Synchronous: Client blocks until reply received
- ✅ Strict order: Must alternate send/recv
- ❌ Limited: One outstanding request at a time

### 3. PUB/SUB (Publish-Subscribe)

**Pattern**: One-to-many broadcasting

```
Publisher  ───PUB──→  Subscriber 1
                  └──→  Subscriber 2
                  └──→  Subscriber 3
```

**Use case**: Broadcasting events, notifications

```python
# Publisher
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

socket.send_pyobj({"event": "user_login", "user_id": 123})

# Subscriber
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.subscribe(b"")  # Subscribe to all messages

while True:
    event = socket.recv_pyobj()
    print(f"Event: {event}")
```

**Key properties**:
- ✅ One-to-many: Multiple subscribers receive same message
- ✅ Filtering: Subscribers can filter by topic
- ❌ No delivery guarantee: Late subscribers miss messages

### 4. DEALER/ROUTER (Advanced)

**Pattern**: Asynchronous request-reply with routing

Used for complex routing scenarios. SGLang uses DEALER for RPC communication.

**DEALER** = asynchronous REQ (can send multiple requests without waiting)
**ROUTER** = asynchronous REP (can route replies to specific clients)

---

## How SGLang Uses ZMQ

### Communication Topology

```
┌────────────────────────────────────┐
│     TokenizerManager (Main)        │
│                                    │
│  send_to_scheduler (PUSH)          │
│         │                          │
└─────────┼──────────────────────────┘
          │
          │ (IPC socket)
          │
          ▼
┌─────────────────────────────────────┐
│      Scheduler (Subprocess)         │
│                                     │
│  recv_from_tokenizer (PULL) ◄───────┤
│                                     │
│  send_to_detokenizer (PUSH)         │
│         │                           │
└─────────┼───────────────────────────┘
          │
          │ (IPC socket)
          │
          ▼
┌─────────────────────────────────────┐
│   DetokenizerManager (Subprocess)   │
│                                     │
│  recv_from_scheduler (PULL) ◄───────┤
│                                     │
│  send_to_tokenizer (PUSH)           │
│         │                           │
└─────────┼───────────────────────────┘
          │
          │ (IPC socket)
          │
          └─────────────► (back to TokenizerManager)
```

### Why PUSH/PULL?

SGLang uses **PUSH/PULL pattern** because:
1. ✅ **One-way flow**: Each stage processes and forwards (tokenize → schedule → detokenize)
2. ✅ **Asynchronous**: Sender doesn't wait for receiver to process
3. ✅ **Queueing**: Messages buffered if receiver is busy (e.g., model forward pass takes time)
4. ✅ **Load balancing**: Could add multiple schedulers in the future

### Socket Configuration in SGLang

**Location**: [engine.py#L139-L142](python/sglang/srt/entrypoints/engine.py#L139-L142)

```python
context = zmq.Context(2)  # 2 I/O threads
self.send_to_rpc = get_zmq_socket(
    context, zmq.DEALER, self.port_args.rpc_ipc_name, True
)
```

**Helper function**: [utils.py (get_zmq_socket)](python/sglang/srt/utils.py)

```python
def get_zmq_socket(context, socket_type, ipc_name, is_bind):
    """Create and configure a ZMQ socket"""
    socket = context.socket(socket_type)

    if is_bind:
        socket.bind(ipc_name)  # Server side
    else:
        socket.connect(ipc_name)  # Client side

    return socket
```

### IPC Naming

**Port allocation**: [server_args.py (PortArgs)](python/sglang/srt/server_args.py)

```python
@dataclass
class PortArgs:
    tokenizer_ipc_name: str
    detokenizer_ipc_name: str
    scheduler_input_ipc_name: str
    # ...

    @classmethod
    def init_new(cls, server_args: ServerArgs):
        # Generate unique IPC names
        nonce = secrets.token_hex(16)
        return cls(
            tokenizer_ipc_name=f"ipc://{tempfile.gettempdir()}/sglang_tokenizer_{nonce}",
            detokenizer_ipc_name=f"ipc://{tempfile.gettempdir()}/sglang_detokenizer_{nonce}",
            # ...
        )
```

**IPC names look like**: `ipc:///tmp/sglang_tokenizer_a3f9b2c1d4e5f6g7h8i9`

This creates a Unix domain socket at `/tmp/sglang_tokenizer_a3f9b2c1d4e5f6g7h8i9`.

---

## ZMQ in Action: Code Walkthrough

Let's trace a request through SGLang's ZMQ communication.

### Step 1: TokenizerManager Sends Request

**Location**: [tokenizer_manager.py (send_to_scheduler)](python/sglang/srt/managers/tokenizer_manager.py)

```python
class TokenizerManager:
    def __init__(self, server_args, port_args):
        # Create ZMQ context
        context = zmq.Context(2)

        # Create PUSH socket to send to Scheduler
        self.send_to_scheduler = get_zmq_socket(
            context,
            zmq.PUSH,
            port_args.scheduler_input_ipc_name,
            False  # Connect (not bind)
        )

    def _send_one_request(self, obj, tokenized_obj, created_time):
        # Send Python object via ZMQ
        self.send_to_scheduler.send_pyobj(tokenized_obj)
```

**What happens**:
1. `send_pyobj(tokenized_obj)` serializes the object using pickle
2. ZMQ sends the serialized bytes through the Unix domain socket
3. If Scheduler's receive buffer is full, message is queued
4. Method returns immediately (non-blocking)

### Step 2: Scheduler Receives Request

**Location**: [scheduler.py (recv_requests)](python/sglang/srt/managers/scheduler.py)

```python
class Scheduler:
    def __init__(self, server_args, port_args, ...):
        context = zmq.Context(2)

        # Create PULL socket to receive from TokenizerManager
        self.recv_from_tokenizer = get_zmq_socket(
            context,
            zmq.PULL,
            port_args.scheduler_input_ipc_name,
            True  # Bind (server)
        )

    def recv_requests(self) -> List:
        """Receive pending requests from TokenizerManager"""
        recv_reqs = []

        # Non-blocking receive loop
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.Again:
                # No more messages available
                break

        return recv_reqs
```

**What happens**:
1. `recv_pyobj(zmq.NOBLOCK)` checks if messages are available
2. If yes: deserialize and return
3. If no: raises `zmq.Again` exception (like `EWOULDBLOCK`)
4. Collects all pending messages in one go

### Step 3: Scheduler Sends to Detokenizer

**Location**: [scheduler.py (send_to_detokenizer)](python/sglang/srt/managers/scheduler.py)

```python
class Scheduler:
    def __init__(self, ...):
        # Create PUSH socket to send to DetokenizerManager
        self.send_to_detokenizer = get_zmq_socket(
            context,
            zmq.PUSH,
            port_args.detokenizer_ipc_name,
            False
        )

    def process_batch_result(self, batch, result):
        """Send output tokens to Detokenizer"""
        output = BatchTokenIDOutput(
            rids=[req.rid for req in batch.reqs],
            output_ids=[result.next_token_ids],
            # ...
        )
        self.send_to_detokenizer.send_pyobj(output)
```

### Step 4: Detokenizer Sends Back to TokenizerManager

**Location**: [detokenizer_manager.py](python/sglang/srt/managers/detokenizer_manager.py)

```python
class DetokenizerManager:
    def __init__(self, server_args, port_args):
        context = zmq.Context(2)

        # Receive from Scheduler
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )

        # Send to TokenizerManager
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

    def event_loop(self):
        while True:
            # Blocking receive
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            # Decode tokens to text
            decoded_output = self.decode_tokens(recv_obj)

            # Send back to TokenizerManager
            self.send_to_tokenizer.send_pyobj(decoded_output)
```

### Complete Message Flow

```
User calls engine.generate("Hello")
         ↓
┌────────────────────────────────────────────────┐
│ TokenizerManager                               │
│   tokenizes → "Hello" becomes [1, 2, 3]       │
│   send_to_scheduler.send_pyobj(tokenized_obj)  │
└────────────────┬───────────────────────────────┘
                 │
                 │ IPC: ipc:///tmp/sglang_scheduler_...
                 │ PUSH → PULL
                 ↓
┌────────────────────────────────────────────────┐
│ Scheduler                                      │
│   recv_from_tokenizer.recv_pyobj()             │
│   batch requests                               │
│   model.forward(batch) → next_token_ids        │
│   send_to_detokenizer.send_pyobj(token_ids)    │
└────────────────┬───────────────────────────────┘
                 │
                 │ IPC: ipc:///tmp/sglang_detokenizer_...
                 │ PUSH → PULL
                 ↓
┌────────────────────────────────────────────────┐
│ DetokenizerManager                             │
│   recv_from_scheduler.recv_pyobj()             │
│   decode: [4, 5] → "world"                     │
│   send_to_tokenizer.send_pyobj(decoded_text)   │
└────────────────┬───────────────────────────────┘
                 │
                 │ IPC: ipc:///tmp/sglang_tokenizer_...
                 │ PUSH → PULL
                 ↓
┌────────────────────────────────────────────────┐
│ TokenizerManager                               │
│   recv_from_detokenizer.recv_pyobj()           │
│   yield response to user                       │
└────────────────────────────────────────────────┘
         ↓
User receives: {"text": "Hello world"}
```

---

## ZMQ vs Other IPC Methods

| Feature | ZMQ | Raw TCP | Unix Sockets | Multiprocessing.Queue | Pipes |
|---------|-----|---------|--------------|----------------------|-------|
| **Setup complexity** | Low | High | Medium | Low | Low |
| **Serialization** | Built-in | Manual | Manual | Built-in | Manual |
| **Patterns** | Many | Manual | Manual | Queue only | One-way |
| **Performance** | Very fast | Fast | Very fast | Medium | Fast |
| **Network support** | Yes | Yes | No | No | No |
| **Async support** | Yes | Yes | Yes | No | No |
| **Reconnection** | Automatic | Manual | Manual | N/A | N/A |
| **Buffering** | Automatic | Manual | Manual | Built-in | Limited |

**Why SGLang chose ZMQ**:
1. ✅ **High performance**: Near-zero overhead for local IPC
2. ✅ **Simple API**: Less boilerplate than raw sockets
3. ✅ **Flexible patterns**: PUSH/PULL, REQ/REP, PUB/SUB
4. ✅ **Robust**: Automatic reconnection, buffering
5. ✅ **Language agnostic**: Could interface with C++/Rust components
6. ✅ **Battle-tested**: Used in many production systems

---

## Practical Examples

### Example 1: Simple PUSH/PULL

Minimal example showing SGLang's communication pattern:

```python
import zmq
import time
import multiprocessing as mp

def worker_process(ipc_name):
    """Worker receives and processes messages"""
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(ipc_name)

    print(f"[Worker] Listening on {ipc_name}")

    while True:
        # Blocking receive
        message = receiver.recv_pyobj()
        print(f"[Worker] Received: {message}")

        if message == "STOP":
            break

        # Simulate processing
        time.sleep(0.1)
        print(f"[Worker] Processed: {message}")

def main():
    ipc_name = "ipc:///tmp/example_socket"

    # Start worker process
    proc = mp.Process(target=worker_process, args=(ipc_name,))
    proc.start()

    # Give worker time to bind
    time.sleep(0.1)

    # Create sender
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect(ipc_name)

    # Send messages
    for i in range(5):
        message = f"Task {i}"
        print(f"[Main] Sending: {message}")
        sender.send_pyobj(message)
        time.sleep(0.05)

    # Stop worker
    sender.send_pyobj("STOP")
    proc.join()

if __name__ == "__main__":
    main()
```

**Output**:
```
[Worker] Listening on ipc:///tmp/example_socket
[Main] Sending: Task 0
[Worker] Received: Task 0
[Main] Sending: Task 1
[Worker] Processed: Task 0
[Worker] Received: Task 1
[Main] Sending: Task 2
...
```

### Example 2: Non-Blocking Receive

How Scheduler polls for new requests:

```python
import zmq

context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("ipc:///tmp/my_socket")

while True:
    messages = []

    # Collect all available messages (non-blocking)
    while True:
        try:
            msg = receiver.recv_pyobj(zmq.NOBLOCK)
            messages.append(msg)
        except zmq.Again:
            # No more messages
            break

    if messages:
        print(f"Received {len(messages)} messages")
        # Process batch
    else:
        # No messages, do other work
        time.sleep(0.001)
```

### Example 3: ZMQ with Async (asyncio)

ZMQ can work with asyncio using `zmq.asyncio`:

```python
import zmq.asyncio
import asyncio

async def async_receiver():
    context = zmq.asyncio.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("ipc:///tmp/async_socket")

    while True:
        # Async receive - doesn't block event loop
        message = await receiver.recv_pyobj()
        print(f"Received: {message}")

        # Can do async operations
        await asyncio.sleep(0.1)

asyncio.run(async_receiver())
```

SGLang's TokenizerManager uses asyncio, and could potentially use `zmq.asyncio` for non-blocking ZMQ operations.

### Example 4: REQ/REP for RPC

SGLang's RPC mechanism (for getting model weights):

```python
# Server (Scheduler subprocess)
def rpc_server(ipc_name):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(ipc_name)

    while True:
        request = socket.recv_pyobj()
        method = request["method"]
        params = request["parameters"]

        # Execute method
        if method == "get_weights":
            result = {"weights": [1, 2, 3, 4, 5]}
        else:
            result = {"error": "Unknown method"}

        socket.send_pyobj(result)

# Client (Engine main process)
def rpc_client(ipc_name):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(ipc_name)

    # Send request
    socket.send_pyobj({
        "method": "get_weights",
        "parameters": {"layer": 0}
    })

    # Wait for response
    response = socket.recv_pyobj()
    print(response)
```

---

## Common Patterns and Pitfalls

### ✅ Best Practices

#### 1. Bind vs Connect

**Rule of thumb**: Stable components bind, dynamic components connect.

```python
# Server (long-lived) - BIND
server = context.socket(zmq.PULL)
server.bind("ipc:///tmp/server")

# Client (may restart) - CONNECT
client = context.socket(zmq.PUSH)
client.connect("ipc:///tmp/server")
```

**Why**: If client restarts, it can reconnect. If server restarts, client will auto-reconnect.

**SGLang**: Scheduler binds (server), TokenizerManager connects (client).

#### 2. Context Management

Create **one context per process**, reuse for all sockets:

```python
# ✅ Good
context = zmq.Context(2)  # 2 I/O threads
socket1 = context.socket(zmq.PUSH)
socket2 = context.socket(zmq.PULL)

# ❌ Bad - don't create multiple contexts
context1 = zmq.Context()
socket1 = context1.socket(zmq.PUSH)
context2 = zmq.Context()  # Wasteful
socket2 = context2.socket(zmq.PULL)
```

#### 3. Non-Blocking Receive

Use non-blocking receives to avoid hanging:

```python
# ✅ Good - non-blocking
try:
    msg = socket.recv_pyobj(zmq.NOBLOCK)
except zmq.Again:
    pass  # No message available

# ❌ Bad - blocks forever if no messages
msg = socket.recv_pyobj()  # Hangs!
```

#### 4. Send Before Bind/Connect

ZMQ queues messages even before connection:

```python
# This works! Message queued until connection established
socket.send_pyobj({"data": "hello"})
socket.connect("ipc:///tmp/socket")
```

#### 5. Cleanup

Always close sockets and terminate context:

```python
socket.close()
context.term()
```

Or use context managers (if available).

### ❌ Common Pitfalls

#### 1. Blocking Receive in Main Thread

```python
# ❌ Bad - blocks main thread
while True:
    msg = socket.recv_pyobj()  # Blocks!
    process(msg)
```

**Solution**: Use non-blocking or timeout:

```python
# ✅ Good
msg = socket.recv_pyobj(zmq.NOBLOCK)

# Or with timeout
socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
msg = socket.recv_pyobj()
```

#### 2. Wrong Socket Pattern

```python
# ❌ Bad - mismatched patterns
sender = context.socket(zmq.PUSH)
receiver = context.socket(zmq.REP)  # Wrong! Should be PULL
```

**Correct pairs**:
- PUSH ↔ PULL
- REQ ↔ REP
- PUB ↔ SUB
- DEALER ↔ ROUTER

#### 3. Forgetting to Subscribe (PUB/SUB)

```python
# ❌ Bad - won't receive anything!
sub = context.socket(zmq.SUB)
sub.connect("ipc:///tmp/pub")
msg = sub.recv_pyobj()  # Never receives!

# ✅ Good
sub = context.socket(zmq.SUB)
sub.subscribe(b"")  # Subscribe to all messages
sub.connect("ipc:///tmp/pub")
```

#### 4. Large Message Serialization

`send_pyobj()` uses pickle, which can be slow for large objects:

```python
# ❌ Slow for large tensors
socket.send_pyobj(large_tensor)

# ✅ Better - use shared memory or specialized serialization
socket.send(large_tensor.numpy().tobytes())
```

SGLang handles this by using shared memory pools for KV cache (not transmitted via ZMQ).

#### 5. Bind Address Already in Use

```python
# ❌ Error if socket file exists
socket.bind("ipc:///tmp/my_socket")
# zmq.error.ZMQError: Address already in use

# ✅ Good - use unique names
import secrets
nonce = secrets.token_hex(8)
socket.bind(f"ipc:///tmp/my_socket_{nonce}")
```

SGLang does this with random nonces in `PortArgs.init_new()`.

### Performance Tips

#### 1. Batching Messages

Instead of sending many small messages:

```python
# ❌ Slow - many small sends
for item in items:
    socket.send_pyobj(item)

# ✅ Fast - batch
socket.send_pyobj(items)
```

#### 2. Zero-Copy with Memory Views

For large data, avoid copies:

```python
import numpy as np

# Zero-copy send
data = np.array([1, 2, 3, 4, 5])
socket.send(data, copy=False)

# Zero-copy receive
message = socket.recv(copy=False)
array = np.frombuffer(message, dtype=np.int64)
```

#### 3. High Water Mark (Buffer Size)

Control message queue size:

```python
# Limit send buffer to 100 messages
socket.setsockopt(zmq.SNDHWM, 100)

# If buffer full, send_pyobj will block or drop (depending on socket type)
```

---

## Summary

### Key Takeaways

1. **IPC** = mechanisms for processes to exchange data (they can't share memory)
2. **ZMQ** = high-level messaging library that makes IPC easy
3. **Patterns**: PUSH/PULL (pipeline), REQ/REP (request-reply), PUB/SUB (broadcast)
4. **SGLang uses PUSH/PULL** for one-way async communication between components
5. **IPC transport** = Unix domain sockets (`ipc:///tmp/...`) for same-machine communication

### When to Use ZMQ

✅ **Use ZMQ when**:
- Multi-process architecture with message passing
- Need high-performance local IPC
- Want built-in patterns (pub/sub, pipeline, etc.)
- Need automatic reconnection and buffering

❌ **Don't use ZMQ when**:
- Single process (use queues or direct calls)
- Need shared memory for large data (use multiprocessing.shared_memory)
- Simple parent-child communication (use pipes)

### SGLang's ZMQ Usage Summary

| Component | Socket Type | Direction | Purpose |
|-----------|-------------|-----------|---------|
| TokenizerManager → Scheduler | PUSH → PULL | Send tokenized requests | Request distribution |
| Scheduler → DetokenizerManager | PUSH → PULL | Send output token IDs | Token passing |
| DetokenizerManager → TokenizerManager | PUSH → PULL | Send decoded text | Response delivery |
| Engine → Scheduler | DEALER → ROUTER | RPC calls | Model operations |

**Transport**: All use `ipc://` (Unix domain sockets) for fast local communication.

---

## Further Reading

- **ZMQ Guide**: https://zguide.zeromq.org/ (excellent, practical guide)
- **ZMQ API Reference**: https://pyzmq.readthedocs.io/
- **Unix IPC**: Stevens & Rago, "Advanced Programming in the UNIX Environment"

---

**Key Files**:
- ZMQ utility: [python/sglang/srt/utils.py (get_zmq_socket)](python/sglang/srt/utils.py)
- Port configuration: [python/sglang/srt/server_args.py (PortArgs)](python/sglang/srt/server_args.py)
- TokenizerManager IPC: [python/sglang/srt/managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)
- Scheduler IPC: [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)
