# SGLang Code Examples & Implementation Details

Detailed code walkthroughs and implementation examples for key SGLang components.

---

## Table of Contents

1. [Batch Processing Pipeline](#batch-processing-pipeline)
2. [Memory Management Deep Dive](#memory-management-deep-dive)
3. [Attention Implementation](#attention-implementation)
4. [Scheduling Algorithms](#scheduling-algorithms)
5. [CUDA Graph Capture](#cuda-graph-capture)
6. [Distributed Communication](#distributed-communication)
7. [Complete Request Trace](#complete-request-trace)

---

## Batch Processing Pipeline

### Request Data Structure

The [`Req`](../../python/sglang/srt/managers/schedule_batch.py) class represents a single generation request:

```python
# From python/sglang/srt/managers/schedule_batch.py:100-149
class Req:
    """
    Represents a single generation request.

    Lifecycle stages:
    1. PREFILLING: Initial prompt processing
    2. RUNNING_PREFILL: Chunked prefill in progress
    3. DECODE: Autoregressive generation
    4. FINISHED: Request completed
    """

    def __init__(
        self,
        rid: str,                          # Request ID
        origin_input_text: str,            # Original prompt
        origin_input_ids: List[int],       # Tokenized prompt
        sampling_params: SamplingParams,   # Generation config
    ):
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids = origin_input_ids
        self.sampling_params = sampling_params

        # Execution state
        self.stage = RequestStage.PREFILLING
        self.output_ids = []               # Generated tokens
        self.finished = False
        self.finish_reason: Optional[BaseFinishReason] = None

        # Memory management
        self.req_pool_idx = None           # Position in ReqToTokenPool
        self.kv_indices = []               # KV cache block indices
        self.prefix_indices = []           # Cached prefix blocks
        self.last_node = None              # RadixCache tree node

        # Multimodal data
        self.multimodal_inputs: Optional[MultimodalInputs] = None

        # Timing and metrics
        self.arrival_time = time.time()
        self.prefill_start_time = None
        self.decode_start_time = None
```

### ScheduleBatch Construction

The scheduler builds batches from the waiting queue:

```python
# Simplified from python/sglang/srt/managers/scheduler.py

class Scheduler:
    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """
        Select requests from waiting queue and form a batch.

        Process:
        1. Apply scheduling policy to prioritize requests
        2. Match prefixes in RadixCache
        3. Allocate KV cache blocks
        4. Build ScheduleBatch
        """
        if not self.waiting_queue and not self.running_batch.reqs:
            return None

        # Get current running batch
        batch = self.running_batch

        # Add new requests from waiting queue
        if self.waiting_queue:
            # Apply scheduling policy
            candidates = self.policy.get_next_batch_to_run(
                self.waiting_queue,
                batch,
                self.available_memory_blocks(),
            )

            for req in candidates:
                # Match prefix in RadixCache
                prefix_match = self._match_and_allocate_kv_cache(req)

                if prefix_match:
                    # Add to batch
                    batch.reqs.append(req)
                    self.waiting_queue.remove(req)

                    # Track in req pool
                    req.req_pool_idx = self.req_to_token_pool.allocate()

        # Determine forward mode
        batch.forward_mode = self._determine_forward_mode(batch)

        return batch

    def _match_and_allocate_kv_cache(self, req: Req) -> bool:
        """
        Match prefix and allocate KV cache for a request.

        Returns True if successful, False if OOM.
        """
        # Create cache key
        cache_key = RadixKey(
            token_ids=req.origin_input_ids,
            extra_key=req.sampling_params.lora_id,  # LoRA adapter ID
        )

        # Match prefix in tree
        prefix_match = self.tree_cache.match_prefix(cache_key)

        if prefix_match:
            matched_indices = prefix_match.matched_indices
            matched_len = prefix_match.matched_len

            req.prefix_indices = matched_indices
            req.last_node = prefix_match.last_node

            # Calculate unmatched suffix length
            unmatched_len = len(req.origin_input_ids) - matched_len

            # Allocate new blocks for suffix
            needed_blocks = (unmatched_len + self.block_size - 1) // self.block_size

            if self.available_memory_blocks() >= needed_blocks:
                new_indices = self.token_to_kv_pool.alloc(needed_blocks)
                req.kv_indices = matched_indices + new_indices
                return True
            else:
                # Try eviction
                evicted = self.tree_cache.evict(needed_blocks)
                if evicted >= needed_blocks:
                    new_indices = self.token_to_kv_pool.alloc(needed_blocks)
                    req.kv_indices = matched_indices + new_indices
                    return True
                else:
                    # OOM, cannot add to batch
                    return False

        return False

    def _determine_forward_mode(self, batch: ScheduleBatch) -> ForwardMode:
        """
        Determine the forward mode for this batch.

        Modes:
        - EXTEND: All requests are in prefill
        - DECODE: All requests are in decode
        - MIXED: Mix of prefill and decode (chunked prefill)
        """
        has_prefill = any(
            req.stage in [RequestStage.PREFILLING, RequestStage.RUNNING_PREFILL]
            for req in batch.reqs
        )
        has_decode = any(
            req.stage == RequestStage.DECODE
            for req in batch.reqs
        )

        if has_prefill and has_decode:
            return ForwardMode.MIXED
        elif has_prefill:
            return ForwardMode.EXTEND
        elif has_decode:
            return ForwardMode.DECODE
        else:
            return ForwardMode.IDLE
```

### Batch Transformation: ScheduleBatch → ForwardBatch

```python
# From python/sglang/srt/model_executor/model_runner.py

class ModelRunner:
    def forward(self, model_worker_batch: ModelWorkerBatch) -> torch.Tensor:
        """
        Execute forward pass on GPU.

        Converts ModelWorkerBatch to ForwardBatch (GPU tensors).
        """
        # Build ForwardBatch with GPU tensors
        forward_batch = self._prepare_forward_batch(model_worker_batch)

        # Select execution path
        if forward_batch.forward_mode == ForwardMode.DECODE:
            # Use CUDA graph for decode
            logits = self._forward_decode_cuda_graph(forward_batch)
        else:
            # Use standard forward for prefill
            logits = self._forward_extend(forward_batch)

        return logits

    def _prepare_forward_batch(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> ForwardBatch:
        """
        Prepare GPU tensors for forward pass.
        """
        # Collect input data from all requests
        input_ids_list = []
        req_pool_indices_list = []
        seq_lens_list = []
        prefix_lens_list = []

        for req in model_worker_batch.reqs:
            if req.stage == RequestStage.DECODE:
                # Decode: only last token
                input_ids_list.append([req.output_ids[-1]])
                seq_lens_list.append(len(req.origin_input_ids) + len(req.output_ids))
                prefix_lens_list.append(len(req.prefix_indices))
            else:
                # Prefill: all unprocessed tokens
                input_ids_list.append(req.origin_input_ids[req.prefill_position:])
                seq_lens_list.append(len(input_ids_list[-1]))
                prefix_lens_list.append(0)

            req_pool_indices_list.append(req.req_pool_idx)

        # Flatten and convert to GPU tensors
        input_ids = torch.tensor(
            flatten_nested_list(input_ids_list),
            dtype=torch.int64,
            device=self.device,
        )

        req_pool_indices = torch.tensor(
            req_pool_indices_list,
            dtype=torch.int32,
            device=self.device,
        )

        seq_lens = torch.tensor(
            seq_lens_list,
            dtype=torch.int32,
            device=self.device,
        )

        prefix_lens = torch.tensor(
            prefix_lens_list,
            dtype=torch.int32,
            device=self.device,
        )

        # Build attention metadata
        attn_metadata = self._build_attention_metadata(
            model_worker_batch,
            seq_lens,
            prefix_lens,
        )

        # Create ForwardBatch
        forward_batch = ForwardBatch(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            attn_metadata=attn_metadata,
            forward_mode=model_worker_batch.forward_mode,
            attn_backend=self.attn_backend,
            kv_pool=self.kv_pool,
            req_to_token_pool=self.req_to_token_pool,
        )

        return forward_batch
```

---

## Memory Management Deep Dive

### ReqToTokenPool: Request → Token Mapping

The [`ReqToTokenPool`](../../python/sglang/srt/mem_cache/memory_pool.py) maps each request to its token positions:

```python
# From python/sglang/srt/mem_cache/memory_pool.py

class ReqToTokenPool:
    """
    Maps request indices to token position indices.

    Structure:
    - Each request gets a contiguous range in the pool
    - Token positions track which KV cache blocks are used
    """

    def __init__(self, max_requests: int, max_tokens_per_request: int):
        self.max_requests = max_requests
        self.max_tokens_per_request = max_tokens_per_request

        # req_to_token[req_idx, token_pos] = kv_block_idx
        self.req_to_token = torch.full(
            (max_requests, max_tokens_per_request),
            fill_value=-1,
            dtype=torch.int32,
            device="cuda",
        )

        # Track free request slots
        self.free_slots = list(range(max_requests))

    def allocate(self) -> int:
        """Allocate a slot for a new request."""
        if not self.free_slots:
            raise RuntimeError("ReqToTokenPool is full")
        return self.free_slots.pop(0)

    def free(self, req_idx: int):
        """Free a request slot."""
        self.req_to_token[req_idx, :] = -1
        self.free_slots.append(req_idx)

    def set_token_positions(
        self,
        req_idx: int,
        token_positions: List[int],
        kv_block_indices: List[int],
    ):
        """
        Set KV cache block indices for token positions.

        Args:
            req_idx: Request index in pool
            token_positions: List of token positions (0, 1, 2, ...)
            kv_block_indices: KV cache block indices for each position
        """
        for token_pos, block_idx in zip(token_positions, kv_block_indices):
            self.req_to_token[req_idx, token_pos] = block_idx

    def get_kv_indices(self, req_idx: int, seq_len: int) -> torch.Tensor:
        """
        Get KV cache block indices for a request's tokens.

        Returns:
            Tensor of shape (seq_len,) with block indices
        """
        return self.req_to_token[req_idx, :seq_len]
```

### TokenToKVPool: Block-Based KV Cache

The [`TokenToKVPool`](../../python/sglang/srt/mem_cache/memory_pool.py) manages the actual KV cache memory:

```python
# From python/sglang/srt/mem_cache/memory_pool.py

class MHATokenToKVPool:
    """
    Multi-Head Attention KV cache memory pool.

    Organization:
    - KV cache is divided into fixed-size blocks (pages)
    - Each block stores KV for multiple tokens (default: 16)
    - Blocks are allocated/freed dynamically
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size  # Tokens per block
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        # Allocate KV cache memory
        # Shape: (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.empty(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.empty(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Free block management
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()

    def alloc(self, num_blocks: int) -> List[int]:
        """
        Allocate blocks for KV cache.

        Returns:
            List of block indices
        """
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"OOM: Need {num_blocks}, have {len(self.free_blocks)}")

        allocated = []
        for _ in range(num_blocks):
            block_idx = self.free_blocks.pop(0)
            self.allocated_blocks.add(block_idx)
            allocated.append(block_idx)

        return allocated

    def free(self, block_indices: List[int]):
        """Free blocks back to the pool."""
        for block_idx in block_indices:
            if block_idx in self.allocated_blocks:
                self.allocated_blocks.remove(block_idx)
                self.free_blocks.append(block_idx)

    def store_kv(
        self,
        layer_id: int,
        k: torch.Tensor,  # (num_tokens, num_kv_heads, head_dim)
        v: torch.Tensor,  # (num_tokens, num_kv_heads, head_dim)
        block_indices: List[int],
        positions_in_block: List[int],
    ):
        """
        Store K/V tensors in cache blocks.

        Args:
            layer_id: Layer index
            k, v: Key/value tensors
            block_indices: Which blocks to write to
            positions_in_block: Position within each block (0-15 for block_size=16)
        """
        for i, (block_idx, pos) in enumerate(zip(block_indices, positions_in_block)):
            self.k_cache[layer_id, block_idx, pos] = k[i]
            self.v_cache[layer_id, block_idx, pos] = v[i]

    def gather_kv(
        self,
        layer_id: int,
        block_indices: torch.Tensor,  # (seq_len,)
        positions_in_block: torch.Tensor,  # (seq_len,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K/V from cache blocks.

        Returns:
            k: (seq_len, num_kv_heads, head_dim)
            v: (seq_len, num_kv_heads, head_dim)
        """
        # Use advanced indexing to gather
        k = self.k_cache[layer_id, block_indices, positions_in_block]
        v = self.v_cache[layer_id, block_indices, positions_in_block]
        return k, v
```

### Memory Allocation Example

```python
# Example: Allocate memory for a request

# Request has 35 tokens with block_size=16
# Need ceil(35 / 16) = 3 blocks

num_tokens = 35
block_size = 16
num_blocks_needed = (num_tokens + block_size - 1) // block_size  # 3

# Allocate from pool
block_indices = kv_pool.alloc(num_blocks_needed)
# block_indices = [5, 12, 7]  (example)

# Map tokens to blocks
# Tokens 0-15 → Block 5, positions 0-15
# Tokens 16-31 → Block 12, positions 0-15
# Tokens 32-34 → Block 7, positions 0-2

token_to_block = []
token_positions = []

for token_idx in range(num_tokens):
    block_offset = token_idx // block_size
    position_in_block = token_idx % block_size

    token_to_block.append(block_indices[block_offset])
    token_positions.append(position_in_block)

# token_to_block = [5,5,5,...,5, 12,12,...,12, 7,7,7]
#                   ↑ 16 times    ↑ 16 times    ↑ 3 times

# Store in ReqToTokenPool
req_pool.set_token_positions(
    req_idx=req.req_pool_idx,
    token_positions=list(range(num_tokens)),
    kv_block_indices=token_to_block,
)
```

---

## Attention Implementation

### FlashInfer Backend

The [`flashinfer_backend.py`](../../python/sglang/srt/layers/attention/flashinfer_backend.py) implements paged attention with FlashInfer kernels:

```python
# Simplified from python/sglang/srt/layers/attention/flashinfer_backend.py

class FlashInferAttnBackend:
    """
    FlashInfer-based attention backend.

    Features:
    - Paged attention with non-contiguous KV cache
    - Fused RoPE (Rotary Position Embedding)
    - Optimized prefill and decode kernels
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Compute attention output.

        Args:
            q: Query tensor (num_tokens, num_heads, head_dim)
            k: Key tensor (num_tokens, num_kv_heads, head_dim)
            v: Value tensor (num_tokens, num_kv_heads, head_dim)
            layer: RadixAttention layer
            forward_batch: Batch information
            save_kv_cache: Whether to update KV cache

        Returns:
            Output tensor (num_tokens, num_heads, head_dim)
        """
        # Step 1: Save new K/V to cache (if needed)
        if save_kv_cache and k is not None:
            self._store_kv_cache(
                k, v,
                layer.layer_id,
                forward_batch,
            )

        # Step 2: Compute attention
        if forward_batch.forward_mode == ForwardMode.DECODE:
            # Decode: single token attends to full context
            output = self._decode_attention(q, layer, forward_batch)
        else:
            # Prefill: all tokens attend to each other
            output = self._prefill_attention(q, layer, forward_batch)

        return output

    def _store_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
    ):
        """
        Store K/V in paged KV cache.
        """
        # Get KV cache pool
        kv_pool = forward_batch.kv_pool

        # Get block indices for each request
        for req_idx, req in enumerate(forward_batch.reqs):
            # Get tokens for this request
            start_idx = forward_batch.req_start_indices[req_idx]
            end_idx = forward_batch.req_end_indices[req_idx]

            k_req = k[start_idx:end_idx]  # (num_tokens_req, num_kv_heads, head_dim)
            v_req = v[start_idx:end_idx]

            # Get KV block indices from req_to_token_pool
            kv_block_indices = forward_batch.req_to_token_pool.get_kv_indices(
                req.req_pool_idx,
                len(k_req),
            )

            # Calculate positions within blocks
            positions = torch.arange(len(k_req), device=k.device) % kv_pool.block_size

            # Store in cache
            kv_pool.store_kv(
                layer_id,
                k_req,
                v_req,
                kv_block_indices,
                positions,
            )

    def _decode_attention(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Decode attention: single query attends to full KV cache.

        Uses FlashInfer's decode kernel optimized for this pattern.
        """
        # Get attention metadata
        attn_metadata = forward_batch.attn_metadata

        # Call FlashInfer decode kernel
        output = flashinfer.batch_decode_with_padded_kv_cache(
            q,
            forward_batch.kv_pool.k_cache,
            forward_batch.kv_pool.v_cache,
            kv_indptr=attn_metadata.kv_indptr,  # Request boundaries
            kv_indices=attn_metadata.kv_indices,  # Block indices
            kv_last_page_len=attn_metadata.kv_last_page_len,  # Last block usage
            sm_scale=layer.scaling,
            rope_scale=attn_metadata.rope_scale,
            rope_theta=attn_metadata.rope_theta,
        )

        return output

    def _prefill_attention(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Prefill attention: all tokens attend to each other + cached prefix.

        Uses FlashInfer's prefill kernel with paged KV cache support.
        """
        attn_metadata = forward_batch.attn_metadata

        # Call FlashInfer prefill kernel
        output = flashinfer.batch_prefill_with_paged_kv_cache(
            q,
            forward_batch.kv_pool.k_cache,
            forward_batch.kv_pool.v_cache,
            qo_indptr=attn_metadata.qo_indptr,  # Query boundaries
            kv_indptr=attn_metadata.kv_indptr,  # KV boundaries
            kv_indices=attn_metadata.kv_indices,  # Block indices
            kv_last_page_len=attn_metadata.kv_last_page_len,
            sm_scale=layer.scaling,
            causal=True,  # Causal masking for decoder
            rope_scale=attn_metadata.rope_scale,
            rope_theta=attn_metadata.rope_theta,
        )

        return output
```

### Attention Metadata Construction

```python
# From python/sglang/srt/model_executor/model_runner.py

def _build_attention_metadata(
    self,
    model_worker_batch: ModelWorkerBatch,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> AttentionMetadata:
    """
    Build attention metadata for FlashInfer kernels.

    Returns:
        AttentionMetadata with:
        - qo_indptr: Query boundaries (for prefill)
        - kv_indptr: KV boundaries
        - kv_indices: Block indices for each token
        - kv_last_page_len: Usage in last block
    """
    batch_size = len(model_worker_batch.reqs)

    # Build query boundaries (cumulative sum)
    qo_indptr = [0]
    for seq_len in seq_lens:
        qo_indptr.append(qo_indptr[-1] + seq_len.item())
    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)

    # Build KV boundaries
    kv_indptr = [0]
    kv_indices_list = []
    kv_last_page_len_list = []

    for req in model_worker_batch.reqs:
        # Get KV block indices for this request
        kv_blocks = self.req_to_token_pool.get_kv_indices(
            req.req_pool_idx,
            seq_len=len(req.origin_input_ids) + len(req.output_ids),
        )

        # Append to global list
        kv_indices_list.extend(kv_blocks.tolist())

        # Update boundary
        kv_indptr.append(kv_indptr[-1] + len(kv_blocks))

        # Calculate last page usage
        total_tokens = len(req.origin_input_ids) + len(req.output_ids)
        last_page_len = total_tokens % self.kv_pool.block_size
        if last_page_len == 0:
            last_page_len = self.kv_pool.block_size
        kv_last_page_len_list.append(last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=self.device)
    kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=self.device)
    kv_last_page_len = torch.tensor(
        kv_last_page_len_list,
        dtype=torch.int32,
        device=self.device,
    )

    return AttentionMetadata(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        rope_scale=1.0,
        rope_theta=10000.0,
    )
```

---

## Scheduling Algorithms

### LPM (Longest Prefix Match) Implementation

```python
# Detailed implementation from python/sglang/srt/managers/schedule_policy.py

class SchedulePolicy:
    def _schedule_lpm_detailed(
        self,
        waiting_queue: List[Req],
        available_blocks: int,
    ) -> ScheduleBatch:
        """
        Longest Prefix Match scheduling with detailed steps.

        Algorithm:
        1. For each waiting request, match against RadixCache
        2. Calculate match score = matched_len / total_len
        3. Sort by score (high to low)
        4. Greedily select until OOM
        5. Optional: In-batch prefix caching optimization
        """
        # Step 1: Score all requests
        scored_requests = []

        for req in waiting_queue:
            # Build cache key
            cache_key = RadixKey(
                token_ids=req.origin_input_ids,
                extra_key=f"lora_{req.sampling_params.lora_id}",
            )

            # Match prefix
            match_result = self.tree_cache.match_prefix(cache_key)

            # Calculate score
            matched_len = match_result.matched_len
            total_len = len(req.origin_input_ids)
            score = matched_len / total_len if total_len > 0 else 0.0

            scored_requests.append({
                "req": req,
                "score": score,
                "matched_len": matched_len,
                "matched_indices": match_result.matched_indices,
                "last_node": match_result.last_node,
            })

        # Step 2: Sort by score (descending)
        scored_requests.sort(key=lambda x: x["score"], reverse=True)

        # Step 3: Greedy selection
        batch = ScheduleBatch()
        used_blocks = 0

        for item in scored_requests:
            req = item["req"]
            matched_len = item["matched_len"]

            # Calculate blocks needed
            unmatched_len = len(req.origin_input_ids) - matched_len
            needed_blocks = (unmatched_len + self.block_size - 1) // self.block_size

            # Check if fits
            if used_blocks + needed_blocks <= available_blocks:
                # Allocate blocks
                new_blocks = self.token_to_kv_pool.alloc(needed_blocks)

                # Set request state
                req.prefix_indices = item["matched_indices"]
                req.kv_indices = item["matched_indices"] + new_blocks
                req.last_node = item["last_node"]

                # Add to batch
                batch.reqs.append(req)
                used_blocks += needed_blocks
            else:
                # OOM, stop
                break

        # Step 4: In-batch prefix caching (optional)
        if self.enable_in_batch_prefix_caching:
            batch = self._optimize_in_batch_prefix(batch)

        return batch

    def _optimize_in_batch_prefix(
        self,
        batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """
        Optimize batch for in-batch prefix caching.

        If multiple requests in the batch share a common prefix,
        we can compute it once and reuse for all.
        """
        # Build temporary radix tree from batch
        temp_tree = RadixCache.create_simulated()

        for req in batch.reqs:
            temp_tree.insert(
                RadixKey(req.origin_input_ids),
                value=None,  # Simulated, no actual blocks
            )

        # Find shared prefixes
        shared_prefixes = temp_tree.find_common_prefixes(min_length=32)

        # Reorder batch to compute shared prefixes first
        if shared_prefixes:
            # Group requests by shared prefix
            groups = defaultdict(list)
            for req in batch.reqs:
                prefix_id = self._find_prefix_group(req, shared_prefixes)
                groups[prefix_id].append(req)

            # Flatten groups (prefix groups come first)
            reordered_reqs = []
            for prefix_id, reqs in groups.items():
                reordered_reqs.extend(reqs)

            batch.reqs = reordered_reqs

        return batch
```

---

## CUDA Graph Capture

### Graph Capture Process

```python
# From python/sglang/srt/model_executor/model_runner.py

class CudaGraphRunner:
    """
    Captures and replays CUDA graphs for decode steps.

    Benefits:
    - Eliminates kernel launch overhead (~10-50us per kernel)
    - Reduces CPU-GPU synchronization
    - 2-3x speedup for small batch decode

    Limitations:
    - Only works for fixed batch sizes
    - Only for decode mode (not prefill)
    - Memory addresses must be stable
    """

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.graphs = {}  # batch_size -> CUDAGraph
        self.graph_inputs = {}  # batch_size -> input tensors
        self.graph_outputs = {}  # batch_size -> output tensors

    def capture(self, batch_size: int):
        """
        Capture CUDA graph for a specific batch size.
        """
        # Create dummy forward batch
        dummy_batch = self._create_dummy_batch(batch_size)

        # Warmup: run forward pass a few times
        for _ in range(3):
            _ = self.model_runner.model(dummy_batch)
        torch.cuda.synchronize()

        # Allocate persistent input/output buffers
        graph_inputs = self._allocate_graph_inputs(batch_size)
        graph_outputs = self._allocate_graph_outputs(batch_size)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Run forward pass
            # All operations are recorded into the graph
            output = self.model_runner.model(dummy_batch)

            # Copy to persistent output buffer
            graph_outputs["logits"].copy_(output)

        # Store graph and buffers
        self.graphs[batch_size] = graph
        self.graph_inputs[batch_size] = graph_inputs
        self.graph_outputs[batch_size] = graph_outputs

        logger.info(f"Captured CUDA graph for batch_size={batch_size}")

    def replay(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """
        Replay captured graph with new inputs.

        Process:
        1. Copy inputs to persistent buffers
        2. Replay graph (all kernels launch instantly)
        3. Copy outputs from persistent buffers
        """
        batch_size = len(forward_batch.reqs)

        # Check if graph exists
        if batch_size not in self.graphs:
            # Capture on first use
            self.capture(batch_size)

        # Get graph and buffers
        graph = self.graphs[batch_size]
        inputs = self.graph_inputs[batch_size]
        outputs = self.graph_outputs[batch_size]

        # Copy inputs to persistent buffers
        inputs["input_ids"].copy_(forward_batch.input_ids)
        inputs["positions"].copy_(forward_batch.positions)
        inputs["req_pool_indices"].copy_(forward_batch.req_pool_indices)
        # ... copy other inputs

        # Replay graph
        graph.replay()

        # Copy outputs
        logits = outputs["logits"].clone()

        return logits

    def _create_dummy_batch(self, batch_size: int) -> ForwardBatch:
        """Create dummy batch for graph capture."""
        # All dummy data must have stable memory addresses
        input_ids = torch.zeros((batch_size,), dtype=torch.int64, device="cuda")
        positions = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        # ... other dummy tensors

        dummy_batch = ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            req_pool_indices=req_pool_indices,
            forward_mode=ForwardMode.DECODE,
            # ... other fields
        )

        return dummy_batch
```

---

## Distributed Communication

### Tensor Parallel All-Reduce

```python
# From python/sglang/srt/distributed/parallel_state.py

def tensor_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce across tensor parallel group.

    Used for:
    - Column-parallel linear: reduce outputs
    - Row-parallel linear: reduce inputs
    """
    if get_tensor_model_parallel_world_size() == 1:
        # No TP, skip
        return tensor

    # Use NCCL all-reduce
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )

    return tensor

# Example usage in linear layer:

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer.

    Input: [batch_size, in_features]
    Weight: [out_features / tp_size, in_features]
    Output: [batch_size, out_features / tp_size]

    After all-reduce: [batch_size, out_features]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul (each GPU has 1/tp_size columns)
        output = F.linear(x, self.weight, self.bias)

        # All-reduce to combine results
        output = tensor_parallel_all_reduce(output)

        return output
```

### Pipeline Parallel Send/Recv

```python
# From python/sglang/srt/managers/scheduler_pp_mixin.py

class SchedulerPPMixin:
    """
    Pipeline parallelism mixin for Scheduler.

    Implements send/recv of activations between pipeline stages.
    """

    def send_to_next_stage(self, hidden_states: torch.Tensor):
        """Send activations to next pipeline stage."""
        if self.pp_rank == self.pp_size - 1:
            # Last stage, no send
            return

        # Get next stage's rank in world group
        next_rank = self.pp_rank + 1

        # Send tensor via NCCL
        torch.distributed.send(
            hidden_states,
            dst=next_rank,
            group=get_pp_group(),
        )

    def recv_from_prev_stage(self) -> torch.Tensor:
        """Receive activations from previous pipeline stage."""
        if self.pp_rank == 0:
            # First stage, no receive
            return None

        # Get previous stage's rank
        prev_rank = self.pp_rank - 1

        # Allocate buffer
        hidden_states = torch.empty(
            self.hidden_states_shape,
            dtype=self.dtype,
            device=self.device,
        )

        # Receive tensor via NCCL
        torch.distributed.recv(
            hidden_states,
            src=prev_rank,
            group=get_pp_group(),
        )

        return hidden_states
```

---

## Complete Request Trace

Let's trace a complete request with actual data:

```python
# Request: "What is the capital of France?"
# Tokens: [1, 1724, 374, 279, 6864, 315, 9822, 30]  (length=8)

# ===== Step 1: HTTP Request =====
# File: python/sglang/srt/entrypoints/http_server.py

POST /v1/completions
{
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.7
}

# ===== Step 2: Tokenization =====
# File: python/sglang/srt/managers/tokenizer_manager.py

token_ids = tokenizer.encode("What is the capital of France?")
# token_ids = [1, 1724, 374, 279, 6864, 315, 9822, 30]

# ===== Step 3: Create Request Object =====
# File: python/sglang/srt/managers/schedule_batch.py

req = Req(
    rid="req_12345",
    origin_input_text="What is the capital of France?",
    origin_input_ids=[1, 1724, 374, 279, 6864, 315, 9822, 30],
    sampling_params=SamplingParams(
        max_new_tokens=50,
        temperature=0.7,
    ),
)

# ===== Step 4: Scheduler Receives Request =====
# File: python/sglang/srt/managers/scheduler.py

# Add to waiting queue
waiting_queue.append(req)

# ===== Step 5: Prefix Matching =====
# File: python/sglang/srt/mem_cache/radix_cache.py

# Assume "What is the capital" was previously cached
# Tokens [1, 1724, 374, 279, 6864] are in cache

cache_key = RadixKey(token_ids=[1, 1724, 374, 279, 6864, 315, 9822, 30])
match = tree_cache.match_prefix(cache_key)

# match.matched_len = 5  ("What is the capital")
# match.matched_indices = [2, 2, 2, 2, 3]  (KV block indices)
#   Tokens 0-15 → Block 2
#   Token 16 → Block 3 (partial)

# Unmatched suffix: [315, 9822, 30]  ("of France?")
# Need 1 more block for 3 tokens

# ===== Step 6: Allocate KV Cache =====

new_blocks = token_to_kv_pool.alloc(1)  # [17]
req.prefix_indices = [2, 2, 2, 2, 3]
req.kv_indices = [2, 2, 2, 2, 3, 17, 17, 17]

req.req_pool_idx = req_to_token_pool.allocate()  # idx=7

# Update req_to_token_pool
req_to_token_pool.set_token_positions(
    req_idx=7,
    token_positions=[0, 1, 2, 3, 4, 5, 6, 7],
    kv_block_indices=[2, 2, 2, 2, 3, 17, 17, 17],
)

# ===== Step 7: Build ScheduleBatch =====

batch = ScheduleBatch(
    reqs=[req],
    forward_mode=ForwardMode.EXTEND,  # Prefill
)

# ===== Step 8: Forward Pass (TpModelWorker) =====
# File: python/sglang/srt/managers/tp_worker.py

model_worker_batch = batch.to_model_worker_batch()

# ===== Step 9: Prepare ForwardBatch =====
# File: python/sglang/srt/model_executor/model_runner.py

forward_batch = ForwardBatch(
    input_ids=torch.tensor([1, 1724, 374, 279, 6864, 315, 9822, 30], device="cuda"),
    req_pool_indices=torch.tensor([7], device="cuda"),
    seq_lens=torch.tensor([8], device="cuda"),
    prefix_lens=torch.tensor([5], device="cuda"),  # 5 tokens cached
    forward_mode=ForwardMode.EXTEND,
    # ... other fields
)

# ===== Step 10: Model Forward =====

# Embedding
hidden = embed_tokens(input_ids)  # (8, 4096)

# Layer 0 Attention
q, k, v = attn_qkv_proj(hidden)  # (8, num_heads, head_dim)

# Store K/V for NEW tokens (indices 5-7)
# Tokens 0-4 already in cache
kv_pool.store_kv(
    layer_id=0,
    k=k[5:],  # Only new tokens
    v=v[5:],
    block_indices=[17, 17, 17],
    positions=[0, 1, 2],
)

# Retrieve FULL K/V (including cached prefix)
k_full = kv_pool.gather_kv(layer_id=0, kv_indices=[2,2,2,2,3,17,17,17])
v_full = kv_pool.gather_kv(layer_id=0, kv_indices=[2,2,2,2,3,17,17,17])

# Compute attention
attn_out = flashinfer_attention(q, k_full, v_full)  # (8, hidden_dim)

# ... continue through all layers ...

# Final logits
logits = lm_head(hidden[-1])  # (vocab_size,)

# ===== Step 11: Sampling =====
# File: python/sglang/srt/layers/sampler.py

# Sample next token
probs = softmax(logits / temperature)
next_token = torch.multinomial(probs, num_samples=1)  # e.g., 791 ("Paris")

req.output_ids.append(791)
req.stage = RequestStage.DECODE

# ===== Step 12: Decode Loop (Iteration 2) =====

# Update batch for decode
forward_batch = ForwardBatch(
    input_ids=torch.tensor([791], device="cuda"),  # Only last token
    req_pool_indices=torch.tensor([7], device="cuda"),
    seq_lens=torch.tensor([9], device="cuda"),  # 8 prompt + 1 generated
    forward_mode=ForwardMode.DECODE,
)

# Forward pass (CUDA graph replay)
logits = cuda_graph_runner.replay(forward_batch)

# Sample
next_token = sample(logits)  # e.g., 13 (".")
req.output_ids.append(13)

# ===== Step 13: Check Finish Condition =====

if next_token == eos_token_id:
    req.finished = True
    req.finish_reason = FINISH_MATCHED_TOKEN(eos_token_id)

# ===== Step 14: Detokenization =====
# File: python/sglang/srt/managers/detokenizer_manager.py

output_text = tokenizer.decode(req.output_ids)
# "Paris."

# ===== Step 15: Return Response =====

response = {
    "id": "req_12345",
    "object": "text_completion",
    "created": 1234567890,
    "model": "meta-llama/Llama-3-8b",
    "choices": [{
        "text": "Paris.",
        "index": 0,
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }
}

# ===== Final State =====

# RadixCache updated:
#   New branch for "What is the capital of France?" → "Paris."
#   Next request with "What is the capital" prefix reuses blocks [2,2,2,2,3]

# Memory usage:
#   Blocks: [2] (16 tokens), [3] (16 tokens), [17] (3 tokens)
#   Total: 35 tokens stored, 10 generated
#   Cache hit rate: 62.5% (5/8 tokens)
```

---

## Conclusion

This detailed code walkthrough demonstrates SGLang's sophisticated implementation of:

1. **Efficient Memory Management**: Paged KV cache with radix tree-based prefix caching
2. **Flexible Scheduling**: Multiple policies balancing cache efficiency and fairness
3. **Optimized Execution**: CUDA graphs, fused kernels, and pluggable attention backends
4. **Distributed Scalability**: Full TP/PP/MoE/DP support with NCCL communication

The codebase showcases production-grade engineering with careful attention to:
- Memory efficiency (shared blocks, eviction policies)
- Performance optimization (CUDA graphs, kernel fusion)
- Flexibility (pluggable backends, multiple scheduling policies)
- Observability (metrics, profiling, tracing)

These design choices enable SGLang to achieve state-of-the-art throughput and latency for LLM inference workloads.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-30
**Companion Document:** `sglang_architecture_deep_dive.md`
