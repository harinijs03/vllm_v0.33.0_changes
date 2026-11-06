# GPU KV-Cache Eviction

## Existing LRU path

GPU KV-cache prefix blocks are managed by [`BlockPool`](../../vllm/v1/core/block_pool.py).
Its `FreeKVCacheBlockQueue` is a doubly linked list that serves both allocation
and LRU ordering:

- `BlockPool.get_new_blocks` takes blocks from the front and calls
  `_maybe_evict_cached_block` before reuse.
- `BlockPool.touch` handles prefix-cache hits: it removes free blocks from the
  queue and increments their reference counts.
- `BlockPool.free_blocks` puts uncached blocks at the front for reuse, and
  appends cached blocks at the tail. Callers order freed request blocks so
  older cached blocks are evicted first.
- `BlockPool._insert_block_hash` makes full or partial block hashes available
  for lookups. `_remove_cached_block_hashes` removes every hash owned by a
  block and resets its hash metadata when the block is reused.
- `FreeKVCacheBlockQueue.popleft_n`, `remove`, `prepend_n`, and `append_n` in
  [`kv_cache_utils.py`](../../vllm/v1/core/kv_cache_utils.py) provide the
  constant-time linked-list operations used by allocation, hits, and frees.

The call chain is `Scheduler.__init__` → `KVCacheManager.__init__` →
`get_kv_cache_coordinator` → one of the coordinator constructors →
`KVCacheCoordinator.__init__`, which creates the shared `BlockPool`. The
scheduler supplies the eviction setting from `CacheConfig`; LRU remains the
default at every layer.

On lookup, `SingleTypeKVCacheManager.find_longest_cache_hit` calls
`BlockPool.get_cached_block`. `KVCacheManager` later sends matched blocks to
`BlockPool.touch`, which is where an access is recorded. New hashes flow
through `BlockPool.cache_full_blocks` and `_insert_block_hash`. This makes
`touch`, rather than the lower-level hash lookup, the single point where an
actual hit increments frequency.

## LFU policy

Set `gpu_kv_cache_eviction_policy` to `"lfu"` (or use
`--gpu-kv-cache-eviction-policy lfu`) to select LFU for cached GPU blocks.
Newly cached blocks start with frequency 1. Each prefix-cache hit increments
the frequency. The least frequently used evictable block is selected first;
least-recent access breaks ties. Uncached blocks retain the existing preference
for immediate reuse. Reference-counted blocks are not eligible for eviction.

[`eviction_policy.py`](../../vllm/v1/core/eviction_policy.py) defines the
policy extension point and LFU's frequency map, evictable set, and lazy
min-heap. Heap versions make old entries harmless after a hit or state change.
`BlockPool` notifies the policy on cache insertion, access, free, removal, and
reset, and delegates block selection to it. The default policy uses the
existing queue ordering unchanged.

## Files changed

- `vllm/v1/core/eviction_policy.py` — policy interface, default LRU behavior,
  LFU tracking, and strategy factory.
- `vllm/v1/core/block_pool.py` — policy lifecycle hooks and policy-based block
  selection.
- `vllm/config/cache.py` — validated `gpu_kv_cache_eviction_policy` setting;
  excluded from compile-graph hashing because it only changes runtime eviction.
- `vllm/engine/arg_utils.py` — exposes the setting through `EngineArgs`, passes
  it into `CacheConfig`, and registers the CLI flag.
- `vllm/v1/core/kv_cache_manager.py` — forwards the policy into the coordinator.
- `vllm/v1/core/kv_cache_coordinator.py` — passes the policy when creating the
  shared GPU block pool.
- `vllm/v1/core/sched/scheduler.py` — connects the user setting to the manager.
- `tests/v1/core/test_prefix_caching.py` — verifies LFU eviction and hit
  frequency updates.
- `tests/config/test_config_utils.py` — verifies eviction choice does not
  affect the compile-graph hash.
- `tests/engine/test_arg_utils.py` — verifies default and CLI argument parsing.
- `docs/configuration/optimization.md` — documents CLI and configuration use.
- `docs/design/gpu_kv_cache_eviction.md` — implementation map and extension
  notes.
