# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU KV-cache block eviction policies."""

import heapq
from abc import ABC

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class GPUKVCacheEvictionPolicy(ABC):
    """Policy interface for choosing free GPU KV-cache blocks for reuse."""

    def allocate_blocks(
        self, free_blocks: FreeKVCacheBlockQueue, num_blocks: int
    ) -> list[KVCacheBlock]:
        """Select free blocks in eviction order."""
        return free_blocks.popleft_n(num_blocks)

    def on_cache_insert(self, block: KVCacheBlock) -> None:
        """Record a newly cached block."""

    def on_block_access(self, block: KVCacheBlock) -> None:
        """Record a prefix-cache hit."""

    def on_block_freed(self, block: KVCacheBlock) -> None:
        """Record that a cached block became evictable."""

    def on_cache_remove(self, block: KVCacheBlock) -> None:
        """Forget a block whose cache entry was removed."""

    def clear(self) -> None:
        """Forget all policy state."""


class LRUEvictionPolicy(GPUKVCacheEvictionPolicy):
    """Least-recently-used policy implemented by the existing free-block queue."""


class LFUEvictionPolicy(GPUKVCacheEvictionPolicy):
    """Least-frequently-used policy, with LRU tie-breaking."""

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.blocks = blocks
        self.frequencies: dict[int, int] = {}
        self._last_access: dict[int, int] = {}
        self._versions: dict[int, int] = {}
        self._evictable: set[int] = set()
        self._heap: list[tuple[int, int, int, int]] = []
        self._clock = 0

    def _record(self, block_id: int, frequency: int) -> None:
        self._clock += 1
        version = self._versions.get(block_id, 0) + 1
        self.frequencies[block_id] = frequency
        self._last_access[block_id] = self._clock
        self._versions[block_id] = version
        if block_id in self._evictable:
            heapq.heappush(
                self._heap,
                (frequency, self._clock, version, block_id),
            )

    def on_cache_insert(self, block: KVCacheBlock) -> None:
        self._record(block.block_id, 1)
        if block.ref_cnt == 0:
            self.on_block_freed(block)

    def on_block_access(self, block: KVCacheBlock) -> None:
        block_id = block.block_id
        frequency = self.frequencies.get(block_id)
        if frequency is None:
            return
        self._evictable.discard(block_id)
        self._record(block_id, frequency + 1)
        self._maybe_compact_heap()

    def on_block_freed(self, block: KVCacheBlock) -> None:
        block_id = block.block_id
        if (
            block_id not in self.frequencies
            or block.ref_cnt != 0
            or block.block_hash is None
            or block_id in self._evictable
        ):
            return
        self._evictable.add(block_id)
        heapq.heappush(
            self._heap,
            (
                self.frequencies[block_id],
                self._last_access[block_id],
                self._versions[block_id],
                block_id,
            ),
        )
        self._maybe_compact_heap()

    def on_cache_remove(self, block: KVCacheBlock) -> None:
        block_id = block.block_id
        self.frequencies.pop(block_id, None)
        self._last_access.pop(block_id, None)
        self._versions.pop(block_id, None)
        self._evictable.discard(block_id)
        self._maybe_compact_heap()

    def clear(self) -> None:
        self.frequencies.clear()
        self._last_access.clear()
        self._versions.clear()
        self._evictable.clear()
        self._heap.clear()
        self._clock = 0

    def allocate_blocks(
        self, free_blocks: FreeKVCacheBlockQueue, num_blocks: int
    ) -> list[KVCacheBlock]:
        allocated: list[KVCacheBlock] = []
        while len(allocated) < num_blocks:
            first = free_blocks.fake_free_list_head.next_free_block
            if first is None or first is free_blocks.fake_free_list_tail:
                raise ValueError("No free blocks available")
            if first.block_hash is None:
                allocated.append(free_blocks.popleft())
                continue

            block = self._pop_lfu_candidate()
            if block is None:
                allocated.extend(
                    free_blocks.popleft_n(num_blocks - len(allocated))
                )
                break
            free_blocks.remove(block)
            self._evictable.discard(block.block_id)
            allocated.append(block)
        return allocated

    def _pop_lfu_candidate(self) -> KVCacheBlock | None:
        while self._heap:
            frequency, last_access, version, block_id = heapq.heappop(self._heap)
            if (
                self.frequencies.get(block_id) != frequency
                or self._last_access.get(block_id) != last_access
                or self._versions.get(block_id) != version
                or block_id not in self._evictable
            ):
                continue
            block = self.blocks[block_id]
            if block.ref_cnt != 0 or block.block_hash is None:
                self._evictable.discard(block_id)
                continue
            return block
        return None

    def _maybe_compact_heap(self) -> None:
        if len(self._heap) < max(64, 2 * len(self._evictable)):
            return
        self._heap = [
            (
                self.frequencies[block_id],
                self._last_access[block_id],
                self._versions[block_id],
                block_id,
            )
            for block_id in self._evictable
        ]
        heapq.heapify(self._heap)


def create_gpu_kv_cache_eviction_policy(
    name: str, blocks: list[KVCacheBlock]
) -> GPUKVCacheEvictionPolicy:
    """Create a GPU KV-cache eviction policy by its configured name."""
    if name == "lru":
        return LRUEvictionPolicy()
    if name == "lfu":
        return LFUEvictionPolicy(blocks)
    raise ValueError(f"Unknown GPU KV-cache eviction policy: {name!r}")
