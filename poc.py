import os
import time

# Set before importing vllm. These are fallbacks if your venv doesn't already set them.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")           # GiB
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine runs in this process

from vllm import LLM, SamplingParams
from vllm.v1.core.eviction_policy import LFUEvictionPolicy

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Short, deterministic generation: 16 new tokens per prompt.
sampling_params = SamplingParams(temperature=0.0, max_tokens=16)


def main():
    t0 = time.perf_counter()
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,   # skip torch.compile warmup
        max_model_len=256,    # tiny context, so a small KV cache is enough
        max_num_seqs=4,       # matches the number of prompts
        enable_prefix_caching=True,
        gpu_kv_cache_eviction_policy="lfu",
    )
    cache_config = llm.llm_engine.vllm_config.cache_config
    assert cache_config.gpu_kv_cache_eviction_policy == "lfu"

    engine_core = llm.llm_engine.engine_core.engine_core
    policy = engine_core.scheduler.kv_cache_manager.block_pool.eviction_policy
    assert isinstance(policy, LFUEvictionPolicy), type(policy).__name__

    print(f"Active GPU KV-cache policy: {type(policy).__name__}")
    t1 = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)
    t2 = time.perf_counter()

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        print(f"Prompt:    {output.prompt!r}")
        print(f"Output:    {output.outputs[0].text!r}")
        print("-" * 60)
    print(f"engine startup: {t1 - t0:.1f}s, generation: {t2 - t1:.1f}s")


if __name__ == "__main__":
    main()
