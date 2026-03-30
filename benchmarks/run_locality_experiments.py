# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache block locality 가설 검증 실험.

한 번 모델을 로드하고 여러 num_prompts 설정을 연속 측정한다.

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/run_locality_experiments.py \
        --model facebook/opt-125m \
        --prompt-counts 8 16 32 64 128

가설:
  H1. GQA 없는 모델(OPT)이 GQA 있는 모델보다 speedup 클 것
  H2. num_prompts가 L2 포화 임계 근방일 때 speedup 최대 (Goldilocks zone)
  H3. 작은 모델(memory-bound)이 큰 모델(compute-bound)보다 speedup 클 것
"""

import os
import random
import statistics
import time
import types

if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


# ── free list 접근 / patch / restitch (benchmark_kv_block_locality.py 와 동일) ──

def _get_free_block_queue(llm):
    return (llm.llm_engine.engine_core.engine_core
            .scheduler.kv_cache_manager.block_pool.free_block_queue)


def _print_block_info(llm):
    import torch
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    coordinator = scheduler.kv_cache_manager.coordinator
    kv_cache_config = coordinator.kv_cache_config
    spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    num_blocks = coordinator.block_pool.num_gpu_blocks
    # layer_names: 이 그룹에 속한 attention 레이어 이름 목록
    num_layers = sum(len(g.layer_names) for g in kv_cache_config.kv_cache_groups)

    dtype_bytes = torch.tensor([], dtype=spec.dtype).element_size()
    dtype_str = str(spec.dtype).replace("torch.", "")
    bytes_per_layer = spec.page_size_bytes          # 레이어 1개당 블록 크기
    bytes_per_block = bytes_per_layer * num_layers  # 논리 블록 1개의 실제 총 크기

    print(f"\n[KV Cache Info]")
    print(f"  block_size         : {spec.block_size} tokens")
    print(f"  num_kv_heads       : {spec.num_kv_heads}")
    print(f"  head_size          : {spec.head_size}")
    print(f"  dtype              : {dtype_str} ({dtype_bytes}B/elem)")
    print(f"  num_layers         : {num_layers}")
    print(f"  bytes/block/layer  : {bytes_per_layer / 1024:.1f} KB  "
          f"(= {spec.block_size}tok × {spec.num_kv_heads}heads "
          f"× {spec.head_size}dim × 2(K+V) × {dtype_bytes}B/{dtype_str})")
    print(f"  bytes/block (total): {bytes_per_block / 1024:.1f} KB  "
          f"(= {bytes_per_layer/1024:.1f} KB × {num_layers} layers)")
    print(f"  num_blocks         : {num_blocks:,}")
    print(f"  total KV mem       : {bytes_per_block * num_blocks / 1024**3:.2f} GiB")
    print()


def _restitch_queue(queue, blocks):
    if not blocks:
        queue.fake_free_list_head.next_free_block = queue.fake_free_list_tail
        queue.fake_free_list_tail.prev_free_block = queue.fake_free_list_head
        return
    queue.fake_free_list_head.next_free_block = blocks[0]
    blocks[0].prev_free_block = queue.fake_free_list_head
    for p, c in zip(blocks, blocks[1:]):
        p.next_free_block = c
        c.prev_free_block = p
    blocks[-1].next_free_block = queue.fake_free_list_tail
    queue.fake_free_list_tail.prev_free_block = blocks[-1]


def _install_patch(queue, mode):
    original = queue.popleft_n.__func__
    def patched(self, n):
        if n == 0:
            return []
        free_blocks = self.get_all_free_blocks()
        if mode == "sequential":
            free_blocks.sort(key=lambda b: b.block_id)
        else:
            random.shuffle(free_blocks)
        _restitch_queue(self, free_blocks)
        return original(self, n)
    queue.popleft_n = types.MethodType(patched, queue)
    return original


def _remove_patch(queue, original):
    queue.popleft_n = types.MethodType(original, queue)


def _fragment(llm, n):
    params = SamplingParams(temperature=0, max_tokens=4)
    llm.generate(["Hello"] * n, sampling_params=params)


def _frag_score(llm) -> float:
    """단편화 정도를 [0, 1]로 반환한다.

    free list의 인접 블록 ID 쌍 중 비연속적인(|id[i+1] - id[i]| != 1) 비율.
    0 = 완전 순차, 1 = 완전 무작위.
    """
    blocks = _get_free_block_queue(llm).get_all_free_blocks()
    if len(blocks) < 2:
        return 0.0
    ids = [b.block_id for b in blocks]
    non_sequential = sum(1 for a, b in zip(ids, ids[1:]) if abs(b - a) != 1)
    return non_sequential / (len(ids) - 1)


def _run_one(llm, prompts, sp, mode):
    q = _get_free_block_queue(llm)
    orig = _install_patch(q, mode)
    try:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sp)
        elapsed = time.perf_counter() - t0
    finally:
        _remove_patch(q, orig)
    total = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total / elapsed


def run_sweep(llm, prompt_counts, output_len, frag_requests,
              num_runs, warmup_runs):
    """num_prompts를 바꿔가며 sequential vs random speedup 측정."""
    sp = SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True)
    results = []  # (num_prompts, seq_mean, rnd_mean, speedup)

    for np_ in prompt_counts:
        prompts = ["The quick brown fox jumps over the lazy dog."] * np_
        print(f"\n  ── num_prompts={np_} ──")

        # 워밍업
        for mode in ("sequential", "random"):
            for _ in range(warmup_runs):
                _fragment(llm, frag_requests)
                _run_one(llm, prompts, sp, mode)

        seq_vals, rnd_vals = [], []
        for r in range(num_runs):
            frag_scores = []
            for mode in ("sequential", "random"):
                _fragment(llm, frag_requests)
                frag_scores.append(_frag_score(llm))
                tps = _run_one(llm, prompts, sp, mode)
                if mode == "sequential":
                    seq_vals.append(tps)
                else:
                    rnd_vals.append(tps)

            sp_ratio = seq_vals[-1] / rnd_vals[-1]
            frag_str = " ".join(f"{s:.3f}" for s in frag_scores)
            print(f"    run {r+1}: seq={seq_vals[-1]:>7,.0f} "
                  f"rnd={rnd_vals[-1]:>7,.0f}  speedup={sp_ratio:.4f}x"
                  f"  frag=[{frag_str}]")

        seq_mean = statistics.mean(seq_vals)
        rnd_mean = statistics.mean(rnd_vals)
        speedup  = seq_mean / rnd_mean
        results.append((np_, seq_mean, rnd_mean, speedup))
        print(f"    → mean speedup: {speedup:.4f}x")

    return results


def print_summary(model, results, output_len):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"모델: {model}  output_len={output_len}")
    print(f"{'num_prompts':>12} {'seq tok/s':>10} {'rnd tok/s':>10} {'speedup':>9}")
    print("-" * 60)
    for np_, seq, rnd, sp in results:
        bar = "█" * int(sp * 10 - 10) if sp > 1 else ""
        print(f"{np_:>12} {seq:>10,.0f} {rnd:>10,.0f} {sp:>8.4f}x  {bar}")
    print(sep)
    best = max(results, key=lambda x: x[3])
    print(f"  최대 speedup: {best[3]:.4f}x at num_prompts={best[0]}")
    print(sep)


def main(args):
    random.seed(42)
    print(f"\n모델 로딩: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    print("로딩 완료.")
    _print_block_info(llm)
    print("실험 시작...\n")

    results = run_sweep(
        llm,
        prompt_counts=args.prompt_counts,
        output_len=args.output_len,
        frag_requests=args.frag_requests,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
    )
    print_summary(args.model, results, args.output_len)


def create_parser():
    p = FlexibleArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompt-counts", type=int, nargs="+",
                   default=[8, 16, 32, 64])
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--frag-requests", type=int, default=150)
    return p


if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)
