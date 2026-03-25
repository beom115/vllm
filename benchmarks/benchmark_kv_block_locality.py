# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark KV cache block allocation locality.

KV cache는 [num_blocks, block_size, num_kv_heads, head_size] 형태의 GPU tensor이며,
attention kernel이 block_table을 통해 kv_cache[block_id, ...] 로 접근한다.

가설: block ID가 연속적(sequential)이면 GPU L2 캐시 적중률이 높아져 성능 향상이 있을 것.
     block ID가 분산(random)되면 캐시 미스가 증가해 느려질 것.

두 모드를 비교:
  sequential - 할당 전 free list를 block_id 오름차순 정렬 (연속 블록 반환)
  random     - 할당 전 free list를 무작위 셔플 (분산 블록 반환)

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/benchmark_kv_block_locality.py \\
        --model facebook/opt-125m \\
        --num-prompts 32 \\
        --output-len 256 \\
        --gpu-memory-utilization 0.85
"""

import os
import random
import statistics
import time
import types

# InprocClient를 사용해야 scheduler에 직접 접근 가능.
# VLLM_ENABLE_V1_MULTIPROCESSING=1(기본값)이면 EngineCore가 subprocess에서 실행되어
# 메인 프로세스에서 scheduler를 직접 건드릴 수 없다.
if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
    print("[WARNING] VLLM_ENABLE_V1_MULTIPROCESSING != 0")
    print("[WARNING] 이 벤치마크는 인프로세스 모드가 필요합니다.")
    print("[WARNING] 다음과 같이 실행하세요:")
    print("[WARNING]   VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/benchmark_kv_block_locality.py ...")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print("[WARNING] 자동으로 0으로 설정했습니다.\n")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


# ---------------------------------------------------------------------------
# Free list 조작 유틸리티
# ---------------------------------------------------------------------------

def _get_free_block_queue(llm: LLM):
    """LLM 객체에서 FreeKVCacheBlockQueue를 꺼낸다.

    경로: LLM -> LLMEngine -> InprocClient -> EngineCore
          -> Scheduler -> KVCacheManager -> BlockPool -> FreeKVCacheBlockQueue
    """
    inproc_client = llm.llm_engine.engine_core
    engine_core = inproc_client.engine_core
    block_pool = engine_core.scheduler.kv_cache_manager.block_pool
    return block_pool.free_block_queue


def _restitch_queue(queue, blocks: list) -> None:
    """이중 연결 리스트를 blocks 순서대로 재연결한다."""
    if not blocks:
        queue.fake_free_list_head.next_free_block = queue.fake_free_list_tail
        queue.fake_free_list_tail.prev_free_block = queue.fake_free_list_head
        return

    queue.fake_free_list_head.next_free_block = blocks[0]
    blocks[0].prev_free_block = queue.fake_free_list_head

    for prev_blk, curr_blk in zip(blocks, blocks[1:]):
        prev_blk.next_free_block = curr_blk
        curr_blk.prev_free_block = prev_blk

    blocks[-1].next_free_block = queue.fake_free_list_tail
    queue.fake_free_list_tail.prev_free_block = blocks[-1]


def _install_patch(queue, mode: str):
    """popleft_n에 monkey-patch를 설치하고 원래 함수를 반환한다.

    popleft_n 호출 전마다 free list를 mode에 따라 재정렬/셔플한다.
    sort/shuffle은 O(N log N)이지만 배치당 한 번씩만 호출되므로 허용 가능.
    """
    original_fn = queue.popleft_n.__func__  # unbound

    def patched_popleft_n(self, n: int) -> list:
        if n == 0:
            return []
        free_blocks = self.get_all_free_blocks()
        if mode == "sequential":
            free_blocks.sort(key=lambda b: b.block_id)
        else:  # random
            random.shuffle(free_blocks)
        _restitch_queue(self, free_blocks)
        return original_fn(self, n)

    queue.popleft_n = types.MethodType(patched_popleft_n, queue)
    return original_fn


def _remove_patch(queue, original_fn) -> None:
    """monkey-patch를 제거하고 원래 함수를 복원한다."""
    queue.popleft_n = types.MethodType(original_fn, queue)


# ---------------------------------------------------------------------------
# 벤치마크 로직
# ---------------------------------------------------------------------------

def _fragment_free_list(llm: LLM, num_requests: int) -> None:
    """tiny 요청을 많이 돌려 free list의 블록 순서를 뒤섞는다.

    각 요청이 몇 개의 블록을 할당했다가 free하면서 LRU 역순으로 free list 뒤에
    추가되므로, 블록 ID 순서가 뒤섞인 파편화 상태가 만들어진다.
    """
    prompts = ["Hello"] * num_requests
    params = SamplingParams(temperature=0, max_tokens=4)
    llm.generate(prompts, sampling_params=params)


def _run_one(llm: LLM,
             prompts: list[str],
             sampling_params: SamplingParams,
             mode: str) -> float:
    """주어진 모드로 한 번 추론을 돌리고 출력 tok/s를 반환한다."""
    queue = _get_free_block_queue(llm)
    original_fn = _install_patch(queue, mode)
    try:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed = time.perf_counter() - t0
    finally:
        _remove_patch(queue, original_fn)

    total_output_tokens = sum(
        len(o.outputs[0].token_ids) for o in outputs
    )
    return total_output_tokens / elapsed


def _print_results(results: dict[str, list[float]],
                   speedups: list[float],
                   args) -> None:
    sep = "=" * 70
    thin = "-" * 70
    print(f"\n{sep}")
    print(f"{'Mode':<12} {'Mean':>10} {'Std':>8} {'Min':>10} "
          f"{'Median':>10} {'Max':>10} {'N':>4}")
    print(thin)

    means = {}
    for mode, vals in results.items():
        mean   = statistics.mean(vals)
        std    = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lo     = min(vals)
        hi     = max(vals)
        med    = statistics.median(vals)
        means[mode] = mean
        print(f"{mode:<12} {mean:>10,.0f} {std:>8,.0f} {lo:>10,.0f} "
              f"{med:>10,.0f} {hi:>10,.0f} {len(vals):>4}")

    print(thin)
    if speedups:
        sp_mean = statistics.mean(speedups)
        sp_std  = statistics.stdev(speedups) if len(speedups) > 1 else 0.0
        sp_min  = min(speedups)
        sp_max  = max(speedups)
        sp_med  = statistics.median(speedups)
        wins    = sum(1 for s in speedups if s > 1.0)
        print(f"\n  Per-run speedup (sequential / random):")
        print(f"    mean={sp_mean:.4f}x  std={sp_std:.4f}  "
              f"min={sp_min:.4f}x  median={sp_med:.4f}x  max={sp_max:.4f}x")
        print(f"    sequential 우위 횟수: {wins}/{len(speedups)} runs "
              f"({'%.0f' % (wins/len(speedups)*100)}%)")
    print(sep)
    print(f"Config: model={args.model}, num_prompts={args.num_prompts}, "
          f"output_len={args.output_len}, num_runs={args.num_runs}, "
          f"frag_requests={args.frag_requests}")


def main(args) -> None:
    random.seed(args.seed)

    print(f"모델 로딩 중: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # prefix caching은 블록 재사용 경로가 달라 비교를 왜곡하므로 끈다
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        # EOS 토큰 무시: 모든 요청이 동일 길이로 끝나야 공정한 비교 가능
        ignore_eos=True,
    )

    # 짧은 고정 프롬프트 — prefill 변동 최소화
    prompts = ["The quick brown fox jumps over the lazy dog."] * args.num_prompts

    # 파편화 단계: 블록 ID가 섞인 상태를 만든다
    print(f"\nfree list 파편화 중 ({args.frag_requests}개 tiny 요청)...")
    _fragment_free_list(llm, args.frag_requests)
    print("파편화 완료.")

    results: dict[str, list[float]] = {"sequential": [], "random": []}
    speedups: list[float] = []

    # 워밍업 (CUDA graph 캡처, JIT 등 안정화)
    print(f"\n워밍업 ({args.warmup_runs}회)...")
    for mode in ("sequential", "random"):
        for _ in range(args.warmup_runs):
            _fragment_free_list(llm, args.frag_requests)
            _run_one(llm, prompts, sampling_params, mode)

    # 본 측정
    # ── 핵심: 각 run 직전에 파편화를 다시 걸어 독립성 보장 ──
    # sequential 실행 후 free list가 정렬된 상태로 돌아오기 때문에
    # 재파편화 없이는 이후 random 실행이 "이미 정렬된 상태에서 shuffle"
    # 하게 되어 두 모드가 수렴하는 현상이 생긴다.
    print(f"\n측정 시작 ({args.num_runs}회 × 2 모드, run마다 재파편화)...")
    for run_idx in range(args.num_runs):
        run_results: dict[str, float] = {}

        # 각 모드 측정 전 동일한 파편화 baseline으로 리셋
        for mode in ("random", "sequential"):
            _fragment_free_list(llm, args.frag_requests)
            tps = _run_one(llm, prompts, sampling_params, mode)
            results[mode].append(tps)
            run_results[mode] = tps

        sp = run_results["sequential"] / run_results["random"]
        speedups.append(sp)
        print(f"  Run {run_idx + 1:2d} | "
              f"seq {run_results['sequential']:>8,.0f} tok/s | "
              f"rnd {run_results['random']:>8,.0f} tok/s | "
              f"speedup {sp:.4f}x {'↑' if sp > 1.0 else '↓'}")

    _print_results(results, speedups, args)


def create_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="KV cache 블록 할당 locality 벤치마크 (sequential vs random)"
    )
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="테스트할 모델")
    parser.add_argument("--num-prompts", type=int, default=32,
                        help="배치당 요청 수")
    parser.add_argument("--output-len", type=int, default=256,
                        help="요청당 출력 토큰 수 (고정)")
    parser.add_argument("--max-model-len", type=int, default=1024,
                        help="최대 시퀀스 길이")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU 메모리 사용률 (WSL은 0.85 권장)")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="측정 반복 횟수")
    parser.add_argument("--warmup-runs", type=int, default=2,
                        help="워밍업 반복 횟수 (결과에 미포함)")
    parser.add_argument("--frag-requests", type=int, default=200,
                        help="파편화용 tiny 요청 수")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
