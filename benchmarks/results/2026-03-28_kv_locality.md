# KV Cache Block Locality Benchmark

**Date**: 2026-03-28
**GPU**: RTX 4060 8GB (WSL2)
**vLLM**: v0.1.dev13749+g7f5b601b2
**Settings**: `output_len=256`, `num_runs=3`, `warmup_runs=1`, `frag_requests=150`, `gpu_memory_utilization=0.85`, `enable_prefix_caching=False`

## 목적

vLLM의 KV cache 블록을 순차적(sequential)으로 할당할 때와 무작위(random)로 할당할 때의 throughput 차이를 측정한다.

- **Sequential**: 매 할당 시 free list를 block_id 오름차순으로 정렬 후 pop
- **Random**: 매 할당 시 free list를 무작위로 shuffle 후 pop

Speedup = sequential tok/s ÷ random tok/s

## 모델 아키텍처

| 모델 | GQA | num_kv_heads | head_dim | num_layers | bytes/block/layer | bytes/block (total) | num_blocks |
|------|-----|-------------|----------|------------|-------------------|----------------------|-----------|
| facebook/opt-125m | No | 12 | 64 | 12 | 48.0 KB | 576.0 KB | 11,014 |
| meta-llama/Llama-3.2-1B | Yes | 8 | 64 | 16 | 32.0 KB | 512.0 KB | 6,649 |
| Qwen/Qwen3-0.6B | Yes | 8 | 128 | 28 | 64.0 KB | 1,792.0 KB | 2,485 |
| Qwen/Qwen3-1.7B | Yes | 8 | 128 | 28 | 64.0 KB | 1,792.0 KB | 1,249 |

## 결과

### facebook/opt-125m

| num_prompts | seq (tok/s) | rnd (tok/s) | speedup | frag [rnd, seq] |
|-------------|------------|------------|---------|-----------------|
| 8 | 3,301 | 2,325 | **1.420x** | [1.000, 0.012] |
| 16 | 5,062 | 3,138 | **1.613x** | [1.000, 0.025] |
| 32 | 6,185 | 3,522 | **1.756x** | [1.000, 0.049] |
| 64 | 7,004 | 3,859 | **1.815x** | [1.000, 0.099] |
| 128 | 8,418 | 4,425 | **1.902x** | [1.000, 0.198] |

### meta-llama/Llama-3.2-1B

| num_prompts | seq (tok/s) | rnd (tok/s) | speedup | frag [rnd, seq] |
|-------------|------------|------------|---------|-----------------|
| 8 | 645 | 631 | 1.022x | [0.999, 0.021] |
| 16 | 1,161 | 1,105 | 1.051x | [1.000, 0.041] |
| 32 | 2,199 | 1,917 | 1.147x | [1.000, 0.082] |
| 64 | 3,575 | 2,879 | **1.242x** | [0.999, 0.164] |
| 128 | 5,012 | 3,997 | **1.254x** | [1.000, 0.328] |

### Qwen/Qwen3-0.6B

| num_prompts | seq (tok/s) | rnd (tok/s) | speedup | frag [rnd, seq] |
|-------------|------------|------------|---------|-----------------|
| 8 | 1,152 | 1,144 | 1.007x | [0.999, 0.055] |
| 16 | 2,084 | 2,018 | 1.033x | [0.999, 0.110] |
| 32 | 3,244 | 3,048 | 1.064x | [0.999, 0.219] |
| 64 | 4,939 | 4,536 | 1.089x | [0.999, 0.439] |
| 128 | 6,417 | 5,846 | **1.098x** | [0.999, 0.877] |

### Qwen/Qwen3-1.7B

| num_prompts | seq (tok/s) | rnd (tok/s) | speedup | frag [rnd, seq] |
|-------------|------------|------------|---------|-----------------|
| 8 | 483 | 485 | 0.997x | [0.999, 0.110] |
| 16 | 915 | 913 | 1.002x | [0.999, 0.219] |
| 32 | 1,706 | 1,704 | 1.002x | [0.998, 0.437] |
| 64 | 2,926 | 2,910 | 1.005x | [0.998, 0.873] |
| 128 | 3,082 | 3,062 | 1.007x | [0.999, 0.032] |

## 분석

### 가설 검증

**H1. GQA 없는 모델이 더 큰 speedup을 가질 것** ✅ 확인

GQA가 없는 OPT-125m이 최대 1.90x로 압도적. GQA 모델은 KV head 수가 줄어 블록당 메모리가 작고, 여러 query head가 같은 KV를 공유해 locality 이득이 감소함.

**H2. num_prompts에 Goldilocks zone 존재** ❌ 기각

OPT-125m, Llama-3.2-1B 모두 num_prompts=128에서 최대로 단조 증가. L2 캐시 포화 후에도 speedup이 꺾이지 않음. 실제 메커니즘은 DRAM row buffer hit rate / 메모리 코얼레싱으로, 배치가 클수록 이득이 누적됨.

**H3. 작은/memory-bound 모델이 더 큰 speedup을 가질 것** ✅ 부분 확인

모델이 커질수록 speedup 감소하는 경향. 단, 모델 크기보다 **GQA 여부와 head_dim이 더 결정적 요인**.

### frag score 해석

`frag=[random 전, sequential 전]`

- **frag[0] ≈ 1.000**: fragmentation 후 free list가 완전히 무작위화됨 (정상)
- **frag[1] ≈ num_prompts × output_len / block_size / num_blocks**: random run이 사용한 블록들이 free list 끝에 scrambled 순서로 반환된 흔적. `_fragment(150)`은 전체 블록 중 일부만 건드리므로 이 흔적이 남음

Qwen3-1.7B @ num_prompts=128에서 frag[1]=0.032로 예외적으로 낮음 → 사용 블록(128×16=2,048)이 num_blocks(1,249)를 초과하여 스케줄러가 여러 배치로 나눠 실행, 블록이 지속적으로 재활용되어 반환 패턴이 복잡해짐.

### 결론

KV cache 블록 locality 효과는 실재하며 다음 요인에 의존:

1. **GQA 여부** (가장 중요): GQA 없으면 블록당 메모리가 크고 locality 이득 명확
2. **모델 크기 / compute-bound 정도**: 클수록 메모리 접근 패턴 의존도 감소
3. **배치 크기**: 클수록 locality 효과 누적 (Goldilocks zone 없음)
4. **메모리 압박**: VRAM 한계에 가까우면 스케줄러 개입으로 효과 희석

## 실험 코드

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/run_locality_experiments.py \
    --model <model> \
    --prompt-counts 8 16 32 64 128 \
    --output-len 256 --num-runs 3 --warmup-runs 1 --frag-requests 150
```

관련 파일:
- `benchmarks/run_locality_experiments.py` — num_prompts sweep 실험
- `benchmarks/benchmark_kv_block_locality.py` — 단일 설정 상세 벤치마크
