# Architecture and Neural Network Optimization

## Modern x86-64 Architecture for Linear Algebra

### Processor Microarchitecture

#### Pipeline Stages

Modern Intel/AMD processors employ deep pipelines (14-20 stages) optimized for data-level parallelism:

1. **Fetch:** Instruction and data prefetching
2. **Decode:** µop generation from x86 instructions
3. **Rename:** Register aliasing for out-of-order execution
4. **Schedule:** Dependency tracking and dispatch
5. **Execute:** ALU, SIMD, memory operations on parallel ports
6. **Memory:** Cache hierarchy traversal
7. **Write-back:** Results committed to register file

**Key Metrics for NN Workloads:**
- **Superscalar width:** 4-6 instructions/cycle
- **Execution ports:** 8-10 parallel functional units
- **Out-of-Order Buffer:** 224-256 entries (reorder buffer)
- **Vector registers:** 256-512 bits (AVX2/AVX-512)

### Cache Architecture for Dense Linear Algebra

#### Cache Hierarchy

**Typical Configuration (Intel Xeon Platinum):**

| Level | Size | Latency | Bandwidth | Shared |
|-------|------|---------|-----------|--------|
| L1 Data | 48 KB | 4 cycles | ~1 TB/s | Per core |
| L1 Instruction | 32 KB | 4 cycles | N/A | Per core |
| L2 Unified | 512 KB | 12 cycles | ~400 GB/s | Per core |
| L3 Unified | 30-40 MB | 40-75 cycles | ~200 GB/s | Shared |
| DRAM | 16-512 GB | 100-300 cycles | ~50-100 GB/s | All cores |

**Cache Line:** 64 bytes = 16 × float32 values (significant for padding)

#### Cache Behavior for Matrix Multiplication

**Problem:** Dense matrix operations with poor data reuse

Example: Multiplying A(m×k) × B(k×n) = C(m×n)

```
Naive O(mkn) algorithm:
  for i = 0 to m-1:
    for j = 0 to n-1:
      c_ij = 0
      for p = 0 to k-1:
        c_ij += a[i,p] * b[p,j]  // Non-contiguous B access
```

**Cache misses:** Each element of B accessed n times (poor reuse)

**Solution - Blocking Strategy:**

Decompose into smaller sub-matrices that fit in L3 cache:

```
Tile A into blocks of size br × bc
Tile B into blocks of size bc × tn
Tile C into blocks of size br × tn

Each block multiplication:
  - Loads A block (br × bc) once from memory
  - Loads B block (bc × tn) repeatedly  
  - Accumulates into C block (br × tn)

Hit rate improvement: ~92% → 98%
```

**Block sizes for MNIST dense layer (128×64):**
- Cache block: 32×32 floats = 4KB (fits in L1)
- Register block: 4×4 floats = 64 bytes (unroll depth)

### Branch Prediction Impact on Neural Networks

#### Branch Predictor Architecture

Neural networks have two branch-heavy patterns:

1. **Loop branches (predictable):**
   - Backward branches in loop bodies
   - Prediction accuracy: 99%+ (trained after 1-2 iterations)
   - Misprediction cost: Absorbed by loop unrolling

2. **Conditional activation functions (less predictable):**
   - ReLU: `if (x > 0) y = x; else y = 0`
   - Problem: Depends on random activation values
   - Misprediction rate: ~30-40% (random inputs)
   - Cost: 15-20 cycles per misprediction

#### Misprediction Penalty Analysis

```
Cycles lost = Pipeline depth - Resolution stage
            ≈ 15-20 cycles on modern CPUs

For ReLU on 1M elements with 35% misprediction:
  Penalty = 1M × 0.35 × 15 = 5.25M cycles
  At 3 GHz = 1.75 ms overhead
```

**Mitigation techniques:**
1. **Branchless ReLU:** Use bit manipulation instead of conditional
   ```c
   // Branching version (~35% mispredictions)
   y = x > 0 ? x : 0;
   
   // Branchless version (0 mispredictions)
   int mask = (x >> 31);  // -1 if negative, 0 if positive
   y = x & ~mask;
   ```

2. **Vectorized activation:** Process 8-16 values with AVX2 (64-byte granularity)

3. **Data normalization:** Keep activation ranges predictable

### SIMD Vectorization for Matrix Operations

#### AVX2 Capabilities (256-bit vectors)

**Data Types:**
- 8 × float32 (common for neural networks)
- 4 × float64
- 32 × int8 (quantized inference)

**Key Instructions for NN:**

| Instruction | Throughput | Latency | Use Case |
|-------------|-----------|---------|----------|
| VMULPS (multiply) | 0.5 cyc | 5 cyc | Element-wise multiply |
| VADDPS (add) | 0.5 cyc | 3 cyc | Accumulation |
| VFMADD231PS (FMA) | 0.5 cyc | 5 cyc | MAC operation |
| VMAXPS (max) | 1 cyc | 1 cyc | ReLU activation |
| VBROADCASTSS (broadcast) | 1 cyc | 3 cyc | Bias addition |

**FMA (Fused Multiply-Add):**

$$c_i += a_i \times b_i$$

Throughput: 8 elements per 0.5 cycles = **16 elements/cycle**

#### Matrix Multiplication Vectorization Example

**SIMD outer product accumulation:**

```c
// Vectorized: Process 8 elements of C per iteration
for (int i = 0; i < m; i += 1) {
  for (int j = 0; j < n; j += 8) {    // 8-wide
    __m256 c_vec = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
      __m256 b_vec = _mm256_loadu_ps(&B[k*n + j]);
      c_vec = _mm256_fmadd_ps(
        _mm256_set1_ps(A[i*K + k]),  // Broadcast a_ik
        b_vec,                        // Load b_k[j:j+8]
        c_vec                         // Accumulate
      );
    }
    _mm256_storeu_ps(&C[i*n + j], c_vec);
  }
}
```

**Performance:**
- Naive: 1 FMA/cycle = 1 FLOP/cycle per core
- SIMD: 8 FMAs/cycle = 8 FLOPs/cycle per core
- **Theoretical: 8× speedup**, achieved: ~6-7× (due to memory bandwidth)

### Memory Bandwidth and Neural Network Throughput

#### Bandwidth Limitation

For matrix multiplication C = A×B (m×K, K×n):

$$\text{Arithmetic intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}} = \frac{2mn K}{(mK + Kn + mn) × 4}$$

For large matrices: ~0.5 FLOPs/byte

**Bandwidth constraint:**
- L3 bandwidth: ~200 GB/s
- Max sustainable throughput: 200 × 0.5 = **100 GFLOPs** (single core)
- Actual achieved: ~50-70 GFLOPs (cache misses, stalls)

#### Optimization: Roofline Model

```
Peak performance = min(compute throughput, memory bandwidth)

For MNIST dense layer (784→128):
  - FLOPs: 2 × 32 × 784 × 128 = 6.4M
  - Memory: 4 × (32×784 + 784×128 + 32×128) ≈ 413 KB
  - Arithmetic intensity: 6.4M / 413K ≈ 15.5 FLOPs/byte
  - At 200 GB/s: 200 × 15.5 / 4 = 775 GFLOPs (not bandwidth-limited!)
  - Achievable: ~70% of peak = 540 GFLOPs
```

### Recommendations for Neural Network Implementation

1. **Use block matrix multiply:** Increase L3 hit rate from 70% → 95%
2. **Implement branchless ReLU:** Eliminate branch mispredictions in activation
3. **Vectorize with AVX2/AVX-512:** Achieve 6-8× speedup with intrinsics
4. **Prefetch aggressively:** Hide DRAM latency for next layer
5. **Align data structures:** 64-byte alignment for cache line efficiency
6. **Minimize TLB misses:** Use large memory pages for weight matrices
7. **Profile with perf:** Measure actual L1/L2/L3 miss rates and stall cycles

### Intel VTune Profiling

Key metrics to monitor:

```bash
# Overall metrics
perf stat ./neural_network_app

# Cache behavior
perf record -e LLC-loads,LLC-load-misses,cache-misses ./app
perf report

# Micro-architectural stalls  
perf record -e cycles,stalled-cycles-frontend,stalled-cycles-backend ./app
```

Target metrics for efficient NN workload:
- **IPC (Instructions per cycle):** >2.5
- **L1 hit rate:** >90%
- **L3 hit rate:** >95%
- **Branch misprediction:** <2%
- **Stalled cycles:** <30% of total

## Conclusion

Modern x86-64 processors can sustain ~2-4 GFLOPs for dense matrix operations through:
- Cache-oblivious blocking for locality
- Vectorization with AVX2
- Branchless conditional operations
- Pipeline-aware unrolling

Well-optimized neural network implementations should achieve 70-85% of theoretical peak performance on contemporary hardware.
