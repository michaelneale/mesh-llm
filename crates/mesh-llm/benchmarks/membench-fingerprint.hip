// membench-fingerprint.hip — Memory bandwidth fingerprint for AMD GPUs (ROCm/HIP)
// Build:  hipcc -O3 -std=c++17 -o membench-fingerprint-hip membench-fingerprint.hip
//         Requires ROCm installed (typically /opt/rocm). No static-link equivalent exists.
//         Runtime dep: /opt/rocm/lib/libamdhip64.so
// Run:    ./membench-fingerprint-hip [--json]
//
// Cross-check rated bandwidth with:
//   rocm-smi --showmeminfo vram          (CLI)
//   rocm-smi -a --json                   (all metrics as JSON)
//   amdsmi_get_gpu_vram_info()            (C++ API, libamd_smi)
//
// HBM note: hipDeviceProp_t.memoryClockRate may already report the effective
// (post-DDR-doubled) rate on Instinct/CDNA cards. The DDR factor is set to ×1
// for detected HBM devices (gfx90x, gfx94x) and ×2 for GDDR (RDNA/RX series).
// Verify against rocm-smi if rated_estimated is true.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_BYTES (512 * 1024 * 1024)  // 512 MB — safely above L2/LLC on all current AMD GPUs
#define WARMUP_RUNS  3
#define TIMED_RUNS   20
#define BLOCK_SIZE   256
#define COMPUTE_ITERS 16384
#define COMPUTE_BLOCKS_PER_CU 16

// Read-only kernel. __ldg maps to scalar cache on RDNA/CDNA.
// Wavefront: 64 threads on GCN/CDNA, 32 on RDNA — gridDim accounts for warpSize.
__global__ void memread(const float4* __restrict__ src, float* sink, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float4 v = __ldg(&src[id]);
        if (v.x == 9999999.0f) atomicAdd(sink, v.x);  // prevent dead-code elimination
    }
}

__global__ void compute_fp32_kernel(float* sink, int iters) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float seed = 1.0f + 0.0001f * (float)(id + 1);
    float a0 = seed;
    float a1 = seed + 1.0f;
    float a2 = seed + 2.0f;
    float a3 = seed + 3.0f;
    const float b0 = 1.000001f;
    const float b1 = 0.999991f;
    const float c0 = 0.500001f;
    const float c1 = 0.250001f;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, b0, c0);
        a1 = fmaf(a1, b1, c1);
        a2 = fmaf(a2, b0, c1);
        a3 = fmaf(a3, b1, c0);
        a0 = fmaf(a0, b1, c1);
        a1 = fmaf(a1, b0, c0);
        a2 = fmaf(a2, b1, c0);
        a3 = fmaf(a3, b0, c1);
    }
    sink[id] = a0 + a1 + a2 + a3;
}

__global__ void compute_fp16_kernel(float* sink, int iters) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float seed = 1.0f + 0.0001f * (float)(id + 1);
    __half2 a0 = __halves2half2(__float2half(seed), __float2half(seed + 1.0f));
    __half2 a1 = __halves2half2(__float2half(seed + 2.0f), __float2half(seed + 3.0f));
    const __half2 b0 = __halves2half2(__float2half(1.000001f), __float2half(0.999991f));
    const __half2 b1 = __halves2half2(__float2half(0.999983f), __float2half(1.000013f));
    const __half2 c0 = __halves2half2(__float2half(0.500001f), __float2half(0.250001f));
    const __half2 c1 = __halves2half2(__float2half(0.125001f), __float2half(0.062501f));
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = __hfma2(a0, b0, c0);
        a1 = __hfma2(a1, b1, c1);
        a0 = __hfma2(a0, b1, c1);
        a1 = __hfma2(a1, b0, c0);
        a0 = __hfma2(a0, b0, c1);
        a1 = __hfma2(a1, b1, c0);
        a0 = __hfma2(a0, b1, c0);
        a1 = __hfma2(a1, b0, c1);
    }
    sink[id] = __low2float(a0) + __high2float(a0) + __low2float(a1) + __high2float(a1);
}

static void check(hipError_t err, const char* ctx) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", ctx, hipGetErrorString(err));
        exit(1);
    }
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static int isHBMDevice(const char* gcnArchName) {
    return (strncmp(gcnArchName, "gfx90", 5) == 0 ||
            strncmp(gcnArchName, "gfx94", 5) == 0);
}

int main(int argc, char** argv) {
    int jsonMode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) jsonMode = 1;
    }

    int deviceCount = 0;
    check(hipGetDeviceCount(&deviceCount), "hipGetDeviceCount");
    if (deviceCount == 0) {
        if (jsonMode) printf("{\"error\":\"No HIP/ROCm devices found\"}\n");
        else          printf("No HIP/ROCm devices found\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        check(hipSetDevice(dev), "hipSetDevice");

        hipDeviceProp_t props;
        check(hipGetDeviceProperties(&props, dev), "hipGetDeviceProperties");

        int hbm          = isHBMDevice(props.gcnArchName);
        double ddrFactor = hbm ? 1.0 : 2.0;
        double ratedGBps = (double)props.memoryBusWidth
                         * (double)props.memoryClockRate
                         * ddrFactor / 8.0 / 1e6;

        int elementCount = BUFFER_BYTES / sizeof(float4);
        int gridSize     = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int computeBlocks = props.multiProcessorCount > 0 ? props.multiProcessorCount * COMPUTE_BLOCKS_PER_CU : 256;
        int computeThreads = computeBlocks * BLOCK_SIZE;

        float4* dSrc;
        float*  dSink;
        float*  dComputeSink;
        check(hipMalloc(&dSrc,  BUFFER_BYTES), "hipMalloc src");
        check(hipMalloc(&dSink, sizeof(float)), "hipMalloc sink");
        check(hipMalloc(&dComputeSink, sizeof(float) * computeThreads), "hipMalloc compute sink");
        check(hipMemset(dSrc,  0, BUFFER_BYTES), "hipMemset src");
        check(hipMemset(dSink, 0, sizeof(float)), "hipMemset sink");
        check(hipMemset(dComputeSink, 0, sizeof(float) * computeThreads), "hipMemset compute sink");

        hipEvent_t evStart, evStop;
        check(hipEventCreate(&evStart), "hipEventCreate start");
        check(hipEventCreate(&evStop),  "hipEventCreate stop");

        auto dispatch = [&]() -> double {
            check(hipEventRecord(evStart, 0), "hipEventRecord start");
            memread<<<gridSize, BLOCK_SIZE>>>(dSrc, dSink, elementCount);
            check(hipGetLastError(), "memread launch");
            check(hipEventRecord(evStop, 0), "hipEventRecord stop");
            check(hipEventSynchronize(evStop), "hipEventSync");
            float ms = 0.0f;
            check(hipEventElapsedTime(&ms, evStart, evStop), "hipEventElapsed");
            return (double)BUFFER_BYTES / (ms / 1000.0) / 1e9;
        };

        auto measure_compute_fp32 = [&]() -> double {
            check(hipMemset(dComputeSink, 0, sizeof(float) * computeThreads), "hipMemset compute fp32");
            check(hipEventRecord(evStart, 0), "hipEventRecord compute fp32 start");
            compute_fp32_kernel<<<computeBlocks, BLOCK_SIZE>>>(dComputeSink, COMPUTE_ITERS);
            check(hipGetLastError(), "compute fp32 launch");
            check(hipEventRecord(evStop, 0), "hipEventRecord compute fp32 stop");
            check(hipEventSynchronize(evStop), "hipEventSync compute fp32");
            float ms = 0.0f;
            check(hipEventElapsedTime(&ms, evStart, evStop), "hipEventElapsed compute fp32");
            double totalFlops = (double)computeThreads * (double)COMPUTE_ITERS * 16.0;
            return totalFlops / (ms / 1000.0) / 1e12;
        };

        auto measure_compute_fp16 = [&]() -> double {
            check(hipMemset(dComputeSink, 0, sizeof(float) * computeThreads), "hipMemset compute fp16");
            check(hipEventRecord(evStart, 0), "hipEventRecord compute fp16 start");
            compute_fp16_kernel<<<computeBlocks, BLOCK_SIZE>>>(dComputeSink, COMPUTE_ITERS);
            check(hipGetLastError(), "compute fp16 launch");
            check(hipEventRecord(evStop, 0), "hipEventRecord compute fp16 stop");
            check(hipEventSynchronize(evStop), "hipEventSync compute fp16");
            float ms = 0.0f;
            check(hipEventElapsedTime(&ms, evStart, evStop), "hipEventElapsed compute fp16");
            double totalFlops = (double)computeThreads * (double)COMPUTE_ITERS * 32.0;
            return totalFlops / (ms / 1000.0) / 1e12;
        };

        for (int i = 0; i < WARMUP_RUNS; i++) dispatch();
        for (int i = 0; i < WARMUP_RUNS; i++) {
            (void)measure_compute_fp32();
            (void)measure_compute_fp16();
        }

        struct timespec wallStart, wallEnd;
        clock_gettime(CLOCK_MONOTONIC, &wallStart);

        double samples[TIMED_RUNS];
        double fp32Samples[TIMED_RUNS];
        double fp16Samples[TIMED_RUNS];
        for (int i = 0; i < TIMED_RUNS; i++) {
            samples[i] = dispatch();
            fp32Samples[i] = measure_compute_fp32();
            fp16Samples[i] = measure_compute_fp16();
        }

        clock_gettime(CLOCK_MONOTONIC, &wallEnd);
        double runtimeSecs = (wallEnd.tv_sec  - wallStart.tv_sec)
                           + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e9;

        qsort(samples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(fp32Samples, TIMED_RUNS, sizeof(double), cmp_double);
        qsort(fp16Samples, TIMED_RUNS, sizeof(double), cmp_double);
        double p50      = samples[TIMED_RUNS / 2];
        double p90      = samples[(int)(TIMED_RUNS * 0.90) - 1];
        double tf32P90  = fp32Samples[(int)(TIMED_RUNS * 0.90) - 1];
        double tf16P90  = fp16Samples[(int)(TIMED_RUNS * 0.90) - 1];
        double noisePct = (p90 - p50) / p90 * 100.0;
        double effPct   = p90 / ratedGBps * 100.0;

        if (jsonMode) {
            if (dev == 0) printf("[");
            printf("{\"device\":\"%s\"," 
                   "\"gcn_arch\":\"%s\","
                   "\"buffer_mb\":512,"
                   "\"runs\":%d,"
                   "\"p50_gbps\":%.2f,"
                   "\"p90_gbps\":%.2f,"
                   "\"noise_pct\":%.2f,"
                   "\"runtime_s\":%.3f,"
                   "\"rated_gbps\":%.0f,"
                   "\"rated_estimated\":%s,"
                   "\"efficiency_pct\":%.2f,"
                   "\"bus_width_bits\":%d,"
                   "\"mem_clock_mhz\":%.0f,"
                   "\"compute_tflops_fp32\":%.2f,"
                   "\"compute_tflops_fp16\":%.2f,"
                   "\"hbm\":%s}",
                   props.name, props.gcnArchName, TIMED_RUNS,
                   p50, p90, noisePct, runtimeSecs,
                   ratedGBps,
                   hbm ? "true" : "false",
                   effPct,
                   props.memoryBusWidth,
                   props.memoryClockRate / 1000.0,
                   tf32P90,
                   tf16P90,
                   hbm ? "true" : "false");
            if (dev < deviceCount - 1) printf(",");
            else printf("]\n");
        } else {
            printf("=== Memory Bandwidth Fingerprint (ROCm/HIP) ===\n");
            printf("Device  : %s  [%s]\n", props.name, props.gcnArchName);
            if (hbm) {
                printf("Memory  : HBM  (rated_estimated=true — verify DDR factor with rocm-smi)\n");
            }
            printf("Rated   : %.0f GB/s  (%d-bit @ %.0f MHz, DDR factor x%.0f)\n",
                   ratedGBps, props.memoryBusWidth,
                   props.memoryClockRate / 1000.0, ddrFactor);
            printf("Buffer  : 512 MB read-only  (%d runs)\n", TIMED_RUNS);
            printf("p50     : %.1f GB/s\n", p50);
            printf("p90     : %.1f GB/s  efficiency: %.1f%%\n", p90, effPct);
            printf("tf32    : %.2f TFLOPS\n", tf32P90);
            printf("tf16    : %.2f TFLOPS\n", tf16P90);
            printf("noise   : %.1f%%  (p90-p50 spread -- lower is better)\n", noisePct);
            printf("runtime : %.2fs\n", runtimeSecs);
            if (dev < deviceCount - 1) printf("\n");
        }

        hipFree(dSrc);
        hipFree(dSink);
        hipFree(dComputeSink);
        hipEventDestroy(evStart);
        hipEventDestroy(evStop);
    }

    return 0;
}
