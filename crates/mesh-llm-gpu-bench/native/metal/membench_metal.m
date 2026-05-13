#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/sysctl.h>

typedef struct {
    const char *key;
    const char *variant;
    double gbps;
} ChipBandwidth;

static const ChipBandwidth RATED_BANDWIDTH[] = {
    {"M5", "all", 153},       {"M5 Pro", "all", 307},
    {"M5 Max", "18-core CPU / 32-core GPU", 460},
    {"M5 Max", "18-core CPU / 40-core GPU", 614},
    {"M4", "all", 120},       {"M4 Pro", "all", 273},
    {"M4 Max", "14-core CPU / 32-core GPU", 410},
    {"M4 Max", "16-core CPU / 40-core GPU", 546},
    {"M3", "all", 100},       {"M3 Pro", "all", 150},
    {"M3 Max", "14-core CPU / 30-core GPU", 300},
    {"M3 Max", "16-core CPU / 40-core GPU", 400},
    {"M3 Ultra", "all", 819}, {"M2", "all", 100},
    {"M2 Pro", "all", 200},   {"M2 Max", "all", 400},
    {"M2 Ultra", "all", 800}, {"M1", "all", 68},
    {"M1 Pro", "all", 200},   {"M1 Max", "all", 400},
    {"M1 Ultra", "all", 800},
};

static int physical_cpu_count(void) {
    int32_t count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) != 0) {
        return 0;
    }
    return (int)count;
}

static bool rated_for(NSString *device_name, double *gbps_out, bool *estimated_out) {
    NSUInteger best_len = 0;
    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if ([device_name containsString:key] && [key length] > best_len) {
            best_len = [key length];
        }
    }
    if (best_len == 0) {
        return false;
    }

    bool all_variants = true;
    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if ([device_name containsString:key] && [key length] == best_len &&
            strcmp(RATED_BANDWIDTH[i].variant, "all") != 0) {
            all_variants = false;
            break;
        }
    }

    if (all_variants) {
        for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
            NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
            if ([device_name containsString:key] && [key length] == best_len) {
                *gbps_out = RATED_BANDWIDTH[i].gbps;
                *estimated_out = false;
                return true;
            }
        }
    }

    int cpu_count = physical_cpu_count();
    NSString *cpu_pattern = [NSString stringWithFormat:@"%d-core CPU", cpu_count];
    int matches = 0;
    double matched_gbps = 0.0;
    double lowest_gbps = DBL_MAX;

    for (size_t i = 0; i < sizeof(RATED_BANDWIDTH) / sizeof(RATED_BANDWIDTH[0]); ++i) {
        NSString *key = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].key];
        if (![device_name containsString:key] || [key length] != best_len) {
            continue;
        }

        NSString *variant = [NSString stringWithUTF8String:RATED_BANDWIDTH[i].variant];
        if ([variant containsString:cpu_pattern]) {
            matches += 1;
            matched_gbps = RATED_BANDWIDTH[i].gbps;
        }
        if (RATED_BANDWIDTH[i].gbps < lowest_gbps) {
            lowest_gbps = RATED_BANDWIDTH[i].gbps;
        }
    }

    if (matches == 1) {
        *gbps_out = matched_gbps;
        *estimated_out = false;
        return true;
    }

    *gbps_out = lowest_gbps;
    *estimated_out = true;
    return true;
}

static char *copy_c_string(NSString *value) {
    const char *utf8 = [value UTF8String];
    char *copy = malloc(strlen(utf8) + 1);
    if (copy != NULL) {
        strcpy(copy, utf8);
    }
    return copy;
}

static double percentile_value(NSMutableArray<NSNumber *> *values, NSUInteger index) {
    [values sortUsingSelector:@selector(compare:)];
    return [values[index] doubleValue];
}

static double run_memread(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pso,
                          id<MTLBuffer> buf,
                          id<MTLBuffer> sink,
                          MTLSize grid,
                          MTLSize tpg) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:sink offset:0 atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return elapsed;
}

static double run_compute(id<MTLCommandQueue> queue,
                          id<MTLComputePipelineState> pso,
                          id<MTLBuffer> compute_sink,
                          uint32_t compute_thread_count,
                          uint32_t compute_iters,
                          MTLSize grid,
                          MTLSize tpg,
                          double flops_per_thread_per_iter) {
    memset([compute_sink contents], 0, compute_thread_count * sizeof(float));
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    __block double elapsed = 0.0;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> b) {
      elapsed = [b GPUEndTime] - [b GPUStartTime];
    }];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:compute_sink offset:0 atIndex:0];
    [enc setBytes:&compute_iters length:sizeof(compute_iters) atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    double total_flops = (double)compute_thread_count * (double)compute_iters * flops_per_thread_per_iter;
    return total_flops / elapsed / 1e12;
}

char *mesh_llm_gpu_bench_metal_json(char **error_out) {
    @autoreleasepool {
        if (error_out != NULL) {
            *error_out = NULL;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"Metal device not available");
            }
            return NULL;
        }

        NSString *shader_source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void memread(device const float4* src [[buffer(0)]], device float* sink [[buffer(1)]], uint id [[thread_position_in_grid]]) { float4 v = src[id]; if (v.x == 9999999.0f) sink[0] = v.x; }\n"
             "kernel void compute_fp32(device float *sink [[buffer(0)]], constant uint &iters [[buffer(1)]], uint id [[thread_position_in_grid]]) { float a0 = 1.0f + 0.0001f * float(id + 1); float a1 = a0 + 1.0f; float a2 = a0 + 2.0f; float a3 = a0 + 3.0f; constexpr float b0 = 1.000001f; constexpr float b1 = 0.999991f; constexpr float c0 = 0.500001f; constexpr float c1 = 0.250001f; for (uint i = 0; i < iters; ++i) { a0 = fma(a0, b0, c0); a1 = fma(a1, b1, c1); a2 = fma(a2, b0, c1); a3 = fma(a3, b1, c0); a0 = fma(a0, b1, c1); a1 = fma(a1, b0, c0); a2 = fma(a2, b1, c0); a3 = fma(a3, b0, c1); } sink[id] = a0 + a1 + a2 + a3; }\n"
             "kernel void compute_fp16(device float *sink [[buffer(0)]], constant uint &iters [[buffer(1)]], uint id [[thread_position_in_grid]]) { half seed = half(1.0f + 0.0001f * float(id + 1)); half2 a0 = half2(seed, seed + half(1.0)); half2 a1 = half2(seed + half(2.0), seed + half(3.0)); half2 a2 = half2(seed + half(4.0), seed + half(5.0)); half2 a3 = half2(seed + half(6.0), seed + half(7.0)); constexpr half2 b0 = half2(half(1.0009765625), half(0.9990234375)); constexpr half2 b1 = half2(half(0.99951171875), half(1.00048828125)); constexpr half2 c0 = half2(half(0.1875), half(0.3125)); constexpr half2 c1 = half2(half(0.4375), half(0.5625)); for (uint i = 0; i < iters; ++i) { a0 = fma(a0, b0, c0); a1 = fma(a1, b1, c1); a2 = fma(a2, b0, c1); a3 = fma(a3, b1, c0); a0 = fma(a0, b1, c1); a1 = fma(a1, b0, c0); a2 = fma(a2, b1, c0); a3 = fma(a3, b0, c1); } float2 s0 = float2(a0); float2 s1 = float2(a1); float2 s2 = float2(a2); float2 s3 = float2(a3); sink[id] = s0.x + s0.y + s1.x + s1.y + s2.x + s2.y + s3.x + s3.y; }\n";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader_source options:nil error:&error];
        if (library == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string([NSString stringWithFormat:@"failed to compile Metal benchmark library: %@", error]);
            }
            return NULL;
        }

        id<MTLFunction> memread = [library newFunctionWithName:@"memread"];
        id<MTLFunction> compute_fp32 = [library newFunctionWithName:@"compute_fp32"];
        id<MTLFunction> compute_fp16 = [library newFunctionWithName:@"compute_fp16"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:memread error:&error];
        id<MTLComputePipelineState> pso_fp32 = [device newComputePipelineStateWithFunction:compute_fp32 error:&error];
        id<MTLComputePipelineState> pso_fp16 = [device newComputePipelineStateWithFunction:compute_fp16 error:&error];
        if (pso == nil || pso_fp32 == nil || pso_fp16 == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string([NSString stringWithFormat:@"failed to create Metal benchmark pipeline: %@", error]);
            }
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> sink = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        const NSUInteger buffer_bytes = 512 * 1024 * 1024;
        const NSUInteger float4_bytes = sizeof(float) * 4;
        const NSUInteger element_count = buffer_bytes / float4_bytes;
        id<MTLBuffer> buf = [device newBufferWithLength:buffer_bytes options:MTLResourceStorageModeShared];
        if (queue == nil || sink == nil || buf == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal benchmark resources");
            }
            return NULL;
        }

        float *ptr = (float *)[buf contents];
        NSUInteger float_count = buffer_bytes / sizeof(float);
        NSUInteger step = MAX((NSUInteger)1, float_count / 1024);
        for (NSUInteger i = 0; i < float_count; i += step) {
            ptr[i] = (float)(i % 256);
        }

        MTLSize tpg = MTLSizeMake([pso maxTotalThreadsPerThreadgroup], 1, 1);
        MTLSize grid = MTLSizeMake(element_count, 1, 1);
        const uint32_t compute_iters = 16384;
        const uint32_t compute_thread_count = 262144;
        id<MTLBuffer> compute_sink =
            [device newBufferWithLength:compute_thread_count * sizeof(float) options:MTLResourceStorageModeShared];
        if (compute_sink == nil) {
            if (error_out != NULL) {
                *error_out = copy_c_string(@"failed to allocate Metal compute benchmark resources");
            }
            return NULL;
        }

        MTLSize compute_grid = MTLSizeMake(compute_thread_count, 1, 1);
        MTLSize compute_tpg_fp32 = MTLSizeMake([pso_fp32 maxTotalThreadsPerThreadgroup], 1, 1);
        MTLSize compute_tpg_fp16 = MTLSizeMake([pso_fp16 maxTotalThreadsPerThreadgroup], 1, 1);

        for (int i = 0; i < 3; ++i) {
            (void)run_memread(queue, pso, buf, sink, grid, tpg);
        }
        for (int i = 0; i < 3; ++i) {
            (void)run_compute(queue, pso_fp32, compute_sink, compute_thread_count, compute_iters,
                              compute_grid, compute_tpg_fp32, 16.0);
            (void)run_compute(queue, pso_fp16, compute_sink, compute_thread_count, compute_iters,
                              compute_grid, compute_tpg_fp16, 32.0);
        }

        const int runs = 20;
        NSDate *start = [NSDate date];
        NSMutableArray<NSNumber *> *gbps = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *fp32_samples = [NSMutableArray arrayWithCapacity:runs];
        NSMutableArray<NSNumber *> *fp16_samples = [NSMutableArray arrayWithCapacity:runs];

        for (int i = 0; i < runs; ++i) {
            double elapsed = run_memread(queue, pso, buf, sink, grid, tpg);
            [gbps addObject:@((double)buffer_bytes / elapsed / 1e9)];
            [fp32_samples addObject:@(run_compute(queue, pso_fp32, compute_sink, compute_thread_count, compute_iters,
                                                  compute_grid, compute_tpg_fp32, 16.0))];
            [fp16_samples addObject:@(run_compute(queue, pso_fp16, compute_sink, compute_thread_count, compute_iters,
                                                  compute_grid, compute_tpg_fp16, 32.0))];
        }

        double p50 = percentile_value(gbps, runs / 2);
        double p90 = percentile_value(gbps, (NSUInteger)((double)runs * 0.90) - 1);
        double noise = (p90 - p50) / p90 * 100.0;
        double fp32_measured = percentile_value(fp32_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double fp16_measured = percentile_value(fp16_samples, (NSUInteger)((double)runs * 0.90) - 1);
        double runtime_secs = [[NSDate date] timeIntervalSinceDate:start];

        NSString *device_name = [device name];
        double rated = 0.0;
        bool estimated = false;
        bool has_rated = rated_for(device_name, &rated, &estimated);
        double efficiency = has_rated ? p90 / rated * 100.0 : 0.0;

        NSMutableString *json = [NSMutableString stringWithFormat:
            @"[{\"device\":\"%@\",\"buffer_mb\":512,\"runs\":%d,\"p50_gbps\":%.2f,\"p90_gbps\":%.2f,\"noise_pct\":%.2f,\"runtime_s\":%.3f,\"compute_tflops_fp32\":%.2f,\"compute_tflops_fp16\":%.2f",
            device_name, runs, p50, p90, noise, runtime_secs, fp32_measured, fp16_measured];
        if (has_rated) {
            [json appendFormat:@",\"rated_gbps\":%.0f,\"rated_estimated\":%@", rated, estimated ? @"true" : @"false"];
            [json appendFormat:@",\"efficiency_pct\":%.2f", efficiency];
        }
        [json appendString:@"}]"];
        return copy_c_string(json);
    }
}

void mesh_llm_gpu_bench_free(void *ptr) {
    free(ptr);
}
