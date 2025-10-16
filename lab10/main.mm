#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

// ---- CPU реализация ----
void vectorAddCPU(const std::vector<float>& a,
                  const std::vector<float>& b,
                  std::vector<float>& c) {
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

// ---- Главная программа ----
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            std::cerr << "Usage: ./vector_add <N> <threads>\n";
            return 1;
        }

        const uint32_t N = std::stoi(argv[1]);
        const uint32_t threads = std::stoi(argv[2]);
        std::cout << "N = " << N << ", threads = " << threads << "\n";

        // --- Инициализация данных ---
        std::vector<float> a(N), b(N), c_cpu(N), c_gpu(N);
        for (uint32_t i = 0; i < N; ++i) {
            a[i] = i * 0.5f;
            b[i] = i * 2.0f;
        }

        // ---- CPU версия ----
        auto t1 = std::chrono::high_resolution_clock::now();
        vectorAddCPU(a, b, c_cpu);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "CPU done in " << cpu_ms << " ms\n";

        // ---- GPU версия ----
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not supported\n";
            return 1;
        }

        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"vector_add.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
        if (error) {
            std::cerr << "Failed to read kernel file\n";
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
        if (!library) {
            std::cerr << "Failed to create library: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }

        id<MTLFunction> func = [library newFunctionWithName:@"vectorAdd"];
        if (!func) {
            std::cerr << "Failed to load function vectorAdd\n";
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        id<MTLBuffer> aBuf = [device newBufferWithBytes:a.data()
                                                 length:N * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:b.data()
                                                 length:N * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [device newBufferWithLength:N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> nBuf = [device newBufferWithBytes:&N
                                                 length:sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:cBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake(threads, 1, 1);
        MTLSize threadgroup = MTLSizeMake(1, 1, 1);

        auto g1 = std::chrono::high_resolution_clock::now();
        [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        auto g2 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(g2 - g1).count();

        memcpy(c_gpu.data(), [cBuf contents], N * sizeof(float));

        // ---- Проверка ----
        double max_err = 0;
        for (uint32_t i = 0; i < N; ++i)
            max_err = std::fmax(max_err, fabs(c_cpu[i] - c_gpu[i]));

        std::cout << std::setprecision(6)
                  << "GPU done in " << gpu_ms << " ms\n"
                  << "Max error = " << max_err << "\n"
                  << "Speedup = " << (cpu_ms / gpu_ms) << "x\n";
    }
    return 0;
}

// clang++ -std=c++17 main.mm -framework Metal -framework Foundation -o vector_add

//  ./vector_add 1000000 1024