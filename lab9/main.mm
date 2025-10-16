#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

double CalcPiCPU(int n) {
    double pi = 0.0;
    double coef = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) {
        double xi = (i + 0.5) * coef;
        pi += 4.0 / (1.0 + xi * xi);
    }
    return pi * coef;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        const int n = 10000000;  // количество итераций интеграла
        std::vector<int> threadOptions = {1, 5, 10, 15, 30, 60};

        // ---- CPU baseline ----
        auto t1 = std::chrono::high_resolution_clock::now();
        double pi_cpu = CalcPiCPU(n);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::cout << "CPU pi = " << std::setprecision(12) << pi_cpu
                  << ", time = " << cpu_ms << " ms\n\n";

        // ---- GPU setup ----
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal not supported\n";
            return 1;
        }

        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"kernel.metal"
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
        if (!src) {
            std::cerr << "Failed to read kernel.metal\n";
            return 1;
        }

        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
        id<MTLFunction> func = [lib newFunctionWithName:@"piKernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        std::cout << std::setw(10) << "Threads"
                  << std::setw(15) << "GPU time (ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(20) << "Error (|Δpi|)" << "\n";
        std::cout << std::string(60, '-') << "\n";

        for (int threads : threadOptions) {
            NSUInteger totalThreads = threads;

            id<MTLBuffer> partialsBuf = [device newBufferWithLength:totalThreads * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
            uint n_u = (uint)n;
            id<MTLBuffer> nBuf = [device newBufferWithBytes:&n_u length:sizeof(uint)
                                                    options:MTLResourceStorageModeShared];
            uint total_u = (uint)totalThreads;
            id<MTLBuffer> totalBuf = [device newBufferWithBytes:&total_u length:sizeof(uint)
                                                        options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:partialsBuf offset:0 atIndex:0];
            [enc setBuffer:nBuf offset:0 atIndex:1];
            [enc setBuffer:totalBuf offset:0 atIndex:2];

            MTLSize grid = MTLSizeMake(totalThreads, 1, 1);
            MTLSize group = MTLSizeMake(1, 1, 1);

            auto g1 = std::chrono::high_resolution_clock::now();
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            auto g2 = std::chrono::high_resolution_clock::now();
            double gpu_ms = std::chrono::duration<double, std::milli>(g2 - g1).count();

            float *partials = (float *)[partialsBuf contents];
            double sum = 0.0;
            for (NSUInteger i = 0; i < totalThreads; ++i)
                sum += (double)partials[i];

            double pi_gpu = (sum / (double)n);
            double err = fabs(pi_gpu - pi_cpu);

            std::cout << std::setw(10) << threads
                      << std::setw(15) << std::fixed << std::setprecision(3) << gpu_ms
                      << std::setw(15) << std::setprecision(2) << (cpu_ms / gpu_ms)
                      << std::setw(20) << std::scientific << std::setprecision(3) << err
                      << "\n";
        }

        return 0;
    }
}
