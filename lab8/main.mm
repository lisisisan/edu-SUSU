#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "No GPU found!\n";
            return -1;
        }

        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:@"kernel.metal"
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (error) {
            std::cerr << "Failed to read kernel.metal\n";
            return -1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "Failed to create library: " << [[error localizedDescription] UTF8String] << "\n";
            return -1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"helloKernel"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return -1;
        }

        const int bufferSize = 256;
        id<MTLBuffer> buffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize threadgroup = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        char* output = (char*)[buffer contents];
        std::cout << output << std::endl;
    }
    return 0;
}


// clang++ -std=c++17 main.mm -framework Metal -framework Foundation -o main
// ./main
