#include <metal_stdlib>
using namespace metal;

kernel void piKernel(device float *partials [[ buffer(0) ]],
                     constant uint &n [[ buffer(1) ]],
                     constant uint &totalThreads [[ buffer(2) ]],
                     uint gid [[ thread_position_in_grid ]]) {
    float localSum = 0.0f;
    float coef = 1.0f / (float)n;

    for (uint i = gid; i < n; i += totalThreads) {
        float xi = ((float)i + 0.5f) * coef;
        localSum += 4.0f / (1.0f + xi * xi);
    }

    partials[gid] = localSum;
}

