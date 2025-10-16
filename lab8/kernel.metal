#include <metal_stdlib>
using namespace metal;

// Глобальная константа в памяти constant
constant char message[] = "Hello world from GPU!\n";

kernel void helloKernel(device char *output [[ buffer(0) ]],
                        uint id [[ thread_position_in_grid ]]) {
    for (uint i = 0; i < sizeof(message); i++) {
        output[i] = message[i];
    }
}
