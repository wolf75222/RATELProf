#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA(call) do {                                                        \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status));      \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
} while (false)

__global__ void increment(int *value)
{
    *value += 1;
}

int main()
{
    int actual = 0;
    int *device = nullptr;
    cudaEvent_t ready;
    cudaStream_t producer, consumer;
    CUDA(cudaMalloc(&device, sizeof(*device)));
    CUDA(cudaEventCreate(&ready));
    CUDA(cudaStreamCreate(&producer));
    CUDA(cudaStreamCreate(&consumer));

    // Reuse the same CUDA event so the report must distinguish two records.
    for (int i = 0; i != 2; ++i) {
        CUDA(cudaMemsetAsync(device, i, sizeof(*device), producer));
        CUDA(cudaEventRecord(ready, producer));
        CUDA(cudaStreamWaitEvent(consumer, ready, 0));
        increment<<<1, 1, 0, consumer>>>(device);
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(consumer));
    }

    CUDA(cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost));
    CUDA(cudaStreamDestroy(consumer));
    CUDA(cudaStreamDestroy(producer));
    CUDA(cudaEventDestroy(ready));
    CUDA(cudaFree(device));

    const int expected = 0x01010101 + 1;
    return actual != expected;
}
