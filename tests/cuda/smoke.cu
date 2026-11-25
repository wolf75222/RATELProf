#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA(call) do {                                                        \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status));      \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
} while (false)

__global__ void increment(int *values)
{
    values[threadIdx.x] += 1;
}

int main(int argc, char **argv)
{
    if (argc != 2 || std::strcmp(argv[1], "argument with 'quotes' and spaces") != 0)
        return EXIT_FAILURE;

    cudaStream_t stream;
    cudaEvent_t completed;
    int *device = nullptr, *host = nullptr;
    CUDA(cudaStreamCreate(&stream));
    CUDA(cudaEventCreate(&completed));
    CUDA(cudaMalloc(&device, 32 * sizeof(int)));
    CUDA(cudaMallocHost(&host, 32 * sizeof(int)));

    nvtxRangePushA("Smoke NVTX range");
    CUDA(cudaMemsetAsync(device, 0, 32 * sizeof(int), stream));
    for (int i = 0; i < 10; ++i)
        increment<<<1, 32, 0, stream>>>(device);
    CUDA(cudaGetLastError());
    CUDA(cudaMemcpyAsync(host, device, 32 * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA(cudaEventRecord(completed, stream));
    CUDA(cudaEventSynchronize(completed));
    nvtxMarkA("Smoke NVTX mark");
    nvtxRangePop();

    for (int i = 0; i < 32; ++i)
        if (host[i] != 10)
            return EXIT_FAILURE;

    CUDA(cudaEventDestroy(completed));
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(device));
    CUDA(cudaFreeHost(host));
    return EXIT_SUCCESS;
}
