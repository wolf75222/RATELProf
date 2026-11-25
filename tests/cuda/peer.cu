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

int main()
{
    int count = 0, peer = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < 2)
        return EXIT_FAILURE;
    CUDA(cudaDeviceCanAccessPeer(&peer, 1, 0));
    if (!peer)
        return EXIT_FAILURE;

    int actual = 0, expected = 47;
    int *source = nullptr, *destination = nullptr;
    CUDA(cudaSetDevice(0));
    CUDA(cudaMalloc(&source, sizeof(int)));
    CUDA(cudaMemcpy(source, &expected, sizeof(int), cudaMemcpyHostToDevice));

    CUDA(cudaSetDevice(1));
    CUDA(cudaDeviceEnablePeerAccess(0, 0));
    CUDA(cudaMalloc(&destination, sizeof(int)));
    CUDA(cudaMemcpyPeerAsync(destination, 1, source, 0, sizeof(int)));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(&actual, destination, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(destination));

    CUDA(cudaSetDevice(0));
    CUDA(cudaFree(source));
    return actual != expected;
}
