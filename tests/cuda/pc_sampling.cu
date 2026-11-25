#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void sampling_work(float *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 1.0f + 0.00001f * i;
    #pragma unroll 1
    for (int k = 0; k < 32768; ++k)
        x = fmaf(x, 0.99999f, float((k & 127) + 1) * 0.000002f);
    output[i] = x;
}

int main()
{
    constexpr int blocks = 256;
    constexpr int threads = 128;
    float *output = nullptr;
    cudaError_t status = cudaMalloc(&output, blocks * threads * sizeof(float));
    if (status == cudaSuccess) {
        sampling_work<<<blocks, threads>>>(output);
        status = cudaDeviceSynchronize();
        cudaFree(output);
    }
    if (status != cudaSuccess)
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    return status != cudaSuccess;
}
