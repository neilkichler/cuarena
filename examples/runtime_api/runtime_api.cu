#include <cuarena/cuarena.hpp>

#include <cstdio>

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

__global__ void kernel(int *xs) { xs[0] = 42; }

int main()
{
    using namespace cu;
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int n = 1024;
    int h_xs[n];

    auto n_bytes = sizeof(int) * n;

    arena a;
    memblk buffer = a.allocate(n_bytes);
    int *d_xs     = new (buffer.data()) int[n];

    kernel<<<1, 1>>>(d_xs);
    CUDA_CHECK(cudaMemcpy(h_xs, d_xs, n_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("h_xs[0] = %d\n", h_xs[0]);
    return 0;
}
