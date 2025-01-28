#include <cuarena/cuarena.hpp>

#include <cstdio>

#define CU_CHECK(x)                                                                               \
    do {                                                                                          \
        CUresult err = x;                                                                         \
        if (err != CUDA_SUCCESS) {                                                                \
            const char *error_string;                                                             \
            if (cuGetErrorString(err, &error_string) == CUDA_ERROR_INVALID_VALUE) {               \
                error_string = "Unknown error";                                                   \
            }                                                                                     \
            const char *error_name;                                                               \
            if (cuGetErrorName(err, &error_name) == CUDA_ERROR_INVALID_VALUE) {                   \
                error_name = "CUDA_ERROR_UNKNOWN";                                                \
            }                                                                                     \
            fprintf(stderr, "CUDA Driver API error in %s at %s:%d: %s (%s = %d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, error_string, error_name, err);                           \
            abort();                                                                              \
        }                                                                                         \
    } while (0)

__global__ void kernel(int *xs) { xs[0] = 42; }

int main()
{
    using namespace cu;
    cudaSetDevice(0);

    constexpr int n = 1024;
    int h_xs[n];

    auto n_bytes = sizeof(int) * n;

    arena a;
    memblk buffer = a.allocate(n_bytes);
    int *d_xs     = new (buffer.data()) int[n];

    kernel<<<1, 1>>>(d_xs);
    cudaMemcpy(h_xs, d_xs, n_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("h_xs[0] = %d\n", h_xs[0]);
    return 0;
}
