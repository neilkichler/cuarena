#include <cuarena/cuarena.hpp>

#include <boost/ut.hpp>

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

int main()
{
    using namespace boost::ut;

    CUcontext ctx;
    CUdevice dev;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuCtxGetDevice(&dev));

    auto n_bytes = 2_GB;
    arena a(ctx);
    memblk buffer1 = a.allocate(n_bytes);
    expect(eq(buffer1.size(), n_bytes));
    a.deallocate(buffer1);
    memblk buffer2 = a.allocate(n_bytes);
    memblk buffer3 = a.allocate(n_bytes);
    memblk buffer4 = a.allocate(n_bytes);

    arena::size_type freed_size;

    freed_size = a.pop(buffer3); // buffer 3 sits between 2 and 4, so no memory is freed
    expect(eq(freed_size, 0));

    freed_size = a.pop(buffer4); // buffer 4 sits on top, so memory is freed
    expect(eq(freed_size, buffer4.size()));

    memblk buffer5 = a.allocate(n_bytes);
    expect(eq(buffer5.data(), buffer4.data()));

    CU_CHECK(cuDevicePrimaryCtxRelease(0));

    return 0;
}
