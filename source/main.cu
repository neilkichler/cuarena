#include <cstdio>
#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

using idx       = std::size_t;
using s8        = std::int8_t;
using s16       = std::int16_t;
using s32       = std::int32_t;
using s64       = std::int64_t;
using u8        = std::uint8_t;
using u16       = std::uint16_t;
using u32       = std::uint32_t;
using u64       = std::uint64_t;
using f32       = float;
using f64       = double;
using size_type = std::size_t;

struct memory_block
{
    using value_type = std::byte;
    using size_type  = std::size_t;

    value_type *start;
    size_type length;

    [[nodiscard]] constexpr auto begin() const noexcept { return start; }
    [[nodiscard]] constexpr auto data() const noexcept { return start; }
    [[nodiscard]] constexpr auto empty() const noexcept { return length == 0; }
    [[nodiscard]] constexpr auto end() const noexcept { return start + length; }
    [[nodiscard]] constexpr auto size() const noexcept { return length; }

    constexpr value_type &operator[](size_type i) const noexcept
    {
        assert(i < length);
        return start[i];
    }
};

using memblk = memory_block;

constexpr size_type
align_to(size_type p, size_type align) noexcept
{
    return (p + align - 1) & (~(align - 1));
}

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

struct cuda_mem_info {
    CUmemAllocationProp prop;
    std::size_t n_bytes;
};

cuda_mem_info min_to_allocate(memblk::size_type n_bytes) {
    // TODO: this could be done once in an allocator
    CUmemAllocationProp prop {
        .type     = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = { .type = CU_MEM_LOCATION_TYPE_DEVICE,
                      .id   = 0 /* current device */ },
    };
    std::size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    n_bytes = align_to(n_bytes, granularity);
    return {prop, n_bytes};
}

void *os_mem_reserve(memblk::size_type n_bytes)
{
    auto info = min_to_allocate(n_bytes);

    CUdeviceptr addr;
    cuMemAddressReserve(&addr, info.n_bytes, 0, 0, 0);
    return (void *)addr;
}

u64 os_mem_commit(void *addr, memblk::size_type n_bytes)
{
    auto info = min_to_allocate(n_bytes);

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, info.n_bytes, &info.prop, 0));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuMemMap((CUdeviceptr)addr, info.n_bytes, 0, handle, 0));

    // enable access to memory mapping
    CUmemAccessDesc access_desc {
        .location = info.prop.location,
        .flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    CU_CHECK(cuMemSetAccess((CUdeviceptr)addr, info.n_bytes, &access_desc, 1));
    return handle;
}

void os_mem_release(void *addr, memblk::size_type n_bytes, void *data)
{
    auto info = min_to_allocate(n_bytes);
    CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)data;
    CU_CHECK(cuMemUnmap((CUdeviceptr)addr, info.n_bytes));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)addr, info.n_bytes));
    CU_CHECK(cuMemRelease(handle));
}

int main(int argc, char *argv[])
{
    CUcontext ctx;
    CUdevice dev;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuCtxGetDevice(&dev));

    int supports_virtual_memory = 0;
    CU_CHECK(cuDeviceGetAttribute(&supports_virtual_memory, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
    if (supports_virtual_memory) {
        printf("Virtual memory support detected\n");
    }

    printf("Context: %p, Device: %d\n", ctx, dev);

    size_type n_bytes = 1024 * sizeof(f64);
    void *addr        = os_mem_reserve(n_bytes);
    printf("Reserved %zu bytes at %p\n", n_bytes, addr);

    u64 handle = os_mem_commit(addr, n_bytes);
    printf("Committed %zu bytes at %p\n", n_bytes, addr);

    os_mem_release(addr, n_bytes, (void *)handle);
    printf("Released %zu bytes at %p\n", n_bytes, addr);

    CU_CHECK(cuDevicePrimaryCtxRelease(0));

    return 0;
}
