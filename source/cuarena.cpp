#include <cuarena/cuarena.hpp>

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

namespace cu
{

constexpr std::size_t
align_to(std::size_t p, std::size_t align) noexcept
{
    return (p + align - 1) & (~(align - 1));
}

arena::device_ctx arena::thread_current_context() const
{
    CUcontext ctx;
    CU_CHECK(cuCtxGetCurrent(&ctx));
    return ctx;
}

arena::arena()
    : arena(thread_current_context(), DEFAULT_RESERVED_SIZE)
{
}

arena::arena(CUcontext context)
    : arena(context, DEFAULT_RESERVED_SIZE)
{
}

arena::arena(CUcontext context, size_type capacity, size_type alignment)
    : _alignment(alignment)
    , _reserved_size(capacity)
{
    CUdevice device;
    CUcontext prev_ctx;

    CU_CHECK(cuCtxGetCurrent(&prev_ctx));
    if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
        CU_CHECK(cuCtxGetDevice(&device));
        CU_CHECK(cuCtxSetCurrent(prev_ctx));
    }

    int supports_virtual_memory = 0;
    CU_CHECK(cuDeviceGetAttribute(&supports_virtual_memory, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
    assert(supports_virtual_memory && "Cuda device does not support virtual memory");

    _prop = {
        .type     = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = { .type = CU_MEM_LOCATION_TYPE_DEVICE,
                      .id   = device },
    };
    CU_CHECK(cuMemGetAllocationGranularity(&_minimum_commit_size, &_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    _reserved_size = align_to(capacity, _minimum_commit_size);
    CUdeviceptr addr;
    CU_CHECK(cuMemAddressReserve(&addr, _reserved_size, _alignment, 0, 0));
    _start_address = (byte_type *)addr;
}

arena::~arena()
{
    release_memory();
};

memblk arena::allocate(size_type n_bytes)
{
    assert(n_bytes > 0);

    size_type alloc_size = align_to(n_bytes, _alignment);

    // out of memory? -> commit
    if (_alloc_position + alloc_size > _commit_position) {
        size_type handle = commit_memory(alloc_size);
    }

    byte_type *addr = &_start_address[_alloc_position];

    _alloc_position += alloc_size;
    return { addr, n_bytes };
}

memblk arena::reallocate(memblk blk, size_type n_bytes)
{
    if (blk.empty()) {
        return allocate(n_bytes);
    }

    if (pop(blk) > 0) {
        // if we pop the last allocated we can directly allocate without memmove
        return allocate(n_bytes);
    } else {
        memblk new_blk = allocate(n_bytes);
        CU_CHECK(cuMemcpyDtoD((CUdeviceptr)new_blk.data(), (CUdeviceptr)blk.data(), std::min(blk.size(), n_bytes)));
        return new_blk;
    }
}

void arena::deallocate(memblk blk)
{
    pop(blk);
}

arena::size_type arena::pop(memblk blk)
{
    size_type alloc_size = align_to(blk.size(), _alignment);
    bool can_pop         = blk.data() + alloc_size == current_position();
    size_type freed_size = size_type(can_pop) * alloc_size;
    _alloc_position -= freed_size;

    return freed_size;
}

bool arena::owns(memblk blk)
{
    return blk.data() >= _start_address && blk.data() < current_position();
}

void arena::clear()
{
    _alloc_position = 0;
}

bool arena::release_memory()
{
    if (_commit_position == 0) {
        return true;
    }

    clear();
    CU_CHECK(cuMemUnmap((CUdeviceptr)_start_address, _commit_position));

    for (auto [handle, _] : _alloc_info) {
        CU_CHECK(cuMemRelease(handle));
    }

    return true;
}

arena::byte_type *arena::current_position() { return &_start_address[_alloc_position]; }

arena::size_type arena::commit_memory(size_type n_bytes)
{
    size_type commit_size = align_to(n_bytes, _minimum_commit_size);

    CUmemAccessDesc access_desc {
        .location = _prop.location,
        .flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    if (_commit_position + commit_size > _reserved_size) {
        // reserve an additional _reserved_size memory range -> doubling of _reserved_size

        // try direct reallocation
        CUdeviceptr new_addr;
        CUdeviceptr end_addr = (CUdeviceptr)&_start_address[_commit_position];
        CUresult status      = cuMemAddressReserve(&new_addr, _reserved_size, 0, end_addr, 0);
        if (status != CUDA_SUCCESS or new_addr != end_addr) {
            // direct reallocation failed -> perform free and new full reserve
            cuMemAddressFree(new_addr, _reserved_size);
            CU_CHECK(cuMemAddressReserve(&new_addr, 2 * _reserved_size, 0, 0, 0));

            CU_CHECK(cuMemUnmap((CUdeviceptr)_start_address, _commit_position));

            CUdeviceptr alloc_addr = new_addr;
            // remap all the previous allocations to the new virtual address range
            for (auto [handle, alloc_size] : _alloc_info) {
                CU_CHECK(cuMemMap(alloc_addr, alloc_size, 0, handle, 0));
                CU_CHECK(cuMemSetAccess(alloc_addr, alloc_size, &access_desc, 1));
                alloc_addr += alloc_size;
            }

            // free old virtual memory range
            CU_CHECK(cuMemAddressFree((CUdeviceptr)_start_address, _commit_position));

            _start_address = (byte_type *)new_addr;
        }
        _reserved_size *= 2;
    }

    // Actually commit the new memory bytes
    void *current_commit_end = _start_address + _commit_position;

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, commit_size, &_prop, 0));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuMemMap((CUdeviceptr)current_commit_end, commit_size, 0, handle, 0));

    // enable access to memory mapping
    CU_CHECK(cuMemSetAccess((CUdeviceptr)current_commit_end, commit_size, &access_desc, 1));

    _alloc_info.push_back({ handle, commit_size });
    _commit_position += commit_size;
    return handle;
}

static_assert(!std::is_trivially_constructible_v<arena>);
static_assert(!std::is_trivially_destructible_v<arena>);
static_assert(!std::is_copy_constructible_v<arena>);
static_assert(!std::is_copy_assignable_v<arena>);
static_assert(std::is_move_constructible_v<arena>);
static_assert(std::is_move_assignable_v<arena>);

} // namespace cu
