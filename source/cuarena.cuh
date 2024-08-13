#ifndef CUARENA_CUH
#define CUARENA_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

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

constexpr std::size_t operator""_KB(unsigned long long n)
{
    const auto base_two_kilobyte = n << 10;
    return base_two_kilobyte;
}

constexpr std::size_t operator""_MB(unsigned long long n)
{
    const auto base_two_megabyte = n << 20;
    return base_two_megabyte;
}

constexpr std::size_t operator""_GB(unsigned long long n)
{
    const auto base_two_gigabyte = n << 30;
    return base_two_gigabyte;
}

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

constexpr std::size_t
align_to(std::size_t p, std::size_t align) noexcept
{
    return (p + align - 1) & (~(align - 1));
}

// A move-only arena/linear/bump-pointer allocator.
struct arena
{
    using size_type = memblk::size_type;
    using byte_type = memblk::value_type;

    arena(CUcontext context);
    arena(CUcontext context, size_type capacity, size_type alignment = 16);
    arena(const arena &other)                = delete;
    arena &operator=(const arena &other)     = delete;
    arena(arena &&other) noexcept            = default;
    arena &operator=(arena &&other) noexcept = default;
    ~arena();

    // Try to allocate only multiples of alignment. So, ideally: n_bytes = k * alignment
    [[nodiscard]] memblk allocate(size_type n_bytes);

    // Calls allocate and memmove (except for last block since move is not required).
    [[nodiscard]] memblk reallocate(memblk blk, size_type n_bytes);

    // Decrements alloc position if blk is the last allocated blk. No freeing occurs.
    void deallocate(memblk blk);

    // Decrements alloc position if blk is the last allocated blk.
    size_type pop(memblk blk);

    // Check if arena owns the memory block.
    [[nodiscard]] bool owns(memblk blk);

    // Resets alloc position and clears memory.
    void clear();

    // Arena keeps growing so always returns false.
    [[nodiscard]] bool full();

    // Returns a good next size to allocate given the amount specified by n_bytes.
    // It will return the number of bytes the allocator would have to allocate
    // anyways given the alignment properties.
    //
    // Example:
    //
    //          arena a;
    //          memblk b = a.allocate(a.good_size(sizeof(int)*12));
    //
    [[nodiscard]] size_type good_size(size_type n_bytes) const;

    [[nodiscard]] constexpr size_type size_bytes() const noexcept { return _alloc_position; }

    [[nodiscard]] constexpr byte_type *data() noexcept { return _start_address; }

private:
    static constexpr auto DEFAULT_RESERVED_SIZE = 4_GB;

    byte_type *_start_address {};
    size_type _alloc_position {};
    size_type _commit_position {};
    size_type _alignment {};
    size_type _minimum_commit_size {};
    size_type _reserved_size {};
    CUmemAllocationProp _prop {};

    struct alloc_info
    {
        CUmemGenericAllocationHandle handle;
        size_type n_bytes;
    };

    std::vector<alloc_info> _alloc_info;

    static byte_type *reserve_memory(size_type capacity);

    // we assume that we can always commit from the reserved memory. Returns a handle
    size_type commit_memory(size_type n_bytes);

    bool release_memory();

    byte_type *current_position();

    // memset only for debug build
    static void debug_memset(void *address, int val, size_type size);
};

static_assert(!std::is_trivially_constructible_v<arena>);
static_assert(!std::is_trivially_destructible_v<arena>);
static_assert(!std::is_copy_constructible_v<arena>);
static_assert(!std::is_copy_assignable_v<arena>);
static_assert(std::is_move_constructible_v<arena>);
static_assert(std::is_move_assignable_v<arena>);

arena::arena(CUcontext context)
    : arena(context, DEFAULT_RESERVED_SIZE)
{ }

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

    _prop = {
        .type     = CU_MEM_ALLOCATION_TYPE_PINNED,
        .location = { .type = CU_MEM_LOCATION_TYPE_DEVICE,
                      .id   = device },
    };
    CU_CHECK(cuMemGetAllocationGranularity(&_minimum_commit_size, &_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

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
        // _alloc_info.push_back({ handle, alloc_size });
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



// A growing pool allocator with fixed size slots
struct pool
{
    using size_type = std::size_t;

    struct description
    {
        size_type slot_size;
        size_type capacity;
        size_type alignment;
    };

    pool(CUcontext context, description info)
        : _arena(context, info.capacity, info.alignment)
        , _slot_size(align_to(info.slot_size, info.alignment))
        , _head(nullptr)
    {
        assert(_slot_size >= sizeof(node) && "slot size too small");

        memblk blk = _arena.allocate(info.capacity);
        // clear();
        // push_range_onto_free_list(blk.data(), blk.size() / _slot_size);
        push_range_onto_free_list(blk);
    }

    [[nodiscard]] memblk allocate()
    {
        node *old_head = _head;
        if (old_head == nullptr) {

            // grow pool using arena
            memblk blk = _arena.allocate(_slot_size); // TODO: what if we free and allocate instead of reallocating? Have option to fail on this event

            // for (size_type i = 0; i < blk.size() / _slot_size; ++i) {
            //     push_onto_free_list(blk.data() + i * _slot_size);
            // }
            // push_range_onto_free_list(blk.data(), blk.size() / _slot_size);
            push_range_onto_free_list(blk);
        }

        _head = _head->next;

        return { reinterpret_cast<memblk::value_type *>(old_head), _slot_size };
    }

    void deallocate(memblk blk)
    {
        assert(blk.size() == _slot_size && "invalid block size");
        assert(owns(blk) && "block not owned by pool");

        push_onto_free_list(blk.data());
    }

    bool owns(memblk blk)
    {
        return _arena.owns(blk);
    }

    void clear()
    {
        // for (size_type i = 0; i < slot_count(); ++i) {
        //     push_onto_free_list(_arena.data() + i * _slot_size);
        // }
        push_range_onto_free_list({ _arena.data(), _arena.size_bytes() });
    }

    struct node
    {
        node *next;
    };

private:
    size_type slot_count()
    {
        return _arena.size_bytes() / _slot_size;
    }

    void push_range_onto_free_list(memblk blk)
    {
        size_type count         = blk.size() / _slot_size;
        memblk::value_type *ptr = blk.data();
        for (size_type i = 0; i < count; ++i) {
            push_onto_free_list(ptr + i * _slot_size);
        }
    }

    void push_onto_free_list(memblk::value_type *ptr);
    // {
    //     node *n = reinterpret_cast<node *>(ptr);
    //     // CU_CHECK(cuMemcpyDtoD((CUdeviceptr)n->next, (CUdeviceptr)_head, sizeof(CUdeviceptr)));
    //     // CU_CHECK(cuMemcpyDtoD((CUdeviceptr)&_head, (CUdeviceptr)&n, sizeof(CUdeviceptr)));
    //     // n->next = _head;
    //     // _head   = n;
    //
    //     push_onto_free_list_kernel<<<1, 1>>>(_head, n);
    // }

    arena _arena;
    size_type _slot_size;
    node *_head;
};
// struct pool;

// struct pool::node;

__global__ void push_onto_free_list_kernel(pool::node *head, pool::node *n)
{
    // if (threadIdx.x == 0 && blockIdx.x == 0)
    n->next = head;
    head    = n;
}

void pool::push_onto_free_list(memblk::value_type *ptr)
{
    node *n = reinterpret_cast<node *>(ptr);
    // CU_CHECK(cuMemcpyDtoD((CUdeviceptr)n->next, (CUdeviceptr)_head, sizeof(CUdeviceptr)));
    // CU_CHECK(cuMemcpyDtoD((CUdeviceptr)&_head, (CUdeviceptr)&n, sizeof(CUdeviceptr)));
    // n->next = _head;
    // _head   = n;

    push_onto_free_list_kernel<<<1, 1>>>(_head, n);
    CU_CHECK(cuCtxSynchronize());
}


#endif // CUARENA_CUH
