#ifndef CUARENA_CUARENA_H
#define CUARENA_CUARENA_H

#include "defer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

#define CHECKPOINT_1(x, y) x##y
#define CHECKPOINT_2(x, y) CHECKPOINT_1(x, y)
#define CHECKPOINT_3(x)    CHECKPOINT_2(x, __COUNTER__)
#define checkpoint(arena)  auto CHECKPOINT_3(_cuarena_) = (arena).checkpoint_();

namespace cu
{

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

// A move-only arena/linear/bump-pointer allocator.
struct arena
{
    using size_type  = memblk::size_type;
    using byte_type  = memblk::value_type;
    using properties = CUmemAllocationProp;
    using device_ctx = CUcontext;

    arena(); // uses the current context of the calling thread
    arena(device_ctx context);
    arena(device_ctx context, size_type capacity, size_type alignment = 16);
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

    // Points at the start of the contiguous virtual memory.
    [[nodiscard]] constexpr byte_type *data() noexcept { return _start_address; }

    [[nodiscard]] auto checkpoint_()
    {
        assert(!checkpoint_set && "cuarena: checkpoint was already set");
        set_checkpoint(true);
        return finally([&, a = _alloc_position] {
            set_checkpoint(false);
            _alloc_position = a;
        });
    }

    [[nodiscard]] auto allocated_size() const noexcept { return _alloc_position; }

private:
    static constexpr auto DEFAULT_RESERVED_SIZE = 8_GB;

    byte_type *_start_address {};
    size_type _alloc_position {};
    size_type _commit_position {};
    size_type _alignment {};
    size_type _minimum_commit_size {};
    size_type _reserved_size {};
    properties _prop {};

#ifndef NDEBUG
    bool checkpoint_set = false;
#endif

    void set_checkpoint([[maybe_unused]] bool val)
    {
#ifndef NDEBUG
        checkpoint_set = val;
#endif
    }

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

    // only used for initializing the arena context if no context was passed into the constructor
    device_ctx thread_current_context() const;

    // memset only for debug build
    static void debug_memset(void *address, int val, size_type size);
};

template<typename T>
[[nodiscard]] T *make_array(arena &a, std::size_t n) noexcept
{
    // TODO: in C++23 we might want to use: start_lifetime_as
    return new (a.allocate(n * sizeof(T)).data()) T[n];
}

} // namespace cu
#endif // CUARENA_CUARENA_H
