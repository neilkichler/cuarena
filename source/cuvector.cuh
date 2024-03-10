#pragma once
#include <cuda.h>
#include <vector>

namespace cuda_utils {

// Typed wrapper class
template<typename T, class VectorAllocator>
class Vector {
    VectorAllocator allocator;
public:
    Vector(CUcontext ctx) : allocator(ctx) {}

    CUresult reserve(size_t num) {
        return allocator.reserve(num * sizeof(T));
    }

    CUresult grow(size_t num) {
        return allocator.grow(num * sizeof(T));
    }

    T *getPointer() const {
        return (T*)allocator.getPointer();
    }

    size_t getSize() const {
        return allocator.getSize();
    }
};

class VectorMemAlloc {
private:
    CUcontext ctx;
    CUdeviceptr d_p;
    size_t alloc_sz;
public:
    VectorMemAlloc(CUcontext context);
    ~VectorMemAlloc();

    CUdeviceptr getPointer() const {
        return d_p;
    }

    size_t getSize() const {
        return alloc_sz;
    }

    // Reserves some extra space in order to speed up grow()
    CUresult reserve(size_t new_sz);

    // Actually commits num bytes of additional memory
    CUresult grow(size_t new_sz) {
        return reserve(new_sz);
    }
};

class VectorMemAllocManaged {
private:
    CUcontext ctx;
    CUdevice dev;
    CUdeviceptr d_p;
    size_t alloc_sz;
    size_t reserve_sz;
    // Attribute query to see if cuMemPrefetchAsync is available to the target device.
    int supportsConcurrentManagedAccess;

public:
    VectorMemAllocManaged(CUcontext context);
    ~VectorMemAllocManaged();

    CUdeviceptr getPointer() const {
        return d_p;
    }

    size_t getSize() const {
        return alloc_sz;
    }

    // Reserves some extra space in order to speed up grow()
    CUresult reserve(size_t new_sz);

    // Actually commits num bytes of additional memory
    CUresult grow(size_t new_sz);
};

class VectorMemMap {
private:
    CUdeviceptr d_p;
    CUmemAllocationProp prop;
    CUmemAccessDesc accessDesc;
    struct Range {
        CUdeviceptr start;
        size_t sz;
    };
    std::vector<Range> va_ranges;
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<size_t> handle_sizes;
    size_t alloc_sz;
    size_t reserve_sz;
    size_t chunk_sz;
public:
    VectorMemMap(CUcontext context);
    ~VectorMemMap();

    CUdeviceptr getPointer() const {
        return d_p;
    }

    size_t getSize() const {
        return alloc_sz;
    }

    // Reserves some extra space in order to speed up grow()
    CUresult reserve(size_t new_sz);

    // Actually commits num bytes of additional memory
    CUresult grow(size_t new_sz);
};

}
