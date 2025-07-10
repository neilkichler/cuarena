<h1 align='center'>CuArena</h1>

Since CUDA 10.2, virtual memory is supported on CUDA devices via the [Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) (see Nvidia's [blog post](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)).

The simplest and most useful thing to built upon it is probably an arena allocator.
## Minimal Example

```c++
#include <cuarena/cuarena.hpp>

int main()
{
    cudaSetDevice(0); // runtime and driver api is supported

    using namespace cu;
    constexpr int n = 1024;
    {
        arena a; // reserves 8 GiB by default
        memblk buffer = a.allocate(n_bytes * sizeof(float)); // bumps arena pointer (may commit memory)
    
        // a memblk is a { std::byte *start, std::size_t length } struct
        // ...

        a.deallocate(buffer); // nop, unless last allocated buffer (then it is popped from the arena).

    } // automatic release of all reserved memory

    return 0;
}
```

## Installation
> Please make sure that you have installed everything mentioned in the section [Build Requirements](#build-requirements).

### System-wide
```bash
git clone https://github.com/neilkichler/cuarena.git
cd cuarena
cmake --preset release
cmake --build build
cmake --install build
```

### CMake Project


#### [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
```cmake
CPMAddPackage("gh:neilkichler/cuarena@0.1.0")
```

#### [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
```cmake
include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY git@github.com:neilkichler/cuarena.git
  GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(cuarena)
```

In either case, you can link to the library using:
```cmake
target_link_libraries(${PROJECT_NAME} PUBLIC cuarena)
```

## More Examples
Have a look at the [examples folder](https://github.com/neilkichler/cuarena/tree/main/examples).

## Documentation
The documentation is available [here](https://neilkichler.github.io/cuarena).

## Build

### Build Requirements
We require C++17, CMake v3.25.2+, Ninja (optional), and recent C++ and CUDA compilers.

#### Ubuntu
```bash
apt install cmake gcc ninja-build
```
#### Cluster
```bash
module load CMake CUDA GCC Ninja
```

### Build and run tests
#### Using Workflows
```bash
cmake --workflow --preset dev
```
#### Using Presets
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```
#### Using regular CMake
```bash
cmake -S . -B build -GNinja
cmake --build build
./build/tests/tests
```
