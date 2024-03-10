
add_executable(cuarena main.cu)
# add_executable(cuarena vector_main.cu cuvector.cuh cuvector.cu)

set_target_properties(cuarena PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(cuarena PROPERTIES CUDA_ARCHITECTURES 61)

# C++17 is required for all users of the library
target_compile_features(cuarena PUBLIC cxx_std_17)
target_compile_features(cuarena PUBLIC cuda_std_17)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuarena PRIVATE CUDA::cuda_driver)


