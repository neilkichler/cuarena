
add_executable(cuvector main_vector.cu)
set_target_properties(cuvector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuvector PROPERTIES CUDA_ARCHITECTURES 61)

target_compile_features(cuvector PUBLIC cxx_std_20)
target_compile_features(cuvector PUBLIC cuda_std_20)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuvector PRIVATE CUDA::cuda_driver)

add_executable(cuarena main.cu)
set_target_properties(cuarena PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuarena PROPERTIES CUDA_ARCHITECTURES 61)

target_compile_features(cuarena PUBLIC cxx_std_20)
target_compile_features(cuarena PUBLIC cuda_std_20)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuarena PRIVATE CUDA::cuda_driver)

