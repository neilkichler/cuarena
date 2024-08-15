
add_library(cuarena "cuarena.cuh")
set_target_properties(cuarena PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuarena PROPERTIES CUDA_ARCHITECTURES 61)

target_include_directories(
    cuarena PUBLIC $<BUILD_INTERFACE:${INCLUDE_PATH}>
                   $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

target_compile_features(cuarena PUBLIC cxx_std_20 cuda_std_20)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuarena PRIVATE CUDA::cuda_driver)

