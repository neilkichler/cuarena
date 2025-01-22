
add_library(cuarena INTERFACE "cuarena.cuh")
set_target_properties(cuarena PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(
#     cuarena INTERFACE $<BUILD_INTERFACE:${INCLUDE_PATH}>
#                       $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
# )

target_compile_features(cuarena INTERFACE cxx_std_20 cuda_std_20)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuarena INTERFACE CUDA::cuda_driver)

