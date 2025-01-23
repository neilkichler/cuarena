add_executable(runtime_api runtime_api.cu)
target_compile_features(runtime_api PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(runtime_api PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(runtime_api PRIVATE ${PROJECT_NAME})
