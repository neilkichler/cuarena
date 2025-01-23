add_executable(runtime runtime.cu)
target_compile_features(runtime PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(runtime PRIVATE ${PROJECT_NAME})
