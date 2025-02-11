add_executable(driver_api driver_api.cu)
target_compile_features(driver_api PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(driver_api PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(driver_api PRIVATE ${PROJECT_NAME})
