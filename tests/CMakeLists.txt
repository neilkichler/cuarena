include(FetchContent)

set(BOOST_UT_DISABLE_MODULE ON)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.1.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests 
    tests.cpp
)

# Use C++17 in the tests
target_compile_features(tests PRIVATE cxx_std_17 cuda_std_17)

# Specify the CUDA architecture to compile to
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES native)

# Enable separate compilation
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with main library and testing library
target_link_libraries(tests PRIVATE cuarena ut)

# Find and link with CUDA headers
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tests PRIVATE CUDA::cuda_driver)

# Register tests for use in ctest
add_test(NAME testcumccormick COMMAND tests)

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
if(ENABLE_TEST_COVERAGE)
    target_compile_options(tests PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(tests PUBLIC -fprofile-arcs -ftest-coverage)
endif()
