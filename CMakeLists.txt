
cmake_minimum_required (VERSION 3.24)

project(cuarena LANGUAGES CUDA CXX)

# enable_language(CUDA)
# include(CheckLanguage)
# check_language(CUDA)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    
  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Better support of clang based tools
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    
  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

endif()

add_subdirectory(source)

