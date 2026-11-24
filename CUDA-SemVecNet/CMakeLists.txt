cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # Honor ONNXRUNTIME_ROOT in find_package.
endif()

# GPU architecture. Same env var as wato-cuda-bevfusion (CUDASM=86 -> sm_86).
# 86 = RTX 30xx, 89 = RTX 40xx. Override: -DCMAKE_CUDA_ARCHITECTURES=... or CUDASM=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDASM})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDASM}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "86;89")
  endif()
endif()

project(semvecnet VERSION 0.1.0 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Writes build/compile_commands.json, which VS Code IntelliSense reads.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)   # cmake/FindTensorRT.cmake -> TensorRT::TensorRT

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>"
)

# ---------------------------------------------------------------------------
# Sources. Drop .cpp / .cu files into these directories and re-run cmake
# (CONFIGURE_DEPENDS also picks up new files on the next build).
#   src/semvecnet/  model code: engine loading, pre/post-processing, forward()
#   src/common/     shared helpers (CUDA error checks, tensors, timers...)
#   src/plugins/    optional TensorRT plugins -> libsemvecnet_plugins.so
#   src/main.cpp    optional demo executable
# Headers live in include/ (include/semvecnet/foo.hpp -> #include "semvecnet/foo.hpp").
# include/ is installed for the monorepo; src/ is also on the include path, privately,
# for internal headers you don't want to export.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SEMVECNET_CORE_SOURCES CONFIGURE_DEPENDS
  src/semvecnet/*.cpp src/semvecnet/*.cu
  src/common/*.cpp    src/common/*.cu
)
file(GLOB_RECURSE SEMVECNET_PLUGIN_SOURCES CONFIGURE_DEPENDS
  src/plugins/*.cpp src/plugins/*.cu
)

# ----------------------------- libsemvecnet_core.so ------------------------
set(SEMVECNET_HAS_ONNXRUNTIME OFF)
if(SEMVECNET_CORE_SOURCES)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
  find_package(ONNXRuntime REQUIRED MODULE)
  set(SEMVECNET_HAS_ONNXRUNTIME ON)
  add_library(semvecnet_core SHARED ${SEMVECNET_CORE_SOURCES})
  target_link_libraries(semvecnet_core
    PUBLIC ONNXRuntime::ONNXRuntime
    PRIVATE opencv_core opencv_imgproc CUDA::nvtx3  # header-only NVTX ranges for profiling
  )
  set(_scope PUBLIC)
else()
  message(WARNING "No sources under src/semvecnet or src/common yet; "
                  "semvecnet_core is an empty INTERFACE target.")
  add_library(semvecnet_core INTERFACE)
  set(_scope INTERFACE)
endif()
add_library(semvecnet::semvecnet_core ALIAS semvecnet_core)

target_include_directories(semvecnet_core ${_scope}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
if(SEMVECNET_CORE_SOURCES)
  target_include_directories(semvecnet_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
target_link_libraries(semvecnet_core ${_scope}
  CUDA::cudart
  TensorRT::TensorRT
)
set(_install_targets semvecnet_core)

# ----------------------------- libsemvecnet_plugins.so ---------------------
if(SEMVECNET_PLUGIN_SOURCES)
  add_library(semvecnet_plugins SHARED ${SEMVECNET_PLUGIN_SOURCES})
  target_include_directories(semvecnet_plugins PRIVATE include src)
  target_link_libraries(semvecnet_plugins PRIVATE CUDA::cudart TensorRT::TensorRT)
  list(APPEND _install_targets semvecnet_plugins)
endif()

# ----------------------------- semvecnet (demo executable) -----------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  find_package(OpenCV REQUIRED COMPONENTS imgcodecs imgproc)
  add_executable(semvecnet src/main.cpp)
  target_link_libraries(semvecnet PRIVATE semvecnet_core opencv_imgcodecs opencv_imgproc)
  list(APPEND _install_targets semvecnet)
endif()

# ----------------------------- test_plan_loader (test executable) ----------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_plan_loader.cpp")
  add_executable(test_plan_loader src/test_plan_loader.cpp)
  target_link_libraries(test_plan_loader PRIVATE CUDA::cudart TensorRT::TensorRT)
  # Optional: add to install targets if you want it installed
  # list(APPEND _install_targets test_plan_loader)
endif()

include(CTest)
if(BUILD_TESTING AND SEMVECNET_CORE_SOURCES)
  add_executable(test_hrnet_preprocessing tests/hrnet_preprocessing.cpp)
  target_link_libraries(test_hrnet_preprocessing PRIVATE semvecnet_core)
  add_test(NAME hrnet_preprocessing COMMAND test_hrnet_preprocessing)
  add_executable(test_polyline_postprocessing tests/polyline_postprocessing.cpp)
  target_link_libraries(test_polyline_postprocessing PRIVATE semvecnet_core)
  add_test(NAME polyline_postprocessing COMMAND test_polyline_postprocessing)
  add_executable(test_semantic_bev_mapping tests/semantic_bev_mapping.cpp)
  target_link_libraries(test_semantic_bev_mapping PRIVATE semvecnet_core)
  add_test(NAME semantic_bev_mapping COMMAND test_semantic_bev_mapping)
  add_executable(test_semantic_map_renderer tests/semantic_map_renderer.cpp)
  target_link_libraries(test_semantic_map_renderer PRIVATE semvecnet_core)
  add_test(NAME semantic_map_renderer COMMAND test_semantic_map_renderer)
  add_executable(test_lidar_semantic_projection tests/lidar_semantic_projection.cpp)
  target_link_libraries(test_lidar_semantic_projection PRIVATE semvecnet_core)
  add_test(NAME lidar_semantic_projection COMMAND test_lidar_semantic_projection)
endif()

# ----------------------------- install / export ----------------------------
# `cmake --install` produces:  lib/  bin/  include/  lib/cmake/semvecnet/
# find_package(semvecnet) then provides the imported target semvecnet::semvecnet_core.
include(CMakePackageConfigHelpers)

install(TARGETS ${_install_targets}
  EXPORT semvecnetTargets
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT semvecnetTargets
  NAMESPACE semvecnet::
  DESTINATION lib/cmake/semvecnet
)
install(FILES cmake/FindTensorRT.cmake cmake/FindONNXRuntime.cmake
  DESTINATION lib/cmake/semvecnet
)

configure_package_config_file(
  cmake/semvecnetConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/semvecnetConfig.cmake"
  INSTALL_DESTINATION lib/cmake/semvecnet
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/semvecnetConfig.cmake"
  DESTINATION lib/cmake/semvecnet
)
