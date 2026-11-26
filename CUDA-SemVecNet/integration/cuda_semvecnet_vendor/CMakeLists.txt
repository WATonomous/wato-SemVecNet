cmake_minimum_required(VERSION 3.10)
project(cuda_semvecnet_vendor)

# wato-SemVecNet is cloned in perception.Dockerfile (source stage), so colcon
# does not re-clone. Same pattern as cuda_bevfusion_vendor.
set(CUDA_SEMVECNET_SOURCE_DIR
    "$ENV{AMENT_WS}/src/wato-SemVecNet"
    CACHE PATH "Path to the pre-cloned wato-SemVecNet repository")

option(FORCE_BUILD_VENDOR_PKG "Force building CUDA-SemVecNet from source" OFF)

find_package(ament_cmake REQUIRED)

macro(build_cuda_semvecnet)
  include(ExternalProject)
  set(SEMVECNET_VENDOR_PREFIX "${CMAKE_INSTALL_PREFIX}/opt/cuda_semvecnet_vendor")

  set(_build_type "${CMAKE_BUILD_TYPE}")
  if(NOT _build_type)
    set(_build_type Release)
  endif()

  # Defaults mirror cuda_bevfusion_vendor; override via environment.
  foreach(_pair
      "CUDASM=89"
      "CUDA_Inc=/usr/local/cuda/include"
      "CUDA_Lib=/usr/local/cuda/lib64"
      "TensorRT_Inc=/usr/include/x86_64-linux-gnu"
      "TensorRT_Lib=/usr/lib/x86_64-linux-gnu")
    string(REGEX MATCH "^[^=]+" _name "${_pair}")
    string(REGEX REPLACE "^[^=]+=" "" _default "${_pair}")
    if(NOT DEFINED ENV{${_name}})
      set(ENV{${_name}} "${_default}")
    endif()
  endforeach()

  externalproject_add(cuda_semvecnet_src
    SOURCE_DIR ${CUDA_SEMVECNET_SOURCE_DIR}
    SOURCE_SUBDIR CUDA-SemVecNet
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${_build_type}
      -DCMAKE_INSTALL_PREFIX=${SEMVECNET_VENDOR_PREFIX}
      -DCMAKE_CUDA_ARCHITECTURES=$ENV{CUDASM}
    # The library installs itself (lib/, include/, lib/cmake/semvecnet/).
    # DESTDIR is not used: ament installs straight into the colcon prefix.
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    CONFIGURE_HANDLED_BY_BUILD TRUE
  )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_semvecnet_vendor-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_semvecnet_vendor-config.cmake @ONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/env_hook/cuda_semvecnet_vendor_paths.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_semvecnet_vendor_paths.sh @ONLY)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuda_semvecnet_vendor-config.cmake
          DESTINATION share/${PROJECT_NAME}/cmake)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/package.xml
          DESTINATION share/${PROJECT_NAME})
endmacro()

find_package(cuda_semvecnet QUIET NO_CMAKE_PACKAGE_REGISTRY)
if(FORCE_BUILD_VENDOR_PKG OR NOT cuda_semvecnet_FOUND)
  build_cuda_semvecnet()
  ament_environment_hooks(${CMAKE_CURRENT_BINARY_DIR}/cuda_semvecnet_vendor_paths.sh)
endif()

ament_package(CONFIG_EXTRAS
  "${CMAKE_CURRENT_BINARY_DIR}/cuda_semvecnet_vendor-config.cmake")
