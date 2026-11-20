# Finds TensorRT and defines the imported target TensorRT::TensorRT
# (nvinfer + nvinfer_plugin + nvonnxparser when present).
#
# Hints (same env vars wato-cuda-bevfusion and the wato_monorepo vendor package use):
#   TensorRT_Inc   dir containing NvInfer.h   (e.g. /usr/include/x86_64-linux-gnu)
#   TensorRT_Lib   dir containing libnvinfer  (e.g. /usr/lib/x86_64-linux-gnu)

find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS ENV TensorRT_Inc ${TensorRT_ROOT}/include
  PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)
find_library(TensorRT_nvinfer_LIBRARY nvinfer
  HINTS ENV TensorRT_Lib ${TensorRT_ROOT}/lib)
find_library(TensorRT_nvinfer_plugin_LIBRARY nvinfer_plugin
  HINTS ENV TensorRT_Lib ${TensorRT_ROOT}/lib)
find_library(TensorRT_nvonnxparser_LIBRARY nvonnxparser
  HINTS ENV TensorRT_Lib ${TensorRT_ROOT}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT REQUIRED_VARS
  TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvinfer_plugin_LIBRARY)

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
  add_library(TensorRT::TensorRT INTERFACE IMPORTED)
  set_target_properties(TensorRT::TensorRT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  target_link_libraries(TensorRT::TensorRT INTERFACE
    ${TensorRT_nvinfer_LIBRARY} ${TensorRT_nvinfer_plugin_LIBRARY})
  if(TensorRT_nvonnxparser_LIBRARY)
    target_link_libraries(TensorRT::TensorRT INTERFACE ${TensorRT_nvonnxparser_LIBRARY})
  endif()
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
  TensorRT_nvinfer_plugin_LIBRARY TensorRT_nvonnxparser_LIBRARY)
