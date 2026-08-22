# The native data plane: fused pre-processing kernels and device NMS, exposed as
# `shipinfer._C` through pybind11.
#
# Deliberately small. Everything torch already does well — allocation, streams, graph
# capture, resize on its own — stays in torch (ADR-003). What lives here is the one thing
# it cannot express: four memory-bound passes fused into one.
#
# Build:
#   cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build native/build -j
# or simply:
#   python scripts/build_native.py

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(shipinfer_native LANGUAGES CXX VERSION 0.1.0)

option(SHIPINFER_WITH_CUDA "Build the CUDA kernels" ON)
option(SHIPINFER_WITH_HIP  "Build the HIP/ROCm kernels instead of CUDA" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  # A debug build of a preprocessing kernel is ~10x slower and would make every benchmark
  # meaningless, so Release is the default rather than something to remember.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(SHIPINFER_WITH_HIP)
  set(SHIPINFER_WITH_CUDA OFF)
endif()

# ---------------------------------------------------------------------------------------
# Python + pybind11
# ---------------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Ask the active interpreter where pybind11 lives rather than requiring a system package:
# the extension must build against the same interpreter that will import it.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_LOOKUP)
if(NOT PYBIND11_LOOKUP EQUAL 0)
  message(FATAL_ERROR "pybind11 is not installed for ${Python3_EXECUTABLE}. "
                      "Run: ${Python3_EXECUTABLE} -m pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------------------
set(SHIPINFER_SOURCES bindings/module.cpp)

if(SHIPINFER_WITH_HIP)
  enable_language(HIP)
  add_compile_definitions(SHIPINFER_WITH_HIP)
  set_source_files_properties(src/image_ops.cu PROPERTIES LANGUAGE HIP)
  list(APPEND SHIPINFER_SOURCES src/image_ops.cu)
  message(STATUS "shipinfer native: HIP/ROCm backend")

elseif(SHIPINFER_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Pascal through Ada. `native` would be faster to build but produces a binary that
    # silently fails on any other node in the fleet.
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89)
  endif()
  list(APPEND SHIPINFER_SOURCES src/image_ops.cu)
  message(STATUS "shipinfer native: CUDA ${CUDAToolkit_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")

else()
  message(FATAL_ERROR "Enable exactly one of SHIPINFER_WITH_CUDA / SHIPINFER_WITH_HIP")
endif()

# ---------------------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------------------
pybind11_add_module(_C MODULE ${SHIPINFER_SOURCES})

target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

if(SHIPINFER_WITH_CUDA)
  target_link_libraries(_C PRIVATE CUDA::cudart)
  # Separable compilation is deliberately OFF. There are no cross-translation-unit device
  # calls here, so relocatable device code would buy nothing — and combined with the
  # link-time optimisation pybind11 enables, the device-link step's fatbin symbol gets
  # stripped and the module fails to import with "undefined symbol: fatbinData".
  set_target_properties(_C PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Install straight into the package tree so an editable install picks it up with no
# PYTHONPATH games: `import shipinfer._C` must work immediately after a build.
set(SHIPINFER_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/shipinfer")
add_custom_command(TARGET _C POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C> "${SHIPINFER_PACKAGE_DIR}/"
  COMMENT "Installing $<TARGET_FILE_NAME:_C> into src/shipinfer/")
