cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
option(NINFER_BUILD_TOOLS "Build NInfer developer tools" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

set(NINFER_MINIMUM_CUDA_VERSION 13.1)
if(WIN32)
  # CUDA 12.8 is the installed Blackwell toolchain on the registered RTX 5070 Ti
  # development machine. Keep the published Linux/RTX 5090 lane pinned to 13.1.
  set(NINFER_MINIMUM_CUDA_VERSION 12.8)
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS NINFER_MINIMUM_CUDA_VERSION)
  message(FATAL_ERROR
    "NInfer requires CUDA ${NINFER_MINIMUM_CUDA_VERSION} or newer on this platform; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING OR NINFER_BUILD_TOOLS)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR NINFER_BUILD_TOOLS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
set(NINFER_CUDART_TARGET CUDA::cudart)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
endif()

# A static-library-only build is the first native Windows qualification boundary: it compiles
# every CUDA/runtime target without requiring the product media decoder to be installed first.
# Product executables retain the complete media contract and therefore require FFmpeg.
set(NINFER_BUILD_MEDIA_DECODE ON)
if(WIN32 AND NOT NINFER_BUILD_APPS AND NOT NINFER_BUILD_TOOLS)
  set(NINFER_BUILD_MEDIA_DECODE OFF)
endif()

if(NINFER_BUILD_MEDIA_DECODE OR NINFER_BUILD_MEDIA_ACQUIRE)
  find_package(PkgConfig REQUIRED)
endif()
if(NINFER_BUILD_MEDIA_DECODE)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
endif()
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
if(NINFER_BUILD_SERVE)
  find_package(Threads REQUIRED)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(NINFER_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
