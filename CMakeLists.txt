cmake_minimum_required(VERSION 3.15)
project(vrc_core VERSION 1.1.0 LANGUAGES CXX)

# =============================================================================
# VRC Core — CMake Build System
# Platforms: Linux, Windows, macOS
# Dependencies: Python3, pybind11, LZ4
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Default to Release build ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# Find dependencies
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# LZ4 — try pkg-config first, then manual search
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LZ4 QUIET liblz4)
endif()

if(NOT LZ4_FOUND)
  find_path(LZ4_INCLUDE_DIR NAMES lz4.h
    HINTS /usr/include /usr/local/include /opt/homebrew/include
  )
  find_library(LZ4_LIBRARY NAMES lz4 liblz4
    HINTS /usr/lib /usr/local/lib /opt/homebrew/lib
  )
  if(LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    set(LZ4_FOUND TRUE)
    set(LZ4_INCLUDE_DIRS ${LZ4_INCLUDE_DIR})
    set(LZ4_LIBRARIES ${LZ4_LIBRARY})
    message(STATUS "Found LZ4: ${LZ4_LIBRARY}")
  else()
    message(FATAL_ERROR
      "LZ4 not found!\n"
      "  Linux:   sudo apt-get install liblz4-dev\n"
      "  macOS:   brew install lz4\n"
      "  Windows: vcpkg install lz4"
    )
  endif()
endif()

# =============================================================================
# Target: vrc_core (Python extension module)
# =============================================================================

pybind11_add_module(vrc_core
  vrc_core_lossless.cpp
)

target_include_directories(vrc_core PRIVATE
  ${LZ4_INCLUDE_DIRS}
)

target_link_libraries(vrc_core PRIVATE
  ${LZ4_LIBRARIES}
)

# =============================================================================
# Compiler flags
# =============================================================================

if(MSVC)
  target_compile_options(vrc_core PRIVATE
    /O2      # Maximum optimization
    /W4      # High warning level
    /WX      # Warnings as errors
  )
else()
  target_compile_options(vrc_core PRIVATE
    -O3      # Maximum optimization
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    # Optional: uncomment for OpenMP multithreading
    # -fopenmp
  )
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS vrc_core
  LIBRARY DESTINATION ${Python3_SITEARCH}
  RUNTIME DESTINATION ${Python3_SITEARCH}
)

# =============================================================================
# Configuration summary
# =============================================================================

message(STATUS "")
message(STATUS "=== VRC Core Build Configuration ===")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:       ${Python3_VERSION} at ${Python3_EXECUTABLE}")
message(STATUS "  pybind11:     ${pybind11_VERSION}")
message(STATUS "  LZ4:          ${LZ4_LIBRARIES}")
message(STATUS "=====================================")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "  cmake --build . --config Release")
message(STATUS "")
