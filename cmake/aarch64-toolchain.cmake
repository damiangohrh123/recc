# CMake toolchain file for cross-compiling this repo for the RECC board
# (aarch64, Ubuntu 20.04 / glibc 2.31).
#
# Expects aarch64-ubuntu20.04-toolchain.tar.gz to have been extracted into
# $HOME and fix_toolchain_paths.sh to have been run -- see the "Getting
# Started > 1. Build" section of the top-level README.md. That produces the
# wrapper compilers this file points at, which pass the right -B flags for
# the target binutils and the gcc-9 frontend.
#
# Usage, from recc/:
#   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
#         -DOPENCV_INCLUDE_DIR=$HOME/deps-arm64/include/opencv4 ... -DBUILD_TOOLS=ON
#   cmake --build build -j

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(_wrap "$ENV{HOME}/cross20/wrap")

set(CMAKE_C_COMPILER   "${_wrap}/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${_wrap}/aarch64-linux-gnu-g++")
set(CMAKE_AR           "${_wrap}/aarch64-linux-gnu-ar"     CACHE FILEPATH "")
set(CMAKE_RANLIB       "${_wrap}/aarch64-linux-gnu-ranlib" CACHE FILEPATH "")

# Look for headers and libraries only in the cross dependency tree, but keep
# using host programs (cmake, make) from the normal PATH.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
