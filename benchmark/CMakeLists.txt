cmake_minimum_required(VERSION 3.14)

project(BENCHMARK)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARNINGS_FLAGS "-Wall -Wpedantic -Wextra")
# -O2 -march=native: benchmarks report timings, so they must be built the way the library is used.
set(CMAKE_CXX_FLAGS "-g -O2 -march=native -std=c++20 ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../ -I${CMAKE_CURRENT_SOURCE_DIR}/../fdaPDE/core/")

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(../src/)
include_directories(../../fdaPDE/)

# no gtest here: a benchmark reports measurements and checks its own claims, so it needs no test framework
add_executable(fdapde_benchmark main.cpp)
target_link_libraries (fdapde_benchmark Eigen3::Eigen)
