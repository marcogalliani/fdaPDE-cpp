cmake_minimum_required(VERSION 3.14)

project(fdaPDE-cpp-testing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARNINGS_FLAGS "") # "-Wall -Wpedantic -Wextra -Werror")
set(CMAKE_CXX_FLAGS "-O2 -march=native -std=c++20 ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../ -I${CMAKE_CURRENT_SOURCE_DIR}/../fdaPDE/core/")

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# optional: autodiff (https://github.com/autodiff/autodiff), header-only. Point AUTODIFF_INC at a checkout
# (-DAUTODIFF_INC=/path/to/autodiff, or the environment variable of the same name) to build the tests that
# cover the automatic-differentiation adapter; without it those tests are compiled out and everything else
# builds unchanged.
if(NOT DEFINED AUTODIFF_INC AND DEFINED ENV{AUTODIFF_INC})
  set(AUTODIFF_INC $ENV{AUTODIFF_INC})
endif()
if(DEFINED AUTODIFF_INC)
  message(STATUS "autodiff found at ${AUTODIFF_INC}: the AD tests are enabled")
  include_directories(${AUTODIFF_INC})
endif()

include_directories(../src/)
include_directories(../../fdaPDE/)
# enable testing functionality using gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  # version 1.14.0
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(fdapde_test main.cpp)
target_link_libraries (fdapde_test Eigen3::Eigen)
target_link_libraries (fdapde_test gtest_main)

add_test(NAME fdapde_test
	 COMMAND $<TARGET_FILE:fdapde_test>)

include(GoogleTest)
gtest_discover_tests(fdapde_test)
