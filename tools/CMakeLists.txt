cmake_minimum_required(VERSION 3.20)

project(maskgen_cgal_tools LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)

find_package(CGAL CONFIG REQUIRED)

add_executable(optimal_convex_partition_cli optimal_convex_partition_cli.cpp)
target_compile_features(optimal_convex_partition_cli PRIVATE cxx_std_17)
target_link_libraries(optimal_convex_partition_cli PRIVATE CGAL::CGAL)
