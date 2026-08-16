cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(labrador_examples)

set(LABRADOR_COMMON_SOURCES
  utils.cpp 
  prover.cpp
  verifier.cpp
  shared.cpp
  types.cpp
  oracle.cpp
  test_helpers.cpp
)

add_executable(example
  example.cpp
  benchmarking.cpp
  ${LABRADOR_COMMON_SOURCES}
)

add_executable(lab_runner
  relation_runner.cpp
  ${LABRADOR_COMMON_SOURCES}
)

add_executable(lab_c ../scripts/lab.c)
target_link_libraries(lab_c PRIVATE m)

install(TARGETS example lab_runner lab_c
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

foreach(target example lab_runner)
  target_include_directories(${target} PRIVATE "../icicle/include")
  # CMAKE_SOURCE_DIR points to src/, but we need the parent directory.
  target_link_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/../build/icicle")
  target_link_libraries(${target} PRIVATE icicle_ring_babykoala icicle_device icicle_hash)
endforeach()

if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()
