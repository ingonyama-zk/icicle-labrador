cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(labrador_examples)

include(CTest)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(check_lnplabrador_backend_params
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/../scripts/lnplabrador.py
    sync-backend
    --check
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
  COMMENT "Checking generated backend parameters against scripts/para1.py"
  VERBATIM
)

set(LABRADOR_COMMON_SOURCES
  utils.cpp 
  exact_ring_backend.cpp
  prover.cpp
  verifier.cpp
  shared.cpp
  types.cpp
  oracle.cpp
  proof_codec.cpp
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

add_executable(section56_test
  section56_test.cpp
  ${LABRADOR_COMMON_SOURCES}
)

add_executable(exact_ring_backend_test
  exact_ring_backend_test.cpp
  exact_ring_backend.cpp
)

if(BUILD_TESTING)
  add_test(NAME relation_runner_paper_schedule_audit COMMAND lab_runner --audit-paper-schedule)
  add_test(NAME exact_q40_ring_backend COMMAND exact_ring_backend_test)
  add_test(NAME section_5_6_small_proof COMMAND section56_test CPU)
endif()

add_executable(lab_c ../scripts/lab.c)
add_dependencies(lab_c check_lnplabrador_backend_params)
target_link_libraries(lab_c PRIVATE m)

install(TARGETS example lab_runner lab_c
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/")

foreach(target example lab_runner section56_test exact_ring_backend_test)
  add_dependencies(${target} check_lnplabrador_backend_params)
  target_include_directories(${target} PRIVATE "../icicle/include")
  # CMAKE_SOURCE_DIR points to src/, but we need the parent directory.
  target_link_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/../build/icicle")
  target_link_libraries(${target} PRIVATE icicle_ring_labradorq40 icicle_device icicle_hash)
endforeach()

if(BACKEND_DIR)
  add_compile_definitions(BACKEND_DIR="${BACKEND_DIR}")
endif()
