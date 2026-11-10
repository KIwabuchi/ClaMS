# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

file(GLOB_RECURSE PYTHON_BENCHMARK_MODULES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(PYTHON_MODULE IN LISTS PYTHON_BENCHMARK_MODULES)
    file(RELATIVE_PATH MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${PYTHON_MODULE}")
    configure_file("${PYTHON_MODULE}" "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PATH}" COPYONLY)
endforeach()
