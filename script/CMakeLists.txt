# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


file(GLOB PYTHON_MODULES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
file(GLOB_RECURSE PYTHON_ETC_MODULES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/etc/*.py")
foreach(PYTHON_MODULE IN LISTS PYTHON_MODULES PYTHON_ETC_MODULES)
    file(RELATIVE_PATH MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${PYTHON_MODULE}")
    configure_file("${PYTHON_MODULE}" "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PATH}" COPYONLY)
endforeach()

add_subdirectory(benchmark)
