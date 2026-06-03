# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

cmake_minimum_required(VERSION 3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)

# -------------------------------------------------------------------------------- #
# CMake policy
# -------------------------------------------------------------------------------- #
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    cmake_policy(SET CMP0077 NEW)
endif ()

# When using the URL download method with the ExternalProject_Add() or
# FetchContent_Declare() commands, sets the timestamps of extracted contents
# to the time of extraction.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif ()

# Accept <PACKAGENAME>_ROOT variables for find_package() command.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27")
    cmake_policy(SET CMP0144 NEW)
endif ()

# find_package(Boost) to search for the upstream BoostConfig.cmake.
# Disable CMake's FindBoost module.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif ()

# -------------------------------------------------------------------------------- #
# Project configuration
# -------------------------------------------------------------------------------- #
project("ClaMS" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE is set as Release")
endif ()

#
# Threads
#
find_package(Threads REQUIRED)

#
# OpenMP
#
find_package(OpenMP)

#
# MPI
find_package(MPI)

#
# Boost
set(BOOST_SOURCE_DIR CACHE STRING "Path to Boost libraries directory already uncompressed. Must be a version that supports CMake.")
set(BOOST_FETCH_URL
    "https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz"
    CACHE STRING "URL or file path to an archived Boost source. Must be a version that supports CMake.")

# Link Boost UUID with libatomic fails on LC machines
option(BOOST_UUID_LINK_LIBATOMIC "Link Boost UUID with libatomic" OFF)
set(BOOST_COMPS json unordered interprocess container property_tree uuid graph static_string)
include(fetch_boost)
if (BOOST_SOURCE_DIR)
    fetch_boost(SOURCE ${BOOST_SOURCE_DIR} COMPONENTS ${BOOST_COMPS})
elseif (BOOST_FETCH_URL)
    fetch_boost(URL ${BOOST_FETCH_URL} COMPONENTS ${BOOST_COMPS})
else ()
    # Use default Boost version if not specified by the user
    fetch_boost(URL "https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz"
            COMPONENTS ${BOOST_COMPS})
endif ()

# Boost Components to link
foreach (lib ${BOOST_INCLUDE_LIBRARIES})
    list(APPEND BOOST_COMPONENTS "Boost::${lib}")
endforeach ()

# spdlog
#
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.1
        SYSTEM)
set(SPDLOG_INSTALL On)
FetchContent_MakeAvailable(spdlog)

#
# Cereal
#
set(JUST_INSTALL_CEREAL TRUE)
FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG v1.3.2
        SYSTEM
)
FetchContent_MakeAvailable(cereal)

#
#  YGM
if (MPI_CXX_FOUND)
    FetchContent_Declare(ygm
            GIT_REPOSITORY https://github.com/LLNL/ygm.git
            GIT_TAG master
            SYSTEM
    )
    FetchContent_MakeAvailable(ygm)
endif ()

#
# CLI11
find_package(cli11 CONFIG)
if (NOT cli11_FOUND)
    FetchContent_Declare(
            cli11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG v2.3.2
            SYSTEM
    )
    FetchContent_MakeAvailable(cli11)
else ()
    message(STATUS "Found CLI11 dependency " ${cli11_DIR})
endif ()

#
# Metall
#
find_package(Metall QUIET)
if (NOT Metall_FOUND)
    set(JUST_INSTALL_METALL_HEADER TRUE)
    FetchContent_Declare(
            Metall
            GIT_REPOSITORY https://github.com/LLNL/metall.git
            GIT_TAG master
            SYSTEM)
    FetchContent_MakeAvailable(Metall)
endif ()

#
# SALTATLAS
#
set(JUST_INSTALL_SALTATLAS ON)
set(SALTATLAS_USE_METALL ON)
FetchContent_Declare(
        SALTATLAS
        GIT_REPOSITORY https://github.com/LLNL/saltatlas.git
        GIT_TAG v0.3-dev
        SYSTEM
)
FetchContent_MakeAvailable(SALTATLAS)
set(SALTATLAS_INCLUDE_DIRS ${FETCHCONTENT_BASE_DIR}/saltatlas-src/include)
# cmake_path(ABSOLUTE_PATH ${SALTATLAS_INCLUDE_DIRS} OUTPUT_VARIABLE
# SALTATLAS_INCLUDE_DIRS)

#
# Compile options
#
function(add_common_compile_options name)
    # Common
    target_compile_options(${name} PRIVATE -Wall)

    # Debug
    target_compile_options(${name} PRIVATE $<$<CONFIG:Debug>:-Og>)
    target_compile_options(${name} PRIVATE $<$<CONFIG:Debug>:-g3>)
    target_compile_options(${name} PRIVATE $<$<CONFIG:Debug>:-Wextra>)
    if (Linux)
        target_compile_options(${name} PRIVATE $<$<CONFIG:Debug>:-pg>)
    endif ()

    # Release
    target_compile_options(${name} PRIVATE $<$<CONFIG:Release>:-Ofast>)
    target_compile_options(${name} PRIVATE $<$<CONFIG:Release>:-DNDEBUG>)
    target_compile_options(${name} PRIVATE $<$<CONFIG:Release>:-march=native>)

    # Release with debug info
    target_compile_options(${name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-Ofast>)
    target_compile_options(${name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-g3>)
    if (Linux)
        target_compile_options(${name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-pg>)
    endif ()
endfunction()

#
# Add executable
#
function(add_basic_executable target_name src_name)
    add_executable(${target_name} ${src_name})
    #  target_include_directories(${target_name}
    #                             PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_common_compile_options(${target_name})
    target_link_libraries(${target_name} PRIVATE Threads::Threads ${BOOST_COMPONENTS})
endfunction()

#
# spdlog
#
function(setup_spdlog_target exe_name)
    target_link_libraries(${exe_name} PUBLIC spdlog::spdlog)
endfunction()

#
# Link OpenMP
#
function(setup_omp_target target_name)
    target_link_libraries(${target_name} PRIVATE Threads::Threads)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endfunction()

#
# Link MPI
#
function(setup_mpi_target target_name)
    target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
endfunction()

#
# Link YGM
#
function(setup_ygm_target exe_name)
    setup_mpi_target(${exe_name})
    target_link_libraries(${exe_name} PUBLIC cereal::cereal)
    target_link_libraries(${exe_name} PUBLIC Threads::Threads)
    target_link_libraries(${exe_name} PUBLIC ${BOOST_LIBS})
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(${exe_name} PRIVATE rt)
    endif ()
    target_link_libraries(${exe_name} PRIVATE ygm::ygm)
endfunction()

#
# Link AMST
#
function(setup_amst_target exe_name)
    setup_ygm_target(${exe_name})
    target_link_libraries(${exe_name} PUBLIC CLI11::CLI11)
    # target_link_libraries(${exe_name} PUBLIC  amst::amst)
    target_link_libraries(${exe_name} PUBLIC ${BOOST_LIBS})
endfunction()

#
# Metall
#
function(setup_metall_target exe_name)
    target_link_libraries(${exe_name} PUBLIC ${BOOST_LIBS})
    target_link_libraries(${exe_name} PUBLIC Threads::Threads)
    #    target_link_libraries(${exe_name} PRIVATE stdc++fs)
    target_link_libraries(${exe_name} PUBLIC Metall::Metall)
endfunction()

#
# SALTATLAS
#
# message(STATUS "SALTATLAS_INCLUDE_DIRS: ${SALTATLAS_INCLUDE_DIRS}")
function(setup_saltatlas_target exe_name)
    setup_metall_target(${exe_name})
    setup_ygm_target(${exe_name})
    target_include_directories(${exe_name} PUBLIC ${SALTATLAS_INCLUDE_DIRS})
endfunction()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(script)