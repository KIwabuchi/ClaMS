# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


add_basic_executable(connect_ccs_random_dist connect_ccs_random_dist.cpp)
setup_saltatlas_target(connect_ccs_random_dist)

add_basic_executable(connect_mfc connect_mfc.cpp)
setup_saltatlas_target(connect_mfc)