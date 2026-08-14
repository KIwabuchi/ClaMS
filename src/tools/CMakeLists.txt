# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

add_basic_executable(conv_knng_to_mreach_dist conv_knng_to_mreach_dist.cpp)
setup_saltatlas_target(conv_knng_to_mreach_dist)

add_basic_executable(copy_pm_datastore copy_pm_datastore.cpp)
setup_saltatlas_target(copy_pm_datastore)