# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


add_basic_executable(run_hdbscan_clustering run_hdbscan_clustering.cpp)
setup_omp_target(run_hdbscan_clustering)
setup_metall_target(run_hdbscan_clustering)
setup_spdlog_target(run_hdbscan_clustering)

add_basic_executable(cluster_noise_points cluster_noise_points.cpp)
setup_omp_target(cluster_noise_points)
setup_metall_target(cluster_noise_points)
setup_spdlog_target(cluster_noise_points)