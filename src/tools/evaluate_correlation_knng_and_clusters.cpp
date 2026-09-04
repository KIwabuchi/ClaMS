// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// This program evaluates how strongly a k-nearest-neighbor graph (kNNG) agrees
// with clustering result. For each point, it checks whether its nearest k
// neighbors belong to the same cluster as the point itself. It reports the
// correlation rate for each k, which is the fraction of neighbors that are in
// the same cluster as the point. A high correlation rate indicates that the
// clustering result is consistent with the local neighborhood structure of the
// data.

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <boost/unordered/unordered_flat_map.hpp>

#include "../common.hpp"
#include "../details/shm_graph.hpp"

namespace omp = metall::utility::omp;

using cluster_id_table_t = boost::unordered::unordered_flat_map<id_t, id_t>;

// todo:
// add 'max_k' option (-k): the maximum number of neighbors to consider for
// correlation analysis.
bool parse_option(int argc, char *argv[],
                  std::filesystem::path &input_knng_path,
                  std::filesystem::path &input_clusters_path, size_t &max_k) {
  input_knng_path.clear();

  int opt;
  while ((opt = ::getopt(argc, argv, "g:c:k:")) != -1) {
    switch (opt) {
      case 'g': {
        input_knng_path = std::filesystem::path(optarg);
        break;
      }
      case 'c': {
        input_clusters_path = std::filesystem::path(optarg);
        break;
      }
      case 'k': {
        max_k = std::stoul(optarg);
        break;
      }
      default: {
        std::cerr << "Unknown option: " << opt << std::endl;
        return false;
      }
    }
  }

  if (input_knng_path.empty()) {
    std::cerr << "No input kNNG path is specified" << std::endl;
    return false;
  }

  if (input_clusters_path.empty()) {
    std::cerr << "No input clusters path is specified" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  std::filesystem::path input_knng_path;
  std::filesystem::path input_clusters_path;
  size_t                max_k = 100;  // default value

  if (!parse_option(argc, argv, input_knng_path, input_clusters_path, max_k)) {
    return EXIT_FAILURE;
  }

  const auto knng_files = clams::find_files(input_knng_path);

  clams::shm_graph_t graph;

  std::cout << "Read knng" << std::endl;
  clams::read_knng(knng_files, graph);
  std::cout << "#of points: " << graph.num_keys() << std::endl;
  std::cout << "#of neighbors: " << graph.num_values() << std::endl;

  cluster_id_table_t cluster_id_table;
  clams::read_cluster_ids(input_clusters_path, cluster_id_table);
  std::cout << "#of points with clusters: " << cluster_id_table.size()
            << std::endl;
  const auto &cluster_ids = cluster_id_table;

  size_t            max_n_neighbors = 0;
  std::vector<id_t> point_ids;
  point_ids.reserve(graph.num_keys());
  for (auto itr = graph.keys_begin(); itr != graph.keys_end(); ++itr) {
    point_ids.push_back(itr->first);
    max_n_neighbors = std::max(max_n_neighbors, graph.num_values(itr->first));
  }
  std::cout << "max_n_neighbors: " << max_n_neighbors << std::endl;

  size_t n_neighbors_with_same_cluster      = 0;
  size_t n_neighbors_with_different_cluster = 0;
  size_t n_missing_neighbors                = 0;
  size_t n_missing_cluster_ids              = 0;
  size_t n_noise_cluster_points             = 0;
  std::cout
      << "k\tsame_cluster\tdifferent_cluster\tcorrelation_rate(%)\tmissing_"
         "neighbors\tmissing_cluster_ids"
      << std::endl;
  for (size_t k = 0; k < std::min<size_t>(max_n_neighbors, max_k); ++k) {
    OMP_DIRECTIVE(parallel for reduction(+ : n_neighbors_with_same_cluster, n_neighbors_with_different_cluster, n_missing_neighbors, n_missing_cluster_ids))
    for (size_t pi = 0; pi < point_ids.size(); ++pi) {
      const auto pid = point_ids[pi];
      if (graph.num_values(pid) <= k) {
        ++n_missing_neighbors;
        continue;
      }

      const auto neighbor_id          = (graph.values_begin(pid) + k)->first;
      const auto point_cluster_itr    = cluster_ids.find(pid);
      const auto neighbor_cluster_itr = cluster_ids.find(neighbor_id);
      if (point_cluster_itr == cluster_ids.end() ||
          neighbor_cluster_itr == cluster_ids.end()) {
        ++n_missing_cluster_ids;
        continue;
      }
      const auto point_cluster_id    = point_cluster_itr->second;
      const auto neighbor_cluster_id = neighbor_cluster_itr->second;
      if (point_cluster_id == clams::k_noise_cluster_id ||
          neighbor_cluster_id == clams::k_noise_cluster_id) {
        ++n_noise_cluster_points;
        continue;
      }

      if (point_cluster_itr->second == neighbor_cluster_itr->second) {
        ++n_neighbors_with_same_cluster;
      } else {
        ++n_neighbors_with_different_cluster;
      }
    }
    // Show statistics for each k
    const auto n_neighbors_with_clusters =
        n_neighbors_with_same_cluster + n_neighbors_with_different_cluster;
    const auto correlation_rate =
        n_neighbors_with_clusters == 0
            ? 0.0
            : static_cast<double>(n_neighbors_with_same_cluster) /
                  n_neighbors_with_clusters;
    std::cout << k + 1 << "\t" << n_neighbors_with_same_cluster << "\t"
              << n_neighbors_with_different_cluster << "\t"
              << correlation_rate * 100.0 << "\t" << n_missing_neighbors << "\t"
              << n_missing_cluster_ids << std::endl;
  }

  return EXIT_SUCCESS;
}
