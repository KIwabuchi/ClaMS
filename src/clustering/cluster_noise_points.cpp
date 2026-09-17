// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// Assign cluster IDs to noise points by traversing the MST edges.
// Traverse the MST edges from each noise point in BFS manner until a point that
// belongs to a cluster is found.

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "../common.hpp"
#include "../details/multithread_adjacency_list.hpp"

using namespace clams;

template <typename K, typename V>
using map_t = boost::unordered::unordered_flat_map<K, V>;

template <typename K>
using set_t = boost::unordered::unordered_flat_set<K>;

struct option {
  std::filesystem::path mst_edges_path;
  bool                  metall_mst{false};
  std::filesystem::path cluster_ids_input_path;
  std::filesystem::path cluster_ids_out_path;
};

void show_help() {
  std::cout
      << "<<Usage>>\n"
         "Required arguments:\n"
         "  -m <path> path to a file or a directory that contains input MST.\n"
         "  -c <path> path to input cluster IDs.\n"
         "  -o <path> path to output cluster IDs.\n"
         "Optional arguments:\n"
         "  -M If specified, input is a Metall datastore.\n"
         "  -h Show help."
      << std::endl;
}

void parse_option(int argc, char* argv[], option& opt) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "m:c:o:Mh")) != -1) {
    switch (opt_char) {
      case 'm':
        opt.mst_edges_path = optarg;
        break;
      case 'c':
        opt.cluster_ids_input_path = optarg;
        break;
      case 'o':
        opt.cluster_ids_out_path = optarg;
        break;
      case 'M':
        opt.metall_mst = true;
        break;
      case 'h':
        show_help();
        std::exit(EXIT_SUCCESS);
      default:
        show_help();
        std::exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char* argv[]) {
  option opt;
  parse_option(argc, argv, opt);

  multithread_adjacency_list<id_t, std::pair<id_t, distance_t>> mst_graph;
  if (opt.metall_mst) {
    spdlog::info("Attaching MST in Metall datastore");
    metall::manager metall_manager(metall::open_read_only, opt.mst_edges_path);
    auto*           input_mst_edges =
        metall_manager.find<weighted_edge_list_t>(metall::unique_instance)
            .first;
    if (!input_mst_edges) {
      spdlog::critical("Failed to find MST edges in Metall datastore at {}",
                       opt.mst_edges_path.string());
      std::abort();
    }
    spdlog::info("#of MST edges: {}", input_mst_edges->size());
    spdlog::info("Copying MST edges from Metall datastore");
    OMP_DIRECTIVE(parallel for)
    for (size_t i = 0; i < input_mst_edges->size(); ++i) {
      const auto& edge = input_mst_edges->at(i);
      mst_graph.add(edge.ids[0], {edge.ids[1], edge.distance});
      mst_graph.add(edge.ids[1], {edge.ids[0], edge.distance});
    }
  } else {
    spdlog::info("Reading MST edges");
    weighted_edge_list_t input_mst_edges;
    read_edges(opt.mst_edges_path, input_mst_edges);
    spdlog::info("#of MST edges: {}", input_mst_edges.size());
    OMP_DIRECTIVE(parallel for)
    for (size_t i = 0; i < input_mst_edges.size(); ++i) {
      const auto& edge = input_mst_edges.at(i);
      mst_graph.add(edge.ids[0], {edge.ids[1], edge.distance});
      mst_graph.add(edge.ids[1], {edge.ids[0], edge.distance});
    }
  }

  if (mst_graph.empty()) {
    spdlog::warn("No MST edges found in the input file or directory: {}",
                 opt.mst_edges_path.string());
    return EXIT_SUCCESS;
  }

  map_t<id_t, id_t> point_cluster_map;
  read_cluster_ids(clams::find_files(opt.cluster_ids_input_path),
                   point_cluster_map);
  spdlog::info("Read {} points' cluster IDs from {}", point_cluster_map.size(),
               opt.cluster_ids_input_path.string());

  std::vector<id_t> point_ids;
  point_ids.reserve(point_cluster_map.size());
  for (const auto& [point_id, cluster_id] : point_cluster_map) {
    point_ids.push_back(point_id);
  }

  spdlog::info(
      "Assigning cluster IDs to noise points by traversing the MST edges");
  // Start a timer to measure the time taken for assigning cluster IDs to noise
  // points
  auto kernel_timer = spdlog::stopwatch();

  std::size_t n_noise_points    = 0;
  std::size_t n_assigned_points = 0;
  // NOTE: point_cluster_map may be updated concurrently by multiple threads,
  // which causes data races, but it is acceptable for this use case. We employ
  // this algorithm because it is simple and fast.
  OMP_DIRECTIVE(parallel for reduction(+ : n_noise_points, n_assigned_points))
  for (size_t i = 0; i < point_ids.size(); ++i) {
    const auto point_id = point_ids.at(i);
    if (point_cluster_map.at(point_id) != k_noise_cluster_id) {
      continue;  // Already assigned to a cluster
    }
    ++n_noise_points;

    std::vector<std::pair<id_t, distance_t>> bfs_front;

    set_t<id_t> visited;
    bfs_front.push_back({point_id, 0});
    visited.insert(point_id);
    bool found_cluster = false;

    // Level-order traversal (BFS) to find the nearest cluster for the noise
    // point. Within each level, visit closer neighbors first
    while (!bfs_front.empty()) {
      std::vector<std::pair<id_t, distance_t>> next;
      for (const auto& [pid, _] : bfs_front) {
        // Traverse the neighbors of the current point in the MST
        for (auto nitr = mst_graph.values_begin(pid);
             nitr != mst_graph.values_end(pid); ++nitr) {
          const auto& [nid, ndist] = *nitr;
          if (visited.count(nid) > 0) {
            continue;  // Already visited, e.g., the node we came from
          }
          if (point_cluster_map.at(nid) != k_noise_cluster_id) {
            // Found a neighbor that belongs to a cluster
            // Update the table without locking, which may cause data races but
            // is acceptable for this use case
            point_cluster_map.at(point_id) = point_cluster_map.at(nid);
            found_cluster                  = true;
            ++n_assigned_points;
            goto BFS_COMPLETE;
          } else {
            visited.insert(nid);
            next.push_back({nid, ndist});
          }
        }
      }
      bfs_front.swap(next);
      // Sort the BFS front by distance to prioritize closer neighbors
      std::sort(
          bfs_front.begin(), bfs_front.end(),
          [](const auto& a, const auto& b) { return a.second < b.second; });
    }
  BFS_COMPLETE:

    if (!found_cluster) {
      spdlog::warn("Point {} could not be assigned to any cluster.", point_id);
    }
  }
  const auto kernel_elapsed_time = kernel_timer.elapsed();
  spdlog::info("Finished assigning cluster IDs to noise points {}s",
               kernel_elapsed_time.count());
  spdlog::info("Number of noise points in the original data: {}",
               n_noise_points);
  spdlog::info("Number of remaining noise points: {}",
               n_noise_points - n_assigned_points);

  dump_point_cluster_ids(point_cluster_map, opt.cluster_ids_out_path);
  spdlog::info("Dumped point cluster IDs to {}",
               opt.cluster_ids_out_path.string());

  return EXIT_SUCCESS;
}
