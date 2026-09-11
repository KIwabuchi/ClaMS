// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// Assign cluster IDs to noise points by traversing the MST edges.
// Traverse the MST edges from each noise point in BFS manner until a point that
// belongs to a cluster is found.

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <metall/metall.hpp>

#include "../common.hpp"
#include "../details/multithread_adjacency_list.hpp"

using namespace clams;

template <typename K, typename V>
using map_t   = boost::unordered::unordered_flat_map<K, V>;
using graph_t = multithread_adjacency_list<id_t, std::pair<id_t, distance_t>>;

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

  graph_t mst_graph;
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
    spdlog::info("Constructing MST graph from Metall datastore");
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
    spdlog::info("Constructing MST graph from input file");
    OMP_DIRECTIVE(parallel for)
    for (size_t i = 0; i < input_mst_edges.size(); ++i) {
      const auto& edge = input_mst_edges.at(i);
      mst_graph.add(edge.ids[0], {edge.ids[1], edge.distance});
      mst_graph.add(edge.ids[1], {edge.ids[0], edge.distance});
    }
  }
  spdlog::info("Finished constructing MST graph");

  if (mst_graph.empty()) {
    spdlog::warn("No MST edges found in the input file or directory: {}",
                 opt.mst_edges_path.string());
    return EXIT_SUCCESS;
  }

  map_t<id_t, id_t> point_cluster_map;
  read_cluster_ids(opt.cluster_ids_input_path, point_cluster_map);
  spdlog::info("Read {} points' cluster IDs from {}", point_cluster_map.size(),
               opt.cluster_ids_input_path.string());

  spdlog::info("Assigning cluster IDs to noise points by edge weight");
  auto kernel_timer = spdlog::stopwatch();

  using heap_edge_t = std::tuple<distance_t, id_t, id_t>;
  std::priority_queue<heap_edge_t, std::vector<heap_edge_t>, std::greater<>>
      edge_heap;

  // Initialize the heap with edges connected to non-noise points
  std::size_t n_noise_points    = 0;
  std::size_t n_assigned_points = 0;
  for (const auto& [point_id, cluster_id] : point_cluster_map) {
    if (cluster_id == k_noise_cluster_id) {
      ++n_noise_points;
      continue;
    }

    for (auto edge_itr = mst_graph.values_begin(point_id);
         edge_itr != mst_graph.values_end(point_id); ++edge_itr) {
      const auto& neighbor_id = edge_itr->first;
      const auto& distance    = edge_itr->second;
      if (point_cluster_map.at(neighbor_id) == k_noise_cluster_id) {
        edge_heap.emplace(distance, point_id, neighbor_id);
      }
    }
  }

  while (!edge_heap.empty()) {
    const auto edge = edge_heap.top();
    edge_heap.pop();
    const auto predecessor_id   = std::get<1>(edge);
    const auto current_point_id = std::get<2>(edge);

    if (point_cluster_map.at(current_point_id) != k_noise_cluster_id) {
      // Skip this edge if the neighbor point has already been visited.
      continue;
    }

    const auto source_cluster_id = point_cluster_map.at(predecessor_id);
    point_cluster_map.at(current_point_id) = source_cluster_id;
    ++n_assigned_points;

    for (auto edge_itr = mst_graph.values_begin(current_point_id);
         edge_itr != mst_graph.values_end(current_point_id); ++edge_itr) {
      const auto& neighbor_id = edge_itr->first;
      const auto& distance    = edge_itr->second;
      if (point_cluster_map.at(neighbor_id) != k_noise_cluster_id) {
        continue;
      }
      edge_heap.emplace(distance, current_point_id, neighbor_id);
    }
  }

  // for (const auto& [point_id, cluster_id] : point_cluster_map) {
  //   if (cluster_id == k_noise_cluster_id) {
  //     spdlog::warn("Point {} could not be assigned to any cluster.",
  //     point_id);
  //   }
  // }
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
