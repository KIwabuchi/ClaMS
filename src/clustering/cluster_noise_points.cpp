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

#include "../common.hpp"

using namespace clams;

template <typename K, typename V>
using map_t = boost::unordered::unordered_flat_map<K, V>;

static constexpr id_t k_noise_cluster_id = static_cast<id_t>(-1);

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

void read_cluster_ids(const std::filesystem::path& input_path,
                      map_t<id_t, id_t>&           point_cluster_map) {
  spdlog::info("Reading cluster IDs from {}", input_path.string());
  std::ifstream ifs(input_path);
  if (!ifs) {
    std::cerr << "Failed to open " << input_path << std::endl;
    std::abort();
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;  // Skip empty lines and comments
    }
    std::istringstream iss(line);
    id_t               point_id, cluster_id;
    if (!(iss >> point_id >> cluster_id)) {
      std::cerr << "Error parsing line: " << line << std::endl;
      std::abort();
    }
    point_cluster_map[point_id] = cluster_id;
  }
}

void dump_point_cluster_ids(const map_t<id_t, id_t>&     cluster_id,
                            const std::filesystem::path& output_path) {
  std::ofstream ofs(output_path);
  if (!ofs) {
    std::cerr << "Failed to open " << output_path << std::endl;
    std::abort();
  }

  for (const auto& [i, final_cluster_id] : cluster_id) {
    ofs << i << "\t" << final_cluster_id;
    ofs << "\n";
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to write to " << output_path << std::endl;
    std::abort();
  }
}

int main(int argc, char* argv[]) {
  option opt;
  parse_option(argc, argv, opt);

  map_t<id_t, std::vector<id_t>> mst;
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
    spdlog::info("Copying MST edges from Metall datastore");
    for (const auto& edge : *input_mst_edges) {
      mst[edge.ids[0]].push_back(edge.ids[1]);
      mst[edge.ids[1]].push_back(edge.ids[0]);
    }
    spdlog::info("#of MST edges: {}", mst.size());
  } else {
    spdlog::info("Reading MST edges");
    weighted_edge_list_t input_mst_edges;
    read_edges(opt.mst_edges_path, input_mst_edges);
    spdlog::info("#of MST edges: {}", input_mst_edges.size());
    for (const auto& edge : input_mst_edges) {
      mst[edge.ids[0]].push_back(edge.ids[1]);
      mst[edge.ids[1]].push_back(edge.ids[0]);
    }
  }

  if (mst.empty()) {
    spdlog::warn("No MST edges found in the input file or directory: {}",
                 opt.mst_edges_path.string());
    return EXIT_SUCCESS;
  }

  map_t<id_t, id_t> point_cluster_map;
  read_cluster_ids(opt.cluster_ids_input_path, point_cluster_map);
  spdlog::info("Read {} point cluster IDs from {}", point_cluster_map.size(),
               opt.cluster_ids_input_path.string());

  std::vector<id_t> point_ids;
  point_ids.reserve(point_cluster_map.size());
  for (const auto& [point_id, cluster_id] : point_cluster_map) {
    point_ids.push_back(point_id);
  }

  spdlog::info(
      "Assigning cluster IDs to noise points by traversing the MST edges");
  std::size_t n_noise_points    = 0;
  std::size_t n_assigned_points = 0;
  OMP_DIRECTIVE(parallel for reduction(+ : n_noise_points, n_assigned_points))
  for (size_t i = 0; i < point_ids.size(); ++i) {
    const auto point_id = point_ids.at(i);
    if (point_cluster_map.at(point_id) != k_noise_cluster_id) {
      continue;  // Already assigned to a cluster
    }
    ++n_noise_points;

    std::deque<id_t> bfs_queue;
    bfs_queue.push_back(point_id);
    bool found_cluster = false;

    while (!bfs_queue.empty() && !found_cluster) {
      const auto current_point_id = bfs_queue.front();
      bfs_queue.pop_front();

      for (const auto neighbor_id : mst.at(current_point_id)) {
        if (point_cluster_map.at(neighbor_id) != k_noise_cluster_id) {
          point_cluster_map[point_id] = point_cluster_map.at(neighbor_id);
          found_cluster               = true;
          ++n_assigned_points;
          break;
        } else {
          bfs_queue.push_back(neighbor_id);
        }
      }
    }

    if (!found_cluster) {
      spdlog::warn("Point {} could not be assigned to any cluster.", point_id);
    }
  }
  spdlog::info("Finished assigning cluster IDs to noise points");
  spdlog::info("Number of noise points: {}", n_noise_points);
  spdlog::info("Number of noise points assigned to clusters: {}",
               n_assigned_points);

  dump_point_cluster_ids(point_cluster_map, opt.cluster_ids_out_path);
  spdlog::info("Dumped point cluster IDs to {}",
               opt.cluster_ids_out_path.string());

  return EXIT_SUCCESS;
}
