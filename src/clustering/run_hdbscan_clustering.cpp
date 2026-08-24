// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// Implement the algorithm to find clusters in (after the MST construction
// step): https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html

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

struct option {
  std::filesystem::path mst_edges_path;
  bool                  metall_mst{false};
  std::filesystem::path cluster_ids_out_path;
  std::size_t           min_cluster_size{0};
  std::filesystem::path cluster_data_out_path;
  bool                  dump_lamba_p{false};
};

void show_help() {
  std::cout
      << "<<Usage>>\n"
         "Required arguments:\n"
         "  -i <path> path to a file or a directory that contains input MST. \n"
         "  -o <path> path to clustering output.\n"
         "  -m <int> min cluster size parameter in HDBSCAN.\n"
         "Optional arguments: "
         "  -M If specified, input is Metall datastore.\n"
         "  -c <path> path to detailed clustering data output.\n"
         "  -P if specified, dump Lamba p values to the output.\n"
         "  -h Show help.\n"
      << std::endl;
}

// parse option using getopt
// i: input mst edge list directory
// o: output directory
// m: minimum cluster size
void parse_option(int argc, char* argv[], option& opt) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:m:Mc:Ph")) != -1) {
    switch (opt_char) {
      case 'i':
        opt.mst_edges_path = std::filesystem::path(optarg);
        break;
      case 'o':
        opt.cluster_ids_out_path = std::filesystem::path(optarg);
        break;
      case 'm':
        opt.min_cluster_size = std::stol(optarg);
        break;
      case 'M':
        opt.metall_mst = true;
        break;
      case 'c':
        opt.cluster_data_out_path = std::filesystem::path(optarg);
        break;
      case 'P':
        opt.dump_lamba_p = true;
        break;
      case 'h':
        show_help();
        std::exit(EXIT_SUCCESS);
      default:
        spdlog::critical("Unknown option: {}\n", opt_char);
        show_help();
        std::exit(EXIT_FAILURE);
    }
  }
}

// Holds information about the left and right clusters that are connected by an
// edge.
struct bridge_edge {
  // Sizes of the left and right clusters.
  std::size_t left_size{0};
  std::size_t right_size{0};

  // ID of the longest edge in the left (or right) cluster.
  // This information is used to construct cluster tree
  id_t left_longest_edge_id{std::numeric_limits<id_t>::max()};
  id_t right_longest_edge_id{std::numeric_limits<id_t>::max()};
};

// overload operator<< for bridge_edge
std::ostream& operator<<(std::ostream& os, const bridge_edge& edge) {
  os << "left_size: " << edge.left_size << ", right_size: " << edge.right_size
     << ", left_longest_edge_id: " << edge.left_longest_edge_id
     << ", right_longest_edge_id: " << edge.right_longest_edge_id;
  return os;
}

// Disjoint Set (Union-Find) component to represent the hierarchy of the
// clusters
struct cluster_djset_node {
  // The root point ID of the cluster this point belongs to
  id_t root{std::numeric_limits<id_t>::max()};
  // How many point are in the cluster, if this point is the root
  // Initial value is 1 for the root point
  size_t size{1};

  // The longest edge in the cluster. Only root keeps this info
  id_t longest_edge_id{std::numeric_limits<id_t>::max()};
};

// Find the root point of the cluster that the 'start_node_id' node
// belongs to. Also performs path compression to speed up future queries.
id_t find_root(map_t<id_t, cluster_djset_node>& nodes, id_t start_node_id) {
  id_t              current = start_node_id;
  std::vector<id_t> path;  // For path compression
  while (current != nodes.at(current).root) {
    path.push_back(current);
    current = nodes.at(current).root;
  }
  // Now 'current' is the root of the cluster.

  // Path compression
  for (const auto& n : path) {
    nodes.at(n).root = current;
  }

  return current;
}

//  Going to connect points in the MST from the shortest edge to the
// longest edge one by one to build the cluster hierarchy.
void build_cluster_hierarchy(const weighted_edge_list_t& mst_edges,
                             std::vector<bridge_edge>&   bridge_edges) {
  // First, each point belongs to the cluster that contains only itself.
  map_t<id_t, cluster_djset_node> cluster_disset;
  for (const auto& edge : mst_edges) {
    cluster_disset[edge.ids[0]] = cluster_djset_node{.root = edge.ids[0]};
    cluster_disset[edge.ids[1]] = cluster_djset_node{.root = edge.ids[1]};
  }

  // When connecting points, need to find out how many points are in each
  // side of connect component of the edge.
  // Edge ID is sequentially assigned to the edge in the MST (mst_edges).
  // As mst_edges is sorted by distance in ascending order,
  // the shortest edge has ID 0, the longest edge has ID mst_edges.size() - 1.
  bridge_edges.reserve(mst_edges.size());
  for (id_t edge_id = 0; edge_id < mst_edges.size(); ++edge_id) {
    const auto& mst_edge = mst_edges[edge_id];
    const auto& left_id  = mst_edge.ids[0];
    const auto& right_id = mst_edge.ids[1];

    // Use the disjoint set to keep track of the cluster
    // sizes.
    const id_t left_cluster_root_id  = find_root(cluster_disset, left_id);
    const id_t right_cluster_root_id = find_root(cluster_disset, right_id);
    // Sanity check. Because edges are in MST, left_root_id and right_root_id
    // should be different always.
    if (left_cluster_root_id == right_cluster_root_id) {
      std::cerr << "Error: left cluster root ID and right cluster root ID must "
                   "be different."
                << std::endl;
      std::abort();
    }
    auto& left_cluster  = cluster_disset[left_cluster_root_id];
    auto& right_cluster = cluster_disset[right_cluster_root_id];

    bridge_edges.emplace_back(
        bridge_edge{.left_size             = left_cluster.size,
                    .right_size            = right_cluster.size,
                    .left_longest_edge_id  = left_cluster.longest_edge_id,
                    .right_longest_edge_id = right_cluster.longest_edge_id});

    // Join the left cluster to the right cluster
    // (there is no specific reason to choose the right cluster as the parent)
    left_cluster.root = right_cluster_root_id;
    right_cluster.size += left_cluster.size;
    // The longest edge in the cluster is the edge currently being processed
    right_cluster.longest_edge_id = edge_id;
  }
}

struct cluster_data {
  static constexpr id_t k_invalid_cluster_id = std::numeric_limits<id_t>::max();

  bool leaf() const { return children[0] == k_invalid_cluster_id; }

  distance_t          birth_distance{0.0};
  double              stability{0.0};
  std::array<id_t, 2> children = {k_invalid_cluster_id, k_invalid_cluster_id};
  bool                selected{false};
  id_t                final_cluster_id{k_invalid_cluster_id};
  size_t              size{0};
  // The ID of the edge that split this cluster into two subclusters.
  id_t split_edge_id{std::numeric_limits<id_t>::max()};
};

std::ostream& operator<<(std::ostream& ofs, const cluster_data& cluster) {
  ofs << cluster.size << ", " << cluster.birth_distance << ", "
      << cluster.stability << ", " << cluster.children[0] << ", "
      << cluster.children[1] << ", " << cluster.selected << ", "
      << cluster.final_cluster_id << ", " << cluster.split_edge_id;
  return ofs;
}

struct point_cluster_data {
  id_t   cluster_id;
  double lambda_p;
};

// double cal_lambda(const distance_t distance) { return (1.0f / distance); }
double cal_lambda(const distance_t distance) {
  if (distance > 0.0) {
    return (1.0f / distance);
  }
  // Avoid divide by 0 when there's duplicate points (giving distance 0)
  // Note that HDBSCAN works better if data is deduplicated first
  else {
    return std::numeric_limits<double>::max();
  }
}

// Assign the cluster to all points in the clusters 'root_edge_id' edge
// connects.
void assign_cluster_to_points(
    const id_t root_edge_id, const weighted_edge_list_t& mst_edges,
    const std::vector<bridge_edge>& bridge_edges, const id_t cluster_id,
    const double lambda_p, map_t<id_t, point_cluster_data>& point_cluster_map) {
  std::stack<id_t> stack;
  stack.push(root_edge_id);

  while (!stack.empty()) {
    const auto edge_id = stack.top();
    stack.pop();

    assert(edge_id < mst_edges.size());
    const auto edge = mst_edges.at(edge_id);

    assert(edge_id < bridge_edges.size());
    const auto& bridge_edge = bridge_edges.at(edge_id);

    point_cluster_map[edge.ids[0]] = {.cluster_id = cluster_id,
                                      .lambda_p   = lambda_p};
    point_cluster_map[edge.ids[1]] = {.cluster_id = cluster_id,
                                      .lambda_p   = lambda_p};

    if (bridge_edge.left_size > 1) {
      stack.push(bridge_edge.left_longest_edge_id);
    }
    if (bridge_edge.right_size > 1) {
      stack.push(bridge_edge.right_longest_edge_id);
    }
  }
}

void condense_cluster_tree(
    const weighted_edge_list_t&     mst_edges,
    const std::vector<bridge_edge>& bridge_edges, const option& opt,
    std::vector<cluster_data>&       clusters,
    map_t<id_t, point_cluster_data>& point_cluster_id_map) {
  // Init with a dummy root cluster that contains all points
  clusters.resize(1);
  clusters.at(0).size           = mst_edges.size() + 1;
  clusters.at(0).birth_distance = std::numeric_limits<distance_t>::max();

  // Edge ID and the cluster ID that the edge belongs to.
  std::deque<std::pair<id_t, id_t> > work_que;
  // All edges (points) belong to the dummy root cluster '0'.
  work_que.emplace_back(mst_edges.size() - 1, 0);
  // remove from the longest edge
  while (!work_que.empty()) {
    const auto edge_id = work_que.front().first;
    assert(edge_id < mst_edges.size());
    const auto cluster_id = work_que.front().second;
    assert(cluster_id < clusters.size());
    work_que.pop_front();
    const auto& edge               = mst_edges[edge_id];
    const auto& left_pid           = edge.ids[0];
    const auto& right_pid          = edge.ids[1];
    const auto  distance           = edge.distance;
    const auto& cc_bridge          = bridge_edges.at(edge_id);
    const auto  left_cluster_size  = cc_bridge.left_size;
    const auto  right_cluster_size = cc_bridge.right_size;
    const auto  lambda_p           = cal_lambda(distance);

    if (left_cluster_size >= opt.min_cluster_size &&
        right_cluster_size >= opt.min_cluster_size) {
      // New clusters have been born
      clusters.emplace_back(cluster_data{.birth_distance = distance});
      clusters.emplace_back(cluster_data{.birth_distance = distance});
      const id_t left_cluster_id         = clusters.size() - 2;
      const id_t right_cluster_id        = clusters.size() - 1;
      clusters.at(left_cluster_id).size  = left_cluster_size;
      clusters.at(right_cluster_id).size = right_cluster_size;

      point_cluster_id_map[left_pid]  = {.cluster_id = left_cluster_id,
                                         .lambda_p   = lambda_p};
      point_cluster_id_map[right_pid] = {.cluster_id = right_cluster_id,
                                         .lambda_p   = lambda_p};

      // Register new clusters to the parent cluster
      auto& parent_cluster         = clusters.at(cluster_id);
      parent_cluster.children[0]   = left_cluster_id;
      parent_cluster.children[1]   = right_cluster_id;
      parent_cluster.split_edge_id = edge_id;

      // Update the stability of the parent cluster
      // from all of the points in the two child clusters falling out
      parent_cluster.stability +=
          (lambda_p - cal_lambda(parent_cluster.birth_distance)) *
          (left_cluster_size + right_cluster_size);

      work_que.emplace_back(cc_bridge.left_longest_edge_id, left_cluster_id);
      work_que.emplace_back(cc_bridge.right_longest_edge_id, right_cluster_id);
    } else if (left_cluster_size < opt.min_cluster_size &&
               right_cluster_size < opt.min_cluster_size) {
      // Both clusters are too small to be independent clusters
      auto& cluster = clusters.at(cluster_id);
      cluster.stability += (lambda_p - cal_lambda(cluster.birth_distance)) *
                           (left_cluster_size + right_cluster_size);
      // Annex both clusters to the current cluster
      assign_cluster_to_points(edge_id, mst_edges, bridge_edges, cluster_id,
                               lambda_p, point_cluster_id_map);
    } else if (left_cluster_size < opt.min_cluster_size) {
      // Left cluster is too small. Annex it to the current cluster.
      point_cluster_id_map[left_pid] = {.cluster_id = cluster_id,
                                        .lambda_p   = lambda_p};
      if (cc_bridge.left_size > 1) {
        assign_cluster_to_points(cc_bridge.left_longest_edge_id, mst_edges,
                                 bridge_edges, cluster_id, lambda_p,
                                 point_cluster_id_map);
      }
      auto& cluster = clusters.at(cluster_id);
      cluster.stability +=
          (lambda_p - cal_lambda(cluster.birth_distance)) * left_cluster_size;

      assert(cc_bridge.right_longest_edge_id < mst_edges.size());
      work_que.emplace_back(cc_bridge.right_longest_edge_id, cluster_id);
    } else if (right_cluster_size < opt.min_cluster_size) {
      // Right cluster is too small. Annex it to the current cluster.
      point_cluster_id_map[right_pid] = {.cluster_id = cluster_id,
                                         .lambda_p   = lambda_p};
      if (cc_bridge.right_size > 1) {
        assign_cluster_to_points(cc_bridge.right_longest_edge_id, mst_edges,
                                 bridge_edges, cluster_id, lambda_p,
                                 point_cluster_id_map);
      }
      auto& cluster = clusters.at(cluster_id);
      cluster.stability +=
          (lambda_p - cal_lambda(cluster.birth_distance)) * right_cluster_size;

      assert(cc_bridge.left_longest_edge_id < mst_edges.size());
      work_que.emplace_back(cc_bridge.left_longest_edge_id, cluster_id);
    } else {
      // This should never happen.
      assert(false);
    }
  }
  // Do not select the dummy root cluster
  // Dummy root cluster has no stability
  clusters.at(0).stability = 0.0;
}

// Recursively annex all descendant clusters to the new cluster.
// Specifically, deselect all descendant clusters and set their
// final_cluster_id to the new cluster ID.
void annex_descendant_clusters(const std::size_t          id,
                               const std::size_t          new_cluster_id,
                               std::vector<cluster_data>& clusters) {
  assert(id < clusters.size());
  auto& cluster            = clusters.at(id);
  cluster.selected         = false;
  cluster.final_cluster_id = new_cluster_id;
  if (cluster.leaf()) {
    return;
  }
  annex_descendant_clusters(cluster.children[0], new_cluster_id, clusters);
  annex_descendant_clusters(cluster.children[1], new_cluster_id, clusters);
}

double extract_clusters_helper(const std::size_t          id,
                               std::vector<cluster_data>& clusters) {
  assert(id < clusters.size());
  auto& cluster = clusters.at(id);
  if (cluster.leaf()) {
    // Leaf clusters are selected initially.
    cluster.selected         = true;
    cluster.final_cluster_id = id;
    return cluster.stability;
  }

  const auto child_stability_0 =
      extract_clusters_helper(cluster.children[0], clusters);
  const auto child_stability_1 =
      extract_clusters_helper(cluster.children[1], clusters);

  if (cluster.stability < child_stability_0 + child_stability_1) {
    // If the sum of the stabilities of the child clusters is greater,
    // set the cluster stability to be the sum of the child stabilities
    cluster.selected  = false;
    cluster.stability = child_stability_0 + child_stability_1;
  } else {
    // Select the cluster and unselect all descendants
    cluster.selected         = true;
    cluster.final_cluster_id = id;
    annex_descendant_clusters(cluster.children[0], id, clusters);
    annex_descendant_clusters(cluster.children[1], id, clusters);
  }

  return cluster.stability;
}

void extract_clusters(std::vector<cluster_data>& clusters) {
  extract_clusters_helper(0, clusters);
  // Ignore the dummy cluster '0'
  // extract_clusters_helper(1, clusters);
  // extract_clusters_helper(2, clusters);
}

/// Dump cluster IDs for each point
/// If renumber_cluster_ids is true, the cluster IDs will be renumbered ---
/// cluster IDs will be assigned sequentially starting from 0.
void dump_cluster_ids(const id_t                             max_point_id,
                      const map_t<id_t, point_cluster_data>& point_cluster_map,
                      const std::vector<cluster_data>&       clusters,
                      const std::filesystem::path&           output_path,
                      const bool                             dump_lambda_p,
                      const bool renumber_cluster_ids) {
  std::ofstream ofs(output_path);
  if (!ofs) {
    spdlog::error("Failed to open {}", output_path.string());
    std::abort();
  }
  spdlog::info("Output cluster IDs to {}", output_path.string());

  map_t<std::size_t, std::size_t> new_ids;
  // '#' is used for comments.
  ofs << "# Point ID\tCluster ID";
  if (dump_lambda_p) {
    ofs << "\tLambda P";
  }
  ofs << std::endl;
  for (std::size_t i = 0; i <= max_point_id; ++i) {
    if (point_cluster_map.count(i) == 0) {
      continue;  // 0-degree point
    }

    const auto initial_cluster_id = point_cluster_map.at(i).cluster_id;
    assert(initial_cluster_id < clusters.size());
    const auto& cluster          = clusters.at(initial_cluster_id);
    const auto& final_cluster_id = cluster.final_cluster_id;

    // Do not select the dummy root cluster.
    // If initial_cluster_id is 0, it means the point is part of the dummy root
    // cluster, which is not a valid cluster.
    //
    // NOTE: cluster.selected is not used here.
    // Just declaring all points in not-selected clusters as noisy points
    // will discard valid points because the selected flag is false for all
    // sub-clusters of a selected clusters always.
    if (initial_cluster_id > 0 &&
        final_cluster_id != cluster_data::k_invalid_cluster_id) {
      if (renumber_cluster_ids) {
        if (new_ids.count(cluster.final_cluster_id) == 0) {
          const auto new_cluster_id = new_ids.size();
          new_ids[final_cluster_id] = new_cluster_id;
        }
        ofs << i << "\t" << new_ids.at(final_cluster_id);
      } else {
        ofs << i << "\t" << final_cluster_id;
      }
      if (dump_lambda_p) {
        ofs << "\t" << point_cluster_map.at(i).lambda_p;
      }
    } else {
      // noisy point
      ofs << i << "\t-1";
      if (dump_lambda_p) {
        ofs << "\t0.0";  // No lambda_p for noisy points
      }
    }
    ofs << "\n";
  }
}

void dump_cluster_data(const std::vector<cluster_data>& clusters,
                       const std::filesystem::path&     output_path) {
  spdlog::info("Dump cluster data to {}", output_path.string());

  std::ofstream ofs(output_path);
  if (!ofs) {
    std::cerr << "Failed to open " << output_path << std::endl;
    std::abort();
  }

  ofs << "# Cluster ID, Size, Birth distance, Stability, Children[0], "
         "Children[1], "
         "Selected, Final Cluster ID, Split Edge ID"
      << std::endl;
  for (std::size_t cid = 0; cid < clusters.size(); ++cid) {
    const auto& cluster = clusters.at(cid);
    ofs << cid << ", " << cluster << std::endl;
  }
}

int main(int argc, char* argv[]) {
  option opt;
  parse_option(argc, argv, opt);
  spdlog::info("Minimum cluster size: {}", opt.min_cluster_size);

  weighted_edge_list_t mst_edges;
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
    mst_edges = *input_mst_edges;  // Copy the edges from Metall
    spdlog::info("#of MST edges: {}", mst_edges.size());
  } else {
    spdlog::info("Reading MST edges");
    read_edges(opt.mst_edges_path, mst_edges);
    assert(opt.min_cluster_size > 1);  // Current implementation restriction.
    spdlog::info("#of MST edges: {}", mst_edges.size());
  }

  if (mst_edges.empty()) {
    spdlog::warn("No MST edges found in the input file or directory: {}",
                 opt.mst_edges_path.string());
    return EXIT_SUCCESS;
  }

  id_t max_point_id = 0;
  for (const auto& edge : mst_edges) {
    max_point_id = std::max(max_point_id, edge.ids[0]);
    max_point_id = std::max(max_point_id, edge.ids[1]);
  }
  spdlog::info("Max point ID: {}", max_point_id);

  spdlog::info("Start clustering");
  spdlog::stopwatch sw_clustering;

  spdlog::info("  Sorting edges");
  {
    std::sort(mst_edges.begin(), mst_edges.end(),
              [](const auto& a, const auto& b) {
                // sort by distance, ascending order
                return a.distance < b.distance;
              });
  }
  spdlog::info("  Elapsed so far {:.3} sec", sw_clustering);

  spdlog::info("  Build the cluster hierarchy");
  // The n-th element corresponds to the n-th edge in the MST.
  std::vector<bridge_edge> bridge_edges;
  build_cluster_hierarchy(mst_edges, bridge_edges);
  spdlog::info("  Elapsed so far {:.3} sec", sw_clustering);

  spdlog::info("  Condense the cluster tree and compute stability scores");
  // Cluster ID is the index in the vector.
  std::vector<cluster_data> clusters;
  // Holds what cluster each point belongs to
  map_t<id_t, point_cluster_data> point_cluster_id_map;
  condense_cluster_tree(mst_edges, bridge_edges, opt, clusters,
                        point_cluster_id_map);
  spdlog::info("  #of unflatten clusters: {}", clusters.size());
  spdlog::info("  Elapsed so far {:.3} sec", sw_clustering);

  spdlog::info("  Extract clusters");
  extract_clusters(clusters);

  spdlog::info("Finished clustering");
  spdlog::info("Elapsed in total {:.3} sec", sw_clustering);

  // Count #of final clusters
  std::size_t num_final_clusters = 0;
  for (const auto& cluster : clusters) {
    num_final_clusters += cluster.selected;
  }
  spdlog::info("#of final clusters: {}", num_final_clusters);

  // Count #of points assigned to clusters
  std::size_t num_points_in_clusters = 0;
  for (const auto& pcdata : point_cluster_id_map) {
    const auto initial_cluster_id = pcdata.second.cluster_id;
    assert(initial_cluster_id < clusters.size());
    const auto& cluster          = clusters.at(initial_cluster_id);
    const auto& final_cluster_id = cluster.final_cluster_id;
    if (initial_cluster_id > 0 &&
        final_cluster_id != cluster_data::k_invalid_cluster_id) {
      ++num_points_in_clusters;
    }
  }
  spdlog::info("#of points assigned to clusters: {}", num_points_in_clusters);

  if (!opt.cluster_ids_out_path.empty()) {
    dump_cluster_ids(max_point_id, point_cluster_id_map, clusters,
                     opt.cluster_ids_out_path, opt.dump_lamba_p, false);
  }

  if (!opt.cluster_data_out_path.empty()) {
    dump_cluster_data(clusters, opt.cluster_data_out_path);
  }

  return EXIT_SUCCESS;
}