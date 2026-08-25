#define METALL_DISABLE_CONCURRENCY
#define METALL_DISABLE_OBJECT_CACHE

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cereal/types/array.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/io/multi_output.hpp>
#include <ygm/utility/timer.hpp>

#include "../common.hpp"
#include "distributed_clustering_build_hierarchy.hpp"
#include "distributed_clustering_hdbscan.hpp"
#include "distributed_clustering_mst_contraction.hpp"
#include "distributed_clustering_types.hpp"

using namespace clams::clustering;

uint32_t LOCAL_COPY_NUM_SUPERNODES_THRESHOLD = 25000;

struct option {
  std::filesystem::path mst_edges_path;
  uint32_t              min_cluster_size{2};
  std::filesystem::path cluster_ids_out_path;
  std::filesystem::path cluster_data_out_path;
  int                   num_files_to_output{0};
  bool                  metall_mst{false};
  bool                  verbose{false};
  bool                  read_mst_with_edge_ids{false};
  bool                  output_invalid_clusters_also{false};
};

void show_help() {
  std::cout
      << "<<Usage>>\n"
         "Required arguments:\n"
         "  -i <path> Path to a file or a directory that contains input MST. \n"
         "  -o <path> Path to clustering output.\n"
         "  -m <int>  Min cluster size parameter in HDBSCAN.\n"
         "Optional arguments:\n"
         "  -M If specified, input MST is Metall datastore.\n"
         "  -e If specified, read MST file with edge ids included. This option "
         "only works for text file MST input, not Metall input.\n"
         "  -c <path> Path to detailed clustering data output.\n"
         "  -n <int>  Number of files to output. (Default is number of ranks). "
         "If the number of ranks is more than the number of files, then the "
         "outputs will be consolidated to print n files. If the number of "
         "ranks is less than or equal to n, then one file will be outputted "
         "per rank.\n."
         "  -v Verbose printout.\n"
         "  -h Show help.\n"
      << std::endl;
}

// parse option using getopt
// i: input mst edge list directory
// o: output directory
// m: minimum cluster size
std::pair<bool, std::vector<int>> parse_option(int argc, char *argv[],
                                               option &opt) {
  bool             show_help = false;
  std::vector<int> unknown_opts;
  int              opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:m:c:n:Mevh")) != -1) {
    switch (opt_char) {
      case 'i':
        opt.mst_edges_path = std::filesystem::path(optarg);
        break;
      case 'o':
        opt.cluster_ids_out_path = std::filesystem::path(optarg);
        break;
      case 'm':
        opt.min_cluster_size = std::atoi(optarg);
        break;
      case 'c':
        opt.cluster_data_out_path = std::filesystem::path(optarg);
        break;
      case 'n':
        opt.num_files_to_output = std::stoi(optarg);
        break;
      case 'M':
        opt.metall_mst = true;
        break;
      case 'e':
        opt.read_mst_with_edge_ids = true;
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'h':
        show_help = true;
        std::exit(EXIT_SUCCESS);
      default:
        unknown_opts.push_back(opt_char);
        show_help = true;
        // std::exit(EXIT_FAILURE);
    }
  }

  return std::make_pair(show_help, unknown_opts);
}

int main(int argc, char *argv[]) {
  ygm::comm world(&argc, &argv);

  option opt;
  auto   parse_return = parse_option(argc, argv, opt);
  if ((parse_return.first) & world.rank() == 0) {
    if (parse_return.second.size() > 0) {
      for (int opt_char : parse_return.second) {
        spdlog::critical("Ignoring unknown option: {:c}\n", opt_char);
      }
    }
    show_help();
  }

  if (opt.num_files_to_output == 0) {
    opt.num_files_to_output = world.size();
  } else if (world.size() < opt.num_files_to_output) {
    opt.num_files_to_output = world.size();
  }

  if (opt.cluster_ids_out_path.empty()) {
    if (world.rank() == 0) {
      spdlog::info("Error: No output path for cluster labels is provided");
      show_help();
    }
    std::exit(EXIT_FAILURE);
  }

  // Set the min cluster size as a static variable for easy access
  static uint32_t min_cluster_size = opt.min_cluster_size;
  if (world.rank() == 0) {
    spdlog::info("Minimum cluster size: {}", min_cluster_size);
  }

  int num_ranks = world.size();
  if (world.rank() == 0) {
    spdlog::info("Number of ranks: {}", num_ranks);
  }

  // Timers
  spdlog::stopwatch sw_clustering;
  spdlog::stopwatch sw_phase;
  spdlog::stopwatch sw_round;
  spdlog::stopwatch sw_step;

  // Phase 1 times
  float incidence_map_time  = 0.0;
  float contract_edges_time = 0.0;
  float get_components_time = 0.0;
  float update_parent_time  = 0.0;
  float update_edges_time   = 0.0;

  // Phase 2 times
  float assign_edges_to_chains_time              = 0.0;
  float assign_contracted_edges_to_clusters_time = 0.0;
  float fill_initial_cluster_info_time           = 0.0;
  float root_chain_to_clusters_time              = 0.0;

  // Phase 3 times
  float calculate_non_root_chain_size_stability_time = 0.0;
  float calculate_root_chain_size_stability_time     = 0.0;
  float select_clusters_time                         = 0.0;
  float label_points_time                            = 0.0;

  // Initialize edge map of basic endpoints
  // and edge contraction map with edge_contraction_info
  ygm::container::map<id_t, std::pair<id_t, id_t>> edge_endpoints_map(world);
  ygm::container::map<id_t, edge_contraction_info> edge_contraction_map(world);

  // Read MST edges, sort and assign edge ids
  sw_round.reset();
  sw_step.reset();
  if (opt.metall_mst) {
    if (world.rank() == 0) {
      spdlog::info("Attaching MST in Metall datastore");
    }
    metall::manager metall_manager(metall::open_read_only, opt.mst_edges_path);
    auto           *input_mst_edges =
        metall_manager
            .find<clams::weighted_edge_list_t>(metall::unique_instance)
            .first;
    if (world.rank() == 0) {
      if (!input_mst_edges) {
        spdlog::critical("Failed to find MST edges in Metall datastore at {}",
                         opt.mst_edges_path.string());
        std::abort();
      }
      spdlog::info("Copying MST edges from Metall datastore");
    }

    uint64_t num_edges  = (*input_mst_edges).size();
    uint64_t chunk_size = num_edges / world.size() + 1;
    uint64_t start_idx  = chunk_size * world.rank();
    uint64_t end_idx =
        std::min(chunk_size * (world.rank() + 1) - 1, num_edges - 1);

    static std::vector<std::pair<distance_t, std::pair<id_t, id_t>>>
        mst_edge_vector;
    mst_edge_vector.clear();
    for (uint64_t i = start_idx; i <= end_idx; ++i) {
      std::pair<distance_t, std::pair<id_t, id_t>> array_item =
          std::make_pair((*input_mst_edges)[i].distance,
                         std::make_pair((*input_mst_edges)[i].ids[0],
                                        (*input_mst_edges)[i].ids[1]));
      mst_edge_vector.push_back(array_item);
    }

    if ((world.rank() == 0) & (opt.verbose)) {
      spdlog::info("  Time to read edges (s): {:.3f}", sw_step);
    }
    sw_step.reset();

    sort_and_process_mst_edges_into_maps(mst_edge_vector, edge_endpoints_map,
                                         edge_contraction_map);

    if ((world.rank() == 0) & (opt.verbose)) {
      spdlog::info("  Time to sort edges and fill YGM maps (s): {:.3f}",
                   sw_step);
    }

  } else {
    if (opt.read_mst_with_edge_ids) {
      if (world.rank() == 0) {
        spdlog::info(
            "Reading MST edges from file with edge IDs already assigned");
        spdlog::info("Reading edges from: {}", opt.mst_edges_path.string());
      }

      std::vector<std::string> input_paths{opt.mst_edges_path.c_str()};
      ygm::io::line_parser     mst_line_parser(world, input_paths);

      auto line_parser_lambda = [&edge_endpoints_map, &edge_contraction_map](
                                    const std::string &line) {
        if (std::isdigit(line[0])) {
          try {
            std::stringstream ss(line);
            id_t              edge_id;
            id_t              node1, node2;
            distance_t        dist;
            ss >> edge_id >> node1 >> node2 >> dist;
            std::pair<id_t, id_t> edge_endpoints = std::make_pair(node1, node2);
            edge_endpoints_map.async_insert(edge_id, edge_endpoints);
            edge_contraction_map.async_insert(
                edge_id,
                edge_contraction_info{.endpoint_supernode_reps = edge_endpoints,
                                      .distance                = dist});

          } catch (...) {
            std::cout << "Error reading mst line: " << line << std::endl;
          }
        } else {
          std::cout << "Read comment line in mst file: " << line << std::endl;
        }
      };
      mst_line_parser.for_all(line_parser_lambda);
      world.barrier();

    } else {
      if (world.rank() == 0) {
        spdlog::info("Reading MST edges from file and sorting to assign IDs");
        spdlog::info("Reading edges from: {}", opt.mst_edges_path.string());
      }

      std::vector<std::string> input_paths{opt.mst_edges_path.c_str()};
      ygm::io::line_parser     mst_line_parser(world, input_paths);

      // Vector to store mst edges read by this rank
      // Edges stored in the form (distance, endpoint pair (node 1, node2))
      static std::vector<std::pair<distance_t, std::pair<id_t, id_t>>>
          mst_edge_vector;
      mst_edge_vector.clear();

      auto line_parser_lambda = [](const std::string &line) {
        if (std::isdigit(line[0])) {
          try {
            std::stringstream ss(line);
            id_t              node1, node2;
            distance_t        dist;
            ss >> node1 >> node2 >> dist;
            std::pair<distance_t, std::pair<id_t, id_t>> array_item =
                std::make_pair(dist, std::make_pair(node1, node2));
            mst_edge_vector.push_back(array_item);
          } catch (...) {
            std::cout << "Error reading mst line: " << line << std::endl;
          }
        } else {
          std::cout << "Read comment line in mst file: " << line << std::endl;
        }
      };
      mst_line_parser.for_all(line_parser_lambda);
      world.barrier();

      if ((world.rank() == 0) & (opt.verbose)) {
        spdlog::info("  Time to read edges (s): {:.3f}", sw_step);
      }
      sw_step.reset();

      sort_and_process_mst_edges_into_maps(mst_edge_vector, edge_endpoints_map,
                                           edge_contraction_map);

      if ((world.rank() == 0) & (opt.verbose)) {
        spdlog::info("  Time to sort edges and fill YGM maps (s): {:.3f}",
                     sw_step);
      }
    }
  }
  std::stringstream ss;
  ss << "  Number of MST edges: " << edge_endpoints_map.size();
  if (world.rank() == 0) {
    spdlog::info(ss.str());
    spdlog::info(" Time to ingest MST edges (s): {:.3f}", sw_round);
  }

  if (edge_endpoints_map.size() == 0) {
    if (world.rank() == 0) {
      spdlog::info("Error: 0 MST edges read");
      show_help();
    }
    std::exit(EXIT_FAILURE);
  }

  /* Initial set up of needed YGM maps */

  // Map of supernode children of edges in the dendrogram
  // edge id -> array of supernode children (up to 2 children, actually will
  // have 0 or 2 children)
  ygm::container::map<id_t, alpha_edge_info> alpha_edge_map(world);
  auto alpha_edge_map_ptr = alpha_edge_map.get_ygm_ptr();

  /* Phase 1: MST contraction */
  if (world.rank() == 0) {
    spdlog::info("Phase 1: MST contraction");
  }
  sw_phase.reset();

  static uint32_t final_round;
  {
    // // Supernode map of supernode old supernode rep -> new supernode rep
    // // This copies over the ygm disjoint set information for accessibility
    // ygm::container::map<id_t, id_t> supernode_rep_map(world);

    // Incidence map of supernode rep (we know the round, so we drop it) ->
    // vector of all incident edge ids
    ygm::container::map<id_t, id_t> min_incident_edge_map(world);

    // Create a disjoint set
    ygm::container::disjoint_set<id_t> tree_components_djset(world);

    // Initial set up
    sw_step.reset();
    fill_init_min_incident_edge_map(min_incident_edge_map, edge_endpoints_map);
    incidence_map_time += sw_step.elapsed().count();

    // Max possible number of contraction rounds is log2(number of edges in
    // MST)
    uint32_t max_possible_rounds =
        static_cast<int>(std::ceil(std::log2(edge_endpoints_map.size())));

    // Rounds of successive MST contraction
    static uint32_t round;
    round = 1;
    while (round <= max_possible_rounds) {
      sw_round.reset();
      if (opt.verbose && world.rank() == 0) {
        spdlog::info("  MST contraction round {}", round);
      }

      // Find all edges to contract and add to disjoint set
      sw_step.reset();
      contract_edges(round, min_incident_edge_map, edge_contraction_map,
                     alpha_edge_map_ptr, tree_components_djset);

      // Count how many edges were contracted this round
      id_t num_edges_contracted = 0;
      edge_contraction_map.for_all(
          [&num_edges_contracted]([[maybe_unused]] const id_t &edge_id,
                                  const edge_contraction_info  edge_info) {
            if (edge_info.contraction_round == round) {
              ++num_edges_contracted;
            }
          });
      world.barrier();

      contract_edges_time += sw_step.elapsed().count();
      sw_step.reset();

      // Update the succ dendrogram parents for edges that were contracted
      // before this round
      if (round >= 2) {
        if (min_incident_edge_map.size() <=
            LOCAL_COPY_NUM_SUPERNODES_THRESHOLD) {
          update_edge_chain_parent_edge_id_from_local_map(
              round, edge_contraction_map, min_incident_edge_map);
        } else {
          update_edge_chain_parent_edge_id(round, edge_contraction_map,
                                           min_incident_edge_map);
        }
      }
      world.barrier();

      update_parent_time += sw_step.elapsed().count();
      sw_step.reset();

      // clear the min incident edge map since we're done with it for this
      // round
      id_t num_nodes_at_start_of_round = min_incident_edge_map.size();
      min_incident_edge_map.clear();
      world.barrier();

      // Get the supernodes (connected components) for this round of
      // contraction
      sw_step.reset();
      tree_components_djset.all_compress();
      world.barrier();
      get_components_time += sw_step.elapsed().count();
      sw_step.reset();

      if (opt.verbose) {
        std::stringstream ss;
        ss << "    Number of edges contracted: "
           << ygm::sum(num_edges_contracted, world)
           << ", Number of nodes at start of round: "
           << num_nodes_at_start_of_round << ", Number of new supernodes: "
           << tree_components_djset.num_sets();
        if (world.rank() == 0) {
          spdlog::info(ss.str());
        }
      }

      // If all edges are contractable, then stop, don't do another round of
      // contraction
      if (tree_components_djset.num_sets() == 1) {
        --round;
        final_round = round;
        if (world.rank() == 0) {
          spdlog::info(" Total number of MST contraction rounds: {}",
                       final_round);
        }

        break;
      }

      sw_step.reset();

      // Update the edge info: the edge endpoints and chain supernode
      // Fill the keys of the min incident edge map with the new supernode
      // reps

      // To avoid bottleneck of lookups from a small number of supernodes, if
      // we have few enough new supernodes, create a local copy of
      // tree_components_djset to update edge info
      if (tree_components_djset.num_sets() <=
          LOCAL_COPY_NUM_SUPERNODES_THRESHOLD) {
        update_edge_endpoints_and_chain_supernode_from_local_map(
            round, edge_contraction_map, tree_components_djset);
      }
      // Otherwise, avoid creating local copies and look up new edge info in
      // the supernode_map
      else {
        update_edge_endpoints_and_chain_supernode(round, edge_contraction_map,
                                                  tree_components_djset);
      }
      reinit_min_incident_edge_map(min_incident_edge_map,
                                   tree_components_djset);
      tree_components_djset.clear();
      world.barrier();

      update_edges_time += sw_step.elapsed().count();
      sw_step.reset();

      // Update the incidence map
      update_min_incident_edge_map(edge_contraction_map, min_incident_edge_map);
      incidence_map_time += sw_step.elapsed().count();

      ++round;

      if (opt.verbose && world.rank() == 0) {
        spdlog::info("    Time for this contraction round (s): {:.3f}",
                     sw_round);
      }
    }
  }
  if (world.rank() == 0) {
    spdlog::info(" Time for this phase (s): {:.3f}", sw_phase);
    if (opt.verbose) {
      spdlog::info(" Phase 1 time breakdown:");
      spdlog::info("   Time to fill incidence map (s): {:.3f}",
                   incidence_map_time);
      spdlog::info("   Time to contract edges (s): {:.3f}",
                   contract_edges_time);
      spdlog::info(
          "   Time to get new supernode connected components (s): {:.3f}",
          get_components_time);
      spdlog::info("   Time to update chain parents when possible (s): {:.3f}",
                   update_parent_time);
      spdlog::info(
          "   Time to update edge endpoints, chain supernode (s): {:.3f}",
          update_edges_time);
    }
  }

  /* Phase 2: Expand dendrogram and get cluster hierarchy */
  if (world.rank() == 0) {
    spdlog::info("Phase 2: Construct cluster hierarchy");
  }
  sw_phase.reset();

  // Map of leaf clusters (identified by supernode only) to edges and info
  ygm::container::map<supernode_t, full_leaf_cluster_info> leaf_cluster_map(
      world);

  // YGM map for all chains except the root chain
  // Maps a chain name (supernode) to a pair of
  // cluster map (alpha edge id -> cluster info) and chain info
  ygm::container::map<supernode_t, std::pair<std::map<id_t, full_cluster_info>,
                                             full_chain_info>>
      chain_map(world);

  // Map of root chain edge id to its edges added
  ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
      root_chain_cluster_edges_map(world);

  // Map of root chain cluster edge id -> cluster info
  ygm::container::map<id_t, root_chain_cluster_info> root_chain_cluster_map(
      world);
  // Info of the second child for the bottom root chain cluster
  extra_child_cluster_info root_chain_second_child;
  // Store the bottom edge in the root chain for easy access
  id_t root_chain_min_edge_id;

  // A unique chain name for the root chain
  static supernode_t root_chain_supernode = {0, final_round + 1};

  {
    // Initial setup for this phase

    // Local vector of root chain alpha-edges. We will later use this to fill
    // a YGM array to avoid bottlenecks processing the root chain, which we
    // expect to be the longest
    std::vector<edge_id_with_dist_t> local_root_chain_alpha_edges;

    // YGM set of root chain non-alpha edges
    ygm::container::set<edge_id_with_dist_t> root_chain_non_alpha_edge_set(
        world);

    sw_step.reset();

    // Fill in the missing alpha edge info
    fill_alpha_edge_map(alpha_edge_map, edge_contraction_map,
                        root_chain_supernode);

    // Assign edges to chains
    {
      if (world.rank() == 0 && opt.verbose) {
        spdlog::info(" Assigning edges to chains");
      }

      assign_edges_to_chains(
          edge_contraction_map, chain_map, local_root_chain_alpha_edges,
          root_chain_non_alpha_edge_set, leaf_cluster_map, final_round);

      assign_edges_to_chains_time += sw_step.elapsed().count();

      edge_contraction_map.clear();
      world.barrier();

      if (opt.verbose) {
        id_t num_chains =
            chain_map.size() + leaf_cluster_map.size() + 1;  //+1 for root chain
        id_t num_leaf_chains = leaf_cluster_map.size();
        id_t num_root_chain_alpha_edges =
            ygm::sum(local_root_chain_alpha_edges.size(), world);
        id_t num_root_chain_non_alpha_edges =
            root_chain_non_alpha_edge_set.size();
        id_t local_num_non_root_chain_non_alpha_edges = 0;
        chain_map.for_all([&local_num_non_root_chain_non_alpha_edges](
                              [[maybe_unused]] const supernode_t &chain_name,
                              std::pair<std::map<id_t, full_cluster_info>,
                                        full_chain_info>         &chain) {
          local_num_non_root_chain_non_alpha_edges +=
              chain.second.non_alpha_edges.size();
        });
        world.barrier();
        id_t num_non_root_chain_non_alpha_edges =
            ygm::sum(local_num_non_root_chain_non_alpha_edges, world);

        if (world.rank() == 0) {
          spdlog::info("  Number of chains: {} ", num_chains);
          spdlog::info("  Number of leaf chains: {} ", num_leaf_chains);
          spdlog::info(
              "  Number of non-alpha edges (contracted in round 1) "
              "not in the root chain to assign to clusters: {} ",
              num_non_root_chain_non_alpha_edges);
          spdlog::info("  Number of alpha edges in root chain: {} ",
                       num_root_chain_alpha_edges);
          spdlog::info("  Number of non-alpha edges in root chain: {} ",
                       num_root_chain_non_alpha_edges);
          spdlog::info("    Time (s): {:.3f}", sw_step);
          spdlog::info(
              " Splitting chains into clusters and assigning edges to "
              "clusters");
        }
      }
    }

    // Process the non-root chains
    {
      sw_step.reset();
      assign_non_alpha_edges_to_clusters(chain_map);

      assign_contracted_edges_to_clusters_time += sw_step.elapsed().count();

      // Process the cluster map so far and and fill in missing
      // information on birth distance, parent, children
      sw_step.reset();
      fill_missing_leaf_cluster_info(leaf_cluster_map, alpha_edge_map);
      fill_missing_non_root_chain_cluster_info(chain_map, alpha_edge_map);

      fill_initial_cluster_info_time += sw_step.elapsed().count();

      // Erase alpha edges from the map if they are not in the root chain
      auto delete_non_root_chain_alpha_edges_lambda =
          [alpha_edge_map_ptr](const id_t            &edge_id,
                               const alpha_edge_info &edge_info) {
            if (edge_info.chain_supernode != root_chain_supernode) {
              alpha_edge_map_ptr->async_erase(edge_id);
            }
          };
      alpha_edge_map.for_all(delete_non_root_chain_alpha_edges_lambda);
      world.barrier();
    }

    // Process the root chain
    sw_step.reset();

    // Get the root chain clusters in a YGM array to sort them
    std::vector<std::pair<id_t, root_chain_cluster_info>>
        local_root_chain_clusters;
    local_root_chain_clusters.reserve(local_root_chain_alpha_edges.size());
    for (edge_id_with_dist_t &edge : local_root_chain_alpha_edges) {
      root_chain_cluster_info cluster_info{.lambda_min_edge =
                                               lambda_from_dist(edge.second)};
      local_root_chain_clusters.push_back(
          std::make_pair(edge.first, cluster_info));

      // Add this cluster's own alpha edge to its edge map
      root_chain_cluster_edges_map.async_visit(
          edge.first,
          []([[maybe_unused]] const id_t      &cluster_alpha_edge_id,
             std::vector<edge_id_with_dist_t> &edges,
             const edge_id_with_dist_t        &new_edge) {
            edges.push_back(new_edge);
          },
          edge);
    }

    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
        full_root_chain_cluster_array(world, local_root_chain_clusters);
    full_root_chain_cluster_array.sort();
    local_root_chain_clusters.clear();

    assign_edges_to_chains_time += sw_step.elapsed().count();
    root_chain_to_clusters_time += sw_step.elapsed().count();
    sw_step.reset();

    // Fill in the root chain cluster info
    {
      // Send the root chain non-alpha edges to the appropriate rank for
      // processing
      std::vector<edge_id_with_dist_t> root_chain_non_alpha_edges;
      root_chain_non_alpha_edges = split_root_chain_non_alpha_edges_to_process(
          world, full_root_chain_cluster_array, root_chain_non_alpha_edge_set);

      // Assign root-chain non-alpha edges to their clusters
      assign_root_chain_non_alpha_edges_to_clusters(
          root_chain_non_alpha_edges, full_root_chain_cluster_array,
          root_chain_cluster_edges_map);

      assign_contracted_edges_to_clusters_time += sw_step.elapsed().count();
      root_chain_to_clusters_time += sw_step.elapsed().count();
      sw_step.reset();

      // Fill in root chain cluster info
      supernode_t root_chain_second_child_supernode;
      root_chain_second_child_supernode = fill_missing_root_chain_cluster_info(
          full_root_chain_cluster_array, alpha_edge_map,
          root_chain_cluster_edges_map);
      root_chain_second_child.name = root_chain_second_child_supernode;

      alpha_edge_map.clear();

      // Get the min edge id for the root chain
      static id_t local_root_chain_min_edge_id;
      local_root_chain_min_edge_id = 0;
      if (world.rank() == 0) {
        full_root_chain_cluster_array.async_visit(
            0, []([[maybe_unused]] const id_t              &index,
                  std::pair<id_t, root_chain_cluster_info> &value) {
              local_root_chain_min_edge_id = value.first;
            });
      }
      world.barrier();
      MPI_Allreduce(&local_root_chain_min_edge_id, &root_chain_min_edge_id, 1,
                    mpi_id_type(), MPI_MAX, world.get_mpi_comm());

      fill_initial_cluster_info_time += sw_step.elapsed().count();
      sw_step.reset();

      // The root chain cluster array will go out of scope, copy the info
      // to the local root chain clusters map. We also need the map to look up
      // root chain clusters by their alpha edge id in phase 3
      full_root_chain_cluster_array.for_all(
          [&root_chain_cluster_map](
              const id_t                                     &index,
              const std::pair<id_t, root_chain_cluster_info> &value) {
            root_chain_cluster_map.async_insert(value.first, value.second);
          });
      world.barrier();
      root_chain_to_clusters_time += sw_step.elapsed().count();
    }
    world.barrier();

    if (opt.verbose) {
      id_t        local_num_clusters = 0;
      id_t        largest_chain_size = 0;
      supernode_t largest_chain_name = BLANK_SUPERNODE;
      chain_map.for_all(
          [&local_num_clusters, &largest_chain_size, &largest_chain_name](
              const supernode_t                &chain_name,
              const std::pair<std::map<id_t, full_cluster_info>,
                              full_chain_info> &chain) {
            local_num_clusters += chain.first.size();
            if (chain.first.size() > largest_chain_size) {
              largest_chain_size = chain.first.size();
              largest_chain_name = chain_name;
            }
          });
      id_t num_leaf_clusters = leaf_cluster_map.size();
      id_t num_clusters      = leaf_cluster_map.size() +
                          full_root_chain_cluster_array.size() +
                          ygm::sum(local_num_clusters, world);

      std::stringstream ss;
      ss << "  Most cluster splits (edges with supernode children) in a "
            "non-root chain: "
         << ygm::max(largest_chain_size, world) << " splits";
      if (largest_chain_size == ygm::max(largest_chain_size, world)) {
        ss << " in chain " << largest_chain_name << " ";
        spdlog::info(ss.str());
      }

      if (world.rank() == 0) {
        spdlog::info(
            "  Number of vertices in cluster hierarchy disregarding "
            "min_cluster_size: {} ",
            num_clusters);
        spdlog::info(
            "  Number of leaf clusters disregarding min_cluster_size: {} ",
            num_leaf_clusters);
        spdlog::info("   Time (s): {:.3f}", sw_step);
      }
    }

  }  // End Phase 2

  if (world.rank() == 0) {
    spdlog::info(" Time for this phase (s): {:.3f}", sw_phase);
    if (opt.verbose) {
      spdlog::info(" Phase 2 time breakdown:");
      spdlog::info("   Time to assign edges to chains (s): {:.3f}",
                   assign_edges_to_chains_time);
      spdlog::info(
          "   Time to assign round-1 contracted edges to clusters (s): "
          "{:.3f}",
          assign_contracted_edges_to_clusters_time);
      spdlog::info(
          "   Time to fill missing cluster information (birth "
          "distance, parent, children, etc.) (s): {:.3f} ",
          fill_initial_cluster_info_time);
    }
  }

  /* Phase 3: Select flat clusters */
  if (world.rank() == 0) {
    spdlog::info("Phase 3: Extract flat clustering");
  }
  sw_phase.reset();

  // Initialize map of point id -> cluster id
  ygm::container::map<id_t, cluster_id_t> point_to_cluster_id_map(world);

  {
    // Set up variables used through out Phase 3
    static id_t num_valid_clusters;
    num_valid_clusters = 0;

    // Traverse up the cluster hierarchy and fill in cluster stabilities
    // and sizes of everything except clusters in the root chain, which we'll
    // handle separately in serial
    {
      if (world.rank() == 0 && opt.verbose) {
        spdlog::info(
            " Traversing up cluster hierarchy and calculating stabilities "
            "and sizes for clusters not in root chain");
      }

      sw_step.reset();
      id_t num_clusters_returned =
          traverse_up_cluster_hierarchy_until_root_chain(
              leaf_cluster_map, chain_map, root_chain_cluster_map,
              min_cluster_size, root_chain_supernode);
      num_valid_clusters = ygm::sum(num_clusters_returned, world);

      calculate_non_root_chain_size_stability_time += sw_step.elapsed().count();

      if (world.rank() == 0 && opt.verbose) {
        spdlog::info(
            "   Time to traverse up cluster hierarchy before the root "
            "chain (s): {:.3f}",
            sw_step);
      }
    }

    // Make sure that the bottom two root chain children are valid with
    // at least min cluster size. We need this assumption to process merging
    // invalid root chain clusters and then propagating size/stability
    // This function also gets the root chain second child info and sets the
    // top cluster of the second root chain child as valid
    make_sure_root_chain_bottom_is_valid_and_get_second_child(
        root_chain_cluster_map, root_chain_cluster_edges_map, chain_map,
        leaf_cluster_map, root_chain_min_edge_id, root_chain_second_child,
        min_cluster_size, root_chain_supernode);

    // Merge invalid root chain clusters
    {
      sw_step.reset();

      id_t num_total_root_chain_clusters = root_chain_cluster_map.size();
      if (num_total_root_chain_clusters > 0) {
        if (world.rank() == 0 && opt.verbose) {
          spdlog::info(" Merging invalid root chain clusters");
        }

        // Initialize and sort a YGM array for all the root chain clusters
        // (ignoring min cluster size)
        std::vector<std::pair<id_t, root_chain_cluster_info>>
            local_root_chain_clusters;
        root_chain_cluster_map.for_all(
            [&local_root_chain_clusters](
                const id_t                    &cluster_edge_id,
                const root_chain_cluster_info &cluster_info) {
              local_root_chain_clusters.push_back(
                  std::make_pair(cluster_edge_id, cluster_info));
            });
        world.barrier();

        ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
            full_root_chain_array(world, local_root_chain_clusters);
        world.barrier();
        local_root_chain_clusters.clear();
        full_root_chain_array.sort();
        world.barrier();

        merge_invalid_root_chain_clusters(
            full_root_chain_array, root_chain_cluster_map,
            root_chain_cluster_edges_map, chain_map, leaf_cluster_map,
            min_cluster_size);

        if (world.rank() == 0 && opt.verbose) {
          spdlog::info(
              "  Number of root chain clusters before merging too-small "
              "clusters: {}",
              num_total_root_chain_clusters);
          spdlog::info("   Time to merge root chain clusters (s): {:.3f}",
                       sw_step);
        }
      }
    }

    // Track whether we have selected a root chain cluster in our flat
    // clustering and if so, which one
    static id_t selected_root_chain_cluster_edge_id;
    selected_root_chain_cluster_edge_id = 0;

    // Initialize and sort a YGM array for the valid (post-merging) root chain
    // clusters we kept
    std::vector<std::pair<id_t, root_chain_cluster_info>>
        local_root_chain_clusters;
    root_chain_cluster_map.for_all(
        [&local_root_chain_clusters](
            const id_t                    &cluster_edge_id,
            const root_chain_cluster_info &cluster_info) {
          local_root_chain_clusters.push_back(
              std::make_pair(cluster_edge_id, cluster_info));
        });
    world.barrier();

    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
        root_chain_array(world, local_root_chain_clusters);
    world.barrier();
    local_root_chain_clusters.clear();

    id_t num_root_chain_clusters = root_chain_array.size();
    if (world.rank() == 0 && opt.verbose) {
      spdlog::info("  Number of valid root chain clusters: {}",
                   num_root_chain_clusters);
    }
    // Each root chain cluster now has 2 valid children - the bottom root
    // chain cluster has two valid chain-cluster children, and all other
    // root-chain clusters have one root chain cluster and one other-chain
    // cluster Also include the very top root chain cluster as valid
    num_valid_clusters += 2 * num_root_chain_clusters + 1;

    if (world.rank() == 0) {
      spdlog::info("  Number of valid unflattened clusters: {}",
                   num_valid_clusters);
    }

    if (num_root_chain_clusters > 0) {
      sw_step.reset();

      root_chain_array.sort();
      world.barrier();

      // Get the min edge id for the root chain
      static id_t local_root_chain_min_edge_id;
      local_root_chain_min_edge_id = 0;
      if (world.rank() == 0) {
        root_chain_array.async_visit(
            0, []([[maybe_unused]] const id_t              &index,
                  std::pair<id_t, root_chain_cluster_info> &value) {
              local_root_chain_min_edge_id = value.first;
            });
      }
      world.barrier();
      MPI_Allreduce(&local_root_chain_min_edge_id, &root_chain_min_edge_id, 1,
                    mpi_id_type(), MPI_MAX, world.get_mpi_comm());

      // Get cluster size, stability, and stability_traversing_up for root
      // chain by processing each chunk of the root chain array locally and
      // then doing prefix sum. For this first pass, we assume that no
      // root-chain clusters have stability > sum of child stabilities
      // traversing up. We return the edges where that inequality holds for
      // further correction
      std::vector<std::pair<id_t, root_chain_cluster_info>>
          local_possible_clusters_for_selection;
      {
        local_possible_clusters_for_selection =
            calculate_root_chain_size_stability(root_chain_array,
                                                root_chain_second_child, world,
                                                root_chain_supernode);

        // Update the cluster map with new root chain info
        update_root_chain_cluster_info(root_chain_array,
                                       root_chain_cluster_map);
      }

      if (world.rank() == 0 && opt.verbose) {
        spdlog::info(
            "   Time to calculate root chain cluster sizes and "
            "stabilities (s): {:.3f}",
            sw_step);
      }
      sw_step.reset();

      // Create another ygm array of just possible to select clusters
      ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
          possible_clusters_for_selection_array(
              world, local_possible_clusters_for_selection);
      world.barrier();
      local_possible_clusters_for_selection.clear();

      if (possible_clusters_for_selection_array.size() > 0) {
        {
          std::stringstream ss;
          ss << "  Number of root chain clusters to possibly correct "
                "stability: "
             << possible_clusters_for_selection_array.size();
          if (world.rank() == 0 && opt.verbose) {
            spdlog::info(ss.str());
          }
        }

        // Sort the clusters we may need to correct
        possible_clusters_for_selection_array.sort();
        world.barrier();

        // Get the max cluster edge id
        static id_t max_cluster_edge_id;
        max_cluster_edge_id = 0;
        if (world.rank() == 0) {
          root_chain_array.async_visit(
              num_root_chain_clusters - 1,
              []([[maybe_unused]] const id_t              &index,
                 std::pair<id_t, root_chain_cluster_info> &value) {
                max_cluster_edge_id = value.first;
              });
        }
        world.barrier();
        MPI_Allreduce(MPI_IN_PLACE, &max_cluster_edge_id, 1, mpi_id_type(),
                      MPI_MAX, world.get_mpi_comm());

        auto correction_result = correct_root_chain_stability_traversing_up(
            possible_clusters_for_selection_array, world, max_cluster_edge_id);
        selected_root_chain_cluster_edge_id = correction_result.first;
        int num_correction_iterations       = correction_result.second;

        // Update the corrected root chain stability_traversing_up
        auto update_stability_traversing_up_lambda =
            []([[maybe_unused]] const id_t &cluster_edge_id,
               root_chain_cluster_info     &cluster_info,
               const float                  new_stability_traversing_up) {
              cluster_info.stability_traversing_up =
                  new_stability_traversing_up;
            };
        for (auto it = possible_clusters_for_selection_array.local_begin();
             it != possible_clusters_for_selection_array.local_end(); ++it) {
          root_chain_cluster_map.async_visit(
              it->value.first, update_stability_traversing_up_lambda,
              it->value.second.stability_traversing_up);
        }
        world.barrier();

        if (world.rank() == 0 && opt.verbose) {
          spdlog::info(
              "   Number of iterations to correct root chain "
              "stabilities: {}",
              num_correction_iterations);
          spdlog::info(
              "   Root chain cluster with edge id {} selected as a "
              "flat cluster",
              selected_root_chain_cluster_edge_id);
          spdlog::info(
              "   Time to correct root chain cluster stabilities "
              "(s): {:.3f}",
              sw_step);
        }
      }
    }

    // We no longer need the root chain array, just the root chain cluster map
    // at this point
    root_chain_array.clear();
    world.barrier();

    calculate_root_chain_size_stability_time += sw_round.elapsed().count();

    if (world.rank() == 0) {
      // if (selected_root_chain_cluster_edge_id > 0) {
      //   spdlog::info(
      //       "  Root chain cluster with edge id {} selected as a "
      //       "flat cluster",
      //       selected_root_chain_cluster_edge_id);
      // }
      if (opt.verbose) {
        spdlog::info("   Time to process root chain clusters (s): {:.3f}",
                     calculate_root_chain_size_stability_time);
        spdlog::info(" Selecting flat clustering");
      }
    }

    // Select flat clustering, a cluster is selected if its stability is
    // higher than the sum of its child stabilities
    std::vector<cluster_name_t> local_selected_clusters;
    {
      sw_step.reset();

      // Traverse down the child chains for cluster selection from all root
      // chain clusters above the selected one (if there is one)
      local_selected_clusters = traverse_down_hierarchy_and_select_clusters(
          root_chain_cluster_map, chain_map, leaf_cluster_map,
          root_chain_second_child, min_cluster_size, root_chain_supernode,
          selected_root_chain_cluster_edge_id);

      // If we selected a root chain cluster as a flat cluster, set it as
      // selected and add it to local selected clusters on rank 0
      if (world.rank() == 0 && selected_root_chain_cluster_edge_id > 0) {
        cluster_name_t cluster_name = std::make_pair(
            root_chain_supernode, selected_root_chain_cluster_edge_id);
        local_selected_clusters.push_back(cluster_name);
        root_chain_cluster_map.async_visit(
            selected_root_chain_cluster_edge_id,
            []([[maybe_unused]] const id_t &cluster_edge_id,
               root_chain_cluster_info     &cluster_info) {
              cluster_info.selected = true;
            });
      }
      world.barrier();
      select_clusters_time += sw_step.elapsed().count();

      id_t num_clusters       = local_selected_clusters.size();
      id_t total_num_clusters = ygm::sum(num_clusters, world);
      if (world.rank() == 0) {
        spdlog::info(" Number of flat clusters: {}", total_num_clusters);
        if (opt.verbose) {
          spdlog::info("   Time (s): {:.3f}", sw_step);
          spdlog::info(" Labeling points with consecutive cluster ID labels");
        }
      }
    }  // End select flat clustering

    // Assign consecutive IDs to clusters
    {
      sw_step.reset();

      cluster_id_t num_selected_clusters = local_selected_clusters.size();
      cluster_id_t start_cluster_id;

      // Prefix sum on the number of selected clusters to get consecutive
      // cluster IDs
      MPI_Scan(&num_selected_clusters, &start_cluster_id, 1, mpi_id_type(),
               MPI_SUM, world.get_mpi_comm());

      start_cluster_id = start_cluster_id - local_selected_clusters.size();

      // Assign points cluster IDs
      assign_points_cluster_ids(
          chain_map, leaf_cluster_map, root_chain_cluster_map,
          root_chain_cluster_edges_map, point_to_cluster_id_map,
          edge_endpoints_map, local_selected_clusters, start_cluster_id,
          root_chain_supernode, root_chain_second_child.name);

      label_points_time += sw_step.elapsed().count();

      if (world.rank() == 0) {
        if (opt.verbose) {
          spdlog::info("   Time (s): {:.3f}", sw_step);
        }
      }
    }

  }  // End phase 3

  if (world.rank() == 0) {
    spdlog::info(" Time for this phase (s): {:.3f}", sw_phase);
    if (opt.verbose) {
      spdlog::info(" Phase 3 time breakdown:");
      spdlog::info(
          "   Time to calculate non-root-chain cluster stabilities and "
          "sizes (s): {:.3f}",
          calculate_non_root_chain_size_stability_time);
      spdlog::info(
          "   Time to calculate root-chain cluster stabilities and "
          "sizes (s): {:.3f}",
          calculate_root_chain_size_stability_time);
      spdlog::info("   Time to select flat clusters (s): {:.3f}",
                   select_clusters_time);
      spdlog::info("   Time to assign cluster labels to points (s): {:.3f} ",
                   label_points_time);
    }
  }

  if (world.rank() == 0) {
    spdlog::info("Total time other than writing output files (s): {:.3f}",
                 sw_clustering);
  }

  // Write point labels to file
  {
    sw_step.reset();

    // Sort out file names for output
    std::string           output_dir;
    std::string           file_prefix           = "";
    std::filesystem::path no_extension_out_path = opt.cluster_ids_out_path;
    no_extension_out_path.replace_extension();
    if (no_extension_out_path.string() == opt.cluster_ids_out_path.string()) {
      output_dir = opt.cluster_ids_out_path.string();
    } else {
      output_dir  = opt.cluster_ids_out_path.parent_path().string();
      file_prefix = no_extension_out_path.filename().string();
    }

    if (world.rank() == 0) {
      if (opt.num_files_to_output == 1) {
        spdlog::info("Writing cluster labels to: {}",
                     opt.cluster_ids_out_path.string());
      } else {
        spdlog::info("Writing {} files containing cluster labels to: {}",
                     opt.num_files_to_output, output_dir);
      }
    }

    static std::string label_file_name;
    int                file_number = 0;
    if (opt.num_files_to_output == 1) {
      label_file_name = opt.cluster_ids_out_path.filename().string();
    } else {
      // Get file number if outputting multiple files
      if (opt.num_files_to_output < world.size()) {
        file_number = world.rank() % opt.num_files_to_output;
      } else {
        file_number = world.rank();
      }
      label_file_name = file_prefix + std::to_string(file_number) + ".txt";
    }

    // Output points and cluster labels
    ygm::io::multi_output mo_labels(world, output_dir);
    if (world.rank() == file_number) {
      mo_labels.async_write_line(label_file_name, "point_id\tcluster_id");
    }
    world.barrier();
    auto write_labels_lambda = [&mo_labels](const id_t         &point_id,
                                            const cluster_id_t &cluster_id) {
      std::stringstream ss;
      ss << point_id << "\t" << cluster_id;
      mo_labels.async_write_line(label_file_name, ss.str());
    };
    point_to_cluster_id_map.for_all(write_labels_lambda);
    world.barrier();

    if (world.rank() == 0) {
      spdlog::info("  Time to point labels (s): {:.3f}", sw_step);
    }
  }

  // Write cluster info to file
  if (!opt.cluster_data_out_path.empty()) {
    sw_step.reset();

    // Sort out file names for output
    std::string           output_dir;
    std::string           file_prefix           = "";
    std::filesystem::path no_extension_out_path = opt.cluster_data_out_path;
    no_extension_out_path.replace_extension();
    if (no_extension_out_path.string() == opt.cluster_data_out_path.string()) {
      output_dir = opt.cluster_data_out_path.string();
    } else {
      output_dir  = opt.cluster_data_out_path.parent_path().string();
      file_prefix = no_extension_out_path.filename().string();
    }

    if (world.rank() == 0) {
      if (opt.num_files_to_output == 1) {
        spdlog::info("Writing full cluster data to: {}",
                     opt.cluster_data_out_path.string());
      } else {
        spdlog::info("Writing {} files containing full cluster data to: {}",
                     opt.num_files_to_output, output_dir);
      }
    }

    std::string cluster_file_name;
    int         file_number = 0;
    if (opt.num_files_to_output == 1) {
      cluster_file_name = opt.cluster_data_out_path.filename().string();
    } else {
      // Get file number if outputting multiple files
      if (opt.num_files_to_output < world.size()) {
        file_number = world.rank() % opt.num_files_to_output;
      } else {
        file_number = world.rank();
      }
      cluster_file_name = file_prefix + std::to_string(file_number) + ".csv";
    }

    if ((min_cluster_size == 2) | (opt.output_invalid_clusters_also)) {
      write_all_clusters_to_file_including_invalid_clusters(
          world, output_dir, cluster_file_name, file_number,
          root_chain_supernode, root_chain_min_edge_id, root_chain_second_child,
          root_chain_cluster_map, chain_map, leaf_cluster_map);
    } else {
      // YGM map to collect info on valid clusters to write to file
      ygm::container::map<cluster_name_t, full_valid_cluster_info>
          valid_cluster_map(world);

      get_and_keep_valid_clusters_only(valid_cluster_map,
                                       root_chain_cluster_map, chain_map,
                                       leaf_cluster_map, root_chain_supernode);

      if (world.rank() == 0 && opt.verbose) {
        spdlog::info("  Time to filter and keep valid clusters only(s): {:.3f}",
                     sw_step);
      }
      sw_step.reset();

      get_valid_cluster_parent_child_relations(
          valid_cluster_map, chain_map, leaf_cluster_map, root_chain_supernode);

      if (world.rank() == 0 && opt.verbose) {
        spdlog::info(
            "  Time to extract valid clusters and their parent/child "
            "relationships (s): {:.3f}",
            sw_step);
      }
      sw_step.reset();

      write_valid_clusters_to_file(world, output_dir, cluster_file_name,
                                   file_number, valid_cluster_map);
    }

    if (world.rank() == 0) {
      spdlog::info("  Time to write cluster info (s): {:.3f}", sw_step);
    }
  }

  if (world.rank() == 0) {
    spdlog::info("Total time elapsed (s): {:.3f}", sw_clustering);
  }

  return EXIT_SUCCESS;
}