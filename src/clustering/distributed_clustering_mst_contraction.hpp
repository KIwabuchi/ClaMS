#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <cereal/types/array.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility/timer.hpp>

#include "distributed_clustering_types.hpp"

namespace clams::clustering {

/*
 * @brief Read mst edges to ygm edge maps.
 *
 * @param mst_file_vector Vector of files (as strings) containing mst edges to
 * read. Files must be white space separated with each line of the form
 * <edge id><node1><node2><distance>
 * @param edge_endpoints_map An empty YGM map of edge id -> edge_info
 * @param edge_contraction_map An empty YGM map of edge id ->
 * edge_contraction_info
 */
void read_mst_edges_with_id_into_edge_maps(
    std::vector<std::string>                         &mst_file_vector,
    ygm::container::map<id_t, std::pair<id_t, id_t>> &edge_endpoints_map,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map) {
  ygm::comm &comm = edge_endpoints_map.comm();

  ygm::io::line_parser mst_line_parser(comm, mst_file_vector);

  auto line_parser_lambda = [&edge_endpoints_map,
                             &edge_contraction_map](const std::string &line) {
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
  comm.barrier();
}

/*
 * @brief Read mst edges to ygm edge maps.
 *
 * @param mst_file_vector Vector of files (as strings) containing mst edges to
 * read. Files must be white space separated with each line of the form
 * <node1><node2><distance>
 * @param edge_endpoints_map An empty YGM map of edge id -> edge_info
 * @param edge_contraction_map An empty YGM map of edge id ->
 * edge_contraction_info
 */
std::array<float, 3> read_mst_edges_into_edge_maps(
    std::vector<std::string>                         &mst_file_vector,
    ygm::container::map<id_t, std::pair<id_t, id_t>> &edge_endpoints_map,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map) {
  ygm::utility::timer timer{};
  timer.reset();

  ygm::comm &comm = edge_endpoints_map.comm();

  ygm::io::line_parser mst_line_parser(comm, mst_file_vector);

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
  comm.barrier();

  float read_time = timer.elapsed();
  timer.reset();

  ygm::container::array<std::pair<distance_t, std::pair<id_t, id_t>>>
      edge_array(comm, mst_edge_vector);
  comm.barrier();

  edge_array.sort();
  comm.barrier();
  float sort_time = timer.elapsed();
  timer.reset();

  auto visit_array_lambda =
      [&edge_endpoints_map, &edge_contraction_map](
          const std::size_t                                  &index,
          const std::pair<distance_t, std::pair<id_t, id_t>> &value) {
        // Add this edge to the edge map with its sorted position as its id
        distance_t            distance       = value.first;
        std::pair<id_t, id_t> edge_endpoints = value.second;
        edge_endpoints_map.async_insert(index, edge_endpoints);
        edge_contraction_map.async_insert(
            index,
            edge_contraction_info{.endpoint_supernode_reps = edge_endpoints,
                                  .distance                = distance});
      };
  edge_array.for_all(visit_array_lambda);
  comm.barrier();

  float add_time = timer.elapsed();

  std::array<float, 3> time_arr{read_time, sort_time, add_time};
  return time_arr;
}

void fill_init_min_incident_edge_map(
    ygm::container::map<id_t, id_t>                  &min_incident_edge_map,
    ygm::container::map<id_t, std::pair<id_t, id_t>> &edge_endpoints_map) {
  ygm::comm &comm = min_incident_edge_map.comm();

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  // Fill the initial incidence map from the original MST edges
  auto fill_init_min_incident_edge_map_supernode_reps =
      [&min_incident_edge_map]([[maybe_unused]] const id_t &edge_id,
                               std::pair<id_t, id_t>       &edge_endpoints) {
        min_incident_edge_map.async_insert(edge_endpoints.first,
                                           std::numeric_limits<id_t>::max());
        min_incident_edge_map.async_insert(edge_endpoints.second,
                                           std::numeric_limits<id_t>::max());
      };
  edge_endpoints_map.for_all(fill_init_min_incident_edge_map_supernode_reps);
  comm.barrier();

  // Fill the initial incidence map from the original MST edges
  auto fill_init_min_incident_edge_map =
      [&min_incident_edge_map](const id_t            &edge_id,
                               std::pair<id_t, id_t> &edge_endpoints) {
        min_incident_edge_map.async_visit(edge_endpoints.first,
                                          check_incident_edge_lambda, edge_id);
        min_incident_edge_map.async_visit(edge_endpoints.second,
                                          check_incident_edge_lambda, edge_id);
      };
  edge_endpoints_map.for_all(fill_init_min_incident_edge_map);
  comm.barrier();
}

// Find all edges to contract and add to disjoint set
void contract_edges(
    uint32_t _round, ygm::container::map<id_t, id_t> &min_incident_edge_map,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::ygm_ptr<ygm::container::map<id_t, alpha_edge_info>> alpha_edge_map_ptr,
    ygm::container::disjoint_set<id_t> &tree_components_djset) {
  ygm::comm &comm                      = edge_contraction_map.comm();
  auto       tree_components_djset_ptr = tree_components_djset.get_ygm_ptr();

  static uint32_t round;
  round = _round;

  [[maybe_unused]] static auto add_dendrogram_child_lambda =
      []([[maybe_unused]] const id_t &edge_id, alpha_edge_info &info,
         supernode_t new_child) {
        if (info.dendrogram_children[0] == BLANK_SUPERNODE) {
          info.dendrogram_children[0] = new_child;
        } else {
          info.dendrogram_children[1] = new_child;
        }
      };

  auto contract_edges_lambda = [&edge_contraction_map, alpha_edge_map_ptr,
                                tree_components_djset_ptr](
                                   [[maybe_unused]] const id_t &supernode_rep,
                                   const id_t min_incident_edge_id) {
    // The shortest edge incident to each supernode is contracted

    // Add the contracted edge to the disjoint set and set as
    // contracted For an edge between supernodes (rep1, round-1) and
    // (rep2, round-1), we add the rep1-rep2 edge to the disjoint set
    auto add_edge_to_djset_lambda = []([[maybe_unused]] const id_t &edge_id,
                                       edge_contraction_info       &edge_info,
                                       auto tree_components_djset_ptr) {
      edge_info.contraction_round = round;
      tree_components_djset_ptr->async_union(
          edge_info.endpoint_supernode_reps.first,
          edge_info.endpoint_supernode_reps.second);
    };
    edge_contraction_map.async_visit(min_incident_edge_id,
                                     add_edge_to_djset_lambda,
                                     tree_components_djset_ptr);

    // Set this supernode as the dendrogram child of the min incident edge
    if (round >= 2) {
      alpha_edge_map_ptr->async_visit(min_incident_edge_id,
                                      add_dendrogram_child_lambda,
                                      std::make_pair(supernode_rep, round - 1));
    }
  };
  min_incident_edge_map.for_all(contract_edges_lambda);
  comm.barrier();

  return;
}

void update_edge_chain_parent_edge_id(
    uint32_t                                          round,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::map<id_t, id_t>                  &min_incident_edge_map) {
  ygm::comm &comm                     = edge_contraction_map.comm();
  auto       edge_contraction_map_ptr = edge_contraction_map.get_ygm_ptr();

  auto update_chain_parents_lambda =
      [&min_incident_edge_map, round, edge_contraction_map_ptr](
          const id_t &edge_id, edge_contraction_info &edge_info) {
        // Only see if we can update an edge's chain parent edge id if not
        // assigned yet (if its 0) and if the edge is already contracted
        if (edge_info.chain_parent_edge_id == 0 &&
            0 < edge_info.contraction_round &&
            edge_info.contraction_round < round) {
          min_incident_edge_map.async_visit(
              edge_info.chain_supernode.first,
              []([[maybe_unused]] const id_t &supernode_rep,
                 const id_t &min_incident_edge_id, const id_t &edge_id,
                 auto edge_contraction_map_ptr) {
                // If the supernode's min incident edge id is larger than edge
                // id, we found the chain's parent edge id in the dendrogram
                if (min_incident_edge_id > edge_id) {
                  edge_contraction_map_ptr->async_visit(
                      edge_id,
                      []([[maybe_unused]] const id_t &edge_id,
                         edge_contraction_info       &edge_info,
                         id_t                         chain_parent_edge_id) {
                        edge_info.chain_parent_edge_id = chain_parent_edge_id;
                      },
                      min_incident_edge_id);
                }
              },
              edge_id, edge_contraction_map_ptr);
        }
      };
  edge_contraction_map.for_all(update_chain_parents_lambda);
  comm.barrier();

  return;
}

void update_edge_chain_parent_edge_id_from_local_map(
    uint32_t                                          round,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::map<id_t, id_t>                  &min_incident_edge_map) {
  ygm::comm &comm = edge_contraction_map.comm();

  // Give all ranks a full copy of min_incident_edge_map
  std::map<id_t, id_t> local_min_incident_edge_map;
  min_incident_edge_map.for_all(
      [&local_min_incident_edge_map](const id_t &supernode_rep,
                                     const id_t &min_incident_edge_id) {
        local_min_incident_edge_map[supernode_rep] = min_incident_edge_id;
      });
  comm.barrier();

  static std::map<id_t, id_t> min_incident_edge_std_map;
  min_incident_edge_std_map.clear();
  comm.async_bcast(
      [](const std::map<id_t, id_t> &local_map) {
        min_incident_edge_std_map.insert(local_map.begin(), local_map.end());
      },
      local_min_incident_edge_map);
  comm.barrier();
  local_min_incident_edge_map.clear();

  // Update the chain parents of edges using local lookup
  auto update_chain_parents_lambda = [&round](
                                         [[maybe_unused]] const id_t &edge_id,
                                         edge_contraction_info &edge_info) {
    // Only see if we can update an edge's chain parent edge id if not
    // assign yet (if its 0) and if the edge is already contracted
    if (edge_info.chain_parent_edge_id == 0 &&
        0 < edge_info.contraction_round &&
        edge_info.contraction_round < round) {
      if (min_incident_edge_std_map.at(edge_info.chain_supernode.first) >
          edge_id) {
        edge_info.chain_parent_edge_id =
            min_incident_edge_std_map.at(edge_info.chain_supernode.first);
      }
    }
  };
  edge_contraction_map.for_all(update_chain_parents_lambda);
  comm.barrier();

  return;
}

void reinit_min_incident_edge_map(
    ygm::container::map<id_t, id_t>    &min_incident_edge_map,
    ygm::container::disjoint_set<id_t> &tree_components_djset) {
  ygm::comm &comm = min_incident_edge_map.comm();

  min_incident_edge_map.clear();
  comm.barrier();

  auto reinit_min_incident_edge_map_lambda =
      [&min_incident_edge_map]([[maybe_unused]] const id_t &old_supernode_rep,
                               const id_t                  &new_supernode_rep) {
        min_incident_edge_map.async_insert(new_supernode_rep,
                                           std::numeric_limits<id_t>::max());
      };
  tree_components_djset.for_all(reinit_min_incident_edge_map_lambda);
  comm.barrier();

  return;
}

void update_edge_endpoints_and_chain_supernode(
    uint32_t                                          _round,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::disjoint_set<id_t>               &tree_components_djset) {
  static uint32_t round;
  round = _round;

  ygm::comm &comm                     = edge_contraction_map.comm();
  auto       edge_contraction_map_ptr = edge_contraction_map.get_ygm_ptr();

  auto update_edge_endpoints_and_chain_supernode_lambda =
      [&tree_components_djset, edge_contraction_map_ptr](
          const id_t &edge_id, edge_contraction_info &edge_info) {
        // If the edge is not contracted yet, update its supernode endpoints
        if (edge_info.contraction_round == 0) {
          // Update the first supernode endpoint of this edge
          tree_components_djset.async_visit(
              edge_info.endpoint_supernode_reps.first,
              []([[maybe_unused]] const auto &djset_item_data_pair,
                 const id_t &edge_id, auto edge_contraction_map_ptr) {
                edge_contraction_map_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t new_first_endpoint_supernode_rep) {
                      edge_info.endpoint_supernode_reps.first =
                          new_first_endpoint_supernode_rep;
                    },
                    djset_item_data_pair.second.get_parent());
              },
              edge_id, edge_contraction_map_ptr);

          // Update the second supernode endpoint of this edge
          tree_components_djset.async_visit(
              edge_info.endpoint_supernode_reps.second,
              []([[maybe_unused]] const auto &djset_item_data_pair,
                 const id_t &edge_id, auto edge_contraction_map_ptr) {
                edge_contraction_map_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t new_second_endpoint_supernode_rep) {
                      edge_info.endpoint_supernode_reps.second =
                          new_second_endpoint_supernode_rep;
                    },
                    djset_item_data_pair.second.get_parent());
              },
              edge_id, edge_contraction_map_ptr);
        }

        // If an edge is already contracted, see if it already has a known chain
        // parent edge id, or if we should update the chain supernode to find
        // the chain this edge belongs to
        else {
          // If we haven't found a chain_parent_edge_id > edge_id in the
          // dendrogram yet, then update the chain supernode
          if (edge_info.chain_parent_edge_id == 0) {
            // If the edge was contracted this round, then get its first
            // supernode from one of its endpoints
            id_t supernode_rep_to_visit =
                edge_info.endpoint_supernode_reps.first;
            // Otherwise, get the parent supernode of its last supernode
            if (edge_info.contraction_round < round) {
              supernode_rep_to_visit = edge_info.chain_supernode.first;
            }

            // Update the edge's chain supernode
            tree_components_djset.async_visit(
                supernode_rep_to_visit,
                []([[maybe_unused]] const auto &djset_item_data_pair,
                   const id_t edge_id, auto edge_contraction_map_ptr) {
                  edge_contraction_map_ptr->async_visit(
                      edge_id,
                      []([[maybe_unused]] const id_t &edge_id,
                         edge_contraction_info       &edge_info,
                         const supernode_t            new_supernode) {
                        edge_info.chain_supernode = new_supernode;
                      },
                      std::make_pair(djset_item_data_pair.second.get_parent(),
                                     round));
                },
                edge_id, edge_contraction_map_ptr);
          }
        }
      };
  edge_contraction_map.for_all(
      update_edge_endpoints_and_chain_supernode_lambda);
  comm.barrier();

  return;
}

void update_edge_endpoints_and_chain_supernode_from_local_map(
    uint32_t                                          round,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::disjoint_set<id_t>               &tree_components_djset) {
  ygm::comm &comm = edge_contraction_map.comm();

  static std::map<id_t, id_t> old_to_new_supernode_rep_map;
  static std::map<id_t, id_t> local_old_to_new_supernode_rep_map;
  old_to_new_supernode_rep_map.clear();
  local_old_to_new_supernode_rep_map.clear();

  // Create a local copy of the full tree_components djset on every rank
  tree_components_djset.for_all([](const id_t &old_supernode_rep,
                                   const id_t &new_supernode_rep) {
    local_old_to_new_supernode_rep_map[old_supernode_rep] = new_supernode_rep;
  });
  comm.barrier();
  comm.async_bcast(
      [](const std::map<id_t, id_t> &local_map) {
        old_to_new_supernode_rep_map.insert(local_map.begin(), local_map.end());
      },
      local_old_to_new_supernode_rep_map);
  comm.barrier();

  // Update edge info by referencing the contents of the local map
  auto update_edge_endpoints_and_chain_supernode_lambda =
      [round]([[maybe_unused]] const id_t &edge_id,
              edge_contraction_info       &edge_info) {
        // If the edge is not contracted yet, update its supernode
        // endpoints
        if (edge_info.contraction_round == 0) {
          // Update the first supernode endpoint of this edge
          edge_info.endpoint_supernode_reps.first =
              old_to_new_supernode_rep_map.at(
                  edge_info.endpoint_supernode_reps.first);

          // Update the second supernode endpoint of this edge
          edge_info.endpoint_supernode_reps.second =
              old_to_new_supernode_rep_map.at(
                  edge_info.endpoint_supernode_reps.second);
        }
        // If an edge is already contracted, see if it already has a known chain
        // parent edge id, or if we should update the chain supernode to find
        // the chain this edge belongs to
        else {
          // If we haven't found a chain_parent_edge_id > edge_id in the
          // dendrogram yet, then update the chain supernode
          if (edge_info.chain_parent_edge_id == 0) {
            // If the edge was contracted this round, then get its first
            // supernode from one of its endpoints
            id_t old_supernode_rep = edge_info.endpoint_supernode_reps.first;
            // Otherwise, get the parent supernode of its last supernode
            if (edge_info.contraction_round < round) {
              old_supernode_rep = edge_info.chain_supernode.first;
            }

            // Update the edge's chain supernode
            supernode_t new_supernode = std::make_pair(
                old_to_new_supernode_rep_map.at(old_supernode_rep), round);
            edge_info.chain_supernode = new_supernode;
          }
        }
      };
  edge_contraction_map.for_all(
      update_edge_endpoints_and_chain_supernode_lambda);
  comm.barrier();

  return;
}

// Note - assumes that we've already filled the min incident edge map with
// supernode -> max id_t
void update_min_incident_edge_map(
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::map<id_t, id_t>                  &min_incident_edge_map) {
  ygm::comm &comm = edge_contraction_map.comm();

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  auto update_min_incident_edge_map_lambda =
      [&min_incident_edge_map](const id_t            &edge_id,
                               edge_contraction_info &edge_info) {
        // If the edge is not contracted yet
        if (edge_info.contraction_round == 0) {
          min_incident_edge_map.async_visit(
              edge_info.endpoint_supernode_reps.first,
              check_incident_edge_lambda, edge_id);
          min_incident_edge_map.async_visit(
              edge_info.endpoint_supernode_reps.second,
              check_incident_edge_lambda, edge_id);
        }
      };
  edge_contraction_map.for_all(update_min_incident_edge_map_lambda);
  comm.barrier();
}

}  // namespace clams::clustering