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

// Move the required information for alpha edges over from the edge contraction
// map so we can clear it and save space
void fill_alpha_edge_map(
    ygm::container::map<id_t, alpha_edge_info>       &alpha_edge_map,
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    const supernode_t                                &root_chain_supernode) {
  ygm::comm &comm                = alpha_edge_map.comm();
  auto       set_alpha_edge_info = [&alpha_edge_map, &root_chain_supernode](
                                 const id_t                  &edge_id,
                                 const edge_contraction_info &edge) {
    if (edge.contraction_round > 1) {
      supernode_t chain_supernode = root_chain_supernode;
      if (edge.chain_parent_edge_id > 0) {
        chain_supernode = edge.chain_supernode;
      }
      auto set_info_lambda = [](const id_t &edge_id, alpha_edge_info &edge,
                                const distance_t  &distance,
                                const supernode_t &chain_supernode) {
        edge.distance        = distance;
        edge.chain_supernode = chain_supernode;
      };
      alpha_edge_map.async_visit(edge_id, set_info_lambda, edge.distance,
                                 chain_supernode);
    }
  };
  edge_contraction_map.for_all(set_alpha_edge_info);
  comm.barrier();
}

void assign_edges_to_chains(
    ygm::container::map<id_t, edge_contraction_info> &edge_contraction_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>  &chain_map,
    std::vector<edge_id_with_dist_t>         &local_root_chain_alpha_edges,
    ygm::container::set<edge_id_with_dist_t> &root_chain_non_alpha_edge_set,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    const uint32_t                                            final_round) {
  ygm::comm &comm = edge_contraction_map.comm();

  static auto add_alpha_edge_to_chain_lambda =
      []([[maybe_unused]] const supernode_t &chain_name,
         std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
         const edge_id_with_dist_t &new_edge,
         const id_t                &chain_parent_edge_id) {
        full_cluster_info cluster_info{.min_edge = new_edge};
        cluster_info.edges.push_back(new_edge);
        chain.first.try_emplace(new_edge.first, cluster_info);
        chain.second.parent_edge_id = chain_parent_edge_id;
      };

  static auto add_non_alpha_edge_to_chain_lambda =
      []([[maybe_unused]] const supernode_t &chain_name,
         std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
         const edge_id_with_dist_t &new_edge) {
        chain.second.non_alpha_edges.push_back(new_edge);
      };

  static auto add_edge_to_leaf_cluster_lambda =
      []([[maybe_unused]] const supernode_t &cluster_supernode,
         full_leaf_cluster_info             &leaf_cluster_info,
         const edge_id_with_dist_t &new_edge, const id_t &parent_edge_id) {
        leaf_cluster_info.edges.push_back(new_edge);
        leaf_cluster_info.parent_edge_id = parent_edge_id;
      };

  // Match each edge to its chain
  auto assign_edge_to_chain_lambda =
      [&chain_map, &local_root_chain_alpha_edges,
       &root_chain_non_alpha_edge_set, &leaf_cluster_map,
       &final_round](const id_t &edge_id, edge_contraction_info &edge_info) {
        edge_id_with_dist_t edge = std::make_pair(edge_id, edge_info.distance);

        // If the edge is in the final MST, then it belongs to the root
        // chain as an alpha edge
        if (edge_info.contraction_round == final_round + 1) {
          local_root_chain_alpha_edges.push_back(edge);
        }

        // Otherwise, find what chain the edge belongs to
        else {
          // If we found a chain parent edge id, then the edge is assigned to a
          // known chain
          if (edge_info.chain_parent_edge_id > 0) {
            //  If this edge is not an alpha edge (we contracted in round 1), we
            //  either directly put it in a cluster, or add it to its
            //  non-alpha-edge vector to process later
            if (edge_info.contraction_round == 1) {
              // If this edge is in a chain with supernode formed in round 1, we
              // can directly add it to cluster map
              if (edge_info.chain_supernode.second == 1) {
                leaf_cluster_map.async_visit(
                    edge_info.chain_supernode, add_edge_to_leaf_cluster_lambda,
                    edge, edge_info.chain_parent_edge_id);
              } else {
                chain_map.async_visit(edge_info.chain_supernode,
                                      add_non_alpha_edge_to_chain_lambda, edge);
              }
            }
            // If the edge is an alpha edge, add it to the alpha-edge vector
            else {
              chain_map.async_visit(edge_info.chain_supernode,
                                    add_alpha_edge_to_chain_lambda, edge,
                                    edge_info.chain_parent_edge_id);
            }
          }
          // Otherwise, if we didn't find a chain to put the edge in, it belongs
          // to the root chain
          else {
            if (edge_info.contraction_round == 1) {
              root_chain_non_alpha_edge_set.async_insert(edge);
            } else {
              local_root_chain_alpha_edges.push_back(edge);
            }
          }
        }
      };
  edge_contraction_map.for_all(assign_edge_to_chain_lambda);
  comm.barrier();
}

void assign_non_alpha_edges_to_clusters(
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>> &chain_map) {
  ygm::comm &comm = chain_map.comm();

  // Assign non-alpha edges (that were contracted in round 1) to their clusters
  auto assign_non_alpha_edges_to_clusters_lambda =
      []([[maybe_unused]] const supernode_t &chain_name,
         std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain) {
        // If there's only one alpha edge, all non-alpha edges belong to
        // its cluster
        if (chain.first.size() == 1) {
          (chain.first.begin()->second)
              .edges.insert(
                  (chain.first.begin()->second).edges.end(),
                  std::make_move_iterator(chain.second.non_alpha_edges.begin()),
                  std::make_move_iterator(chain.second.non_alpha_edges.end()));
        } else {
          // Sort the non-alpha edges
          std::sort(chain.second.non_alpha_edges.begin(),
                    chain.second.non_alpha_edges.end());

          // For each non-alpha edge, find which alpha edges we fit
          // between and add it to that cluster. A non-alpha edge e belongs to
          // the cluster corresponding to the largest alpha edge smaller than e
          // Walk along both the alpha and non-alpha edges at the same time to
          // do this

          auto it = chain.first.begin();
          for (edge_id_with_dist_t &edge : chain.second.non_alpha_edges) {
            // Keep walking along the alpha edges until we find the largest one
            // that's smaller than edge
            while (it != chain.first.end() && it->first < edge.first) {
              ++it;
            }

            // // No alpha edge smaller than the non-alpha edge
            // if (it == chain.first.begin()) {
            //   continue;
            // }

            auto prev_it = std::prev(it);
            prev_it->second.edges.push_back(edge);
          }
        }

        // Clear the non-alpha edge vector when done
        chain.second.non_alpha_edges.clear();
      };

  chain_map.for_all(assign_non_alpha_edges_to_clusters_lambda);
  comm.barrier();
}

// Send the root chain non-alpha edges to the appropriate rank for
// processing
std::vector<edge_id_with_dist_t> split_root_chain_non_alpha_edges_to_process(
    ygm::comm &comm,
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
                                             &full_root_chain_cluster_array,
    ygm::container::set<edge_id_with_dist_t> &root_chain_non_alpha_edge_set) {
  std::vector<edge_id_with_dist_t> root_chain_non_alpha_edges;
  root_chain_non_alpha_edges.clear();

  int  num_ranks = comm.size();
  auto mpi_comm  = comm.get_mpi_comm();

  // Get the lower bound for each local array chunk
  id_t smallest_edge_id = 0;
  if (full_root_chain_cluster_array.local_size() > 0) {
    smallest_edge_id = full_root_chain_cluster_array.local_begin()->value.first;
  }
  std::vector<id_t> smallest_local_edge_ids(num_ranks, 0);

  MPI_Allgather(&smallest_edge_id, 1, mpi_id_type(),
                smallest_local_edge_ids.data(), 1, mpi_id_type(), mpi_comm);

  // The ith vector contains edges that lie between the min edge id on
  // rank i and min edge id on rank i+1 for root_chain_alpha_edge_array
  // These edges will be sent to rank i to assign to clusters
  // Split the edge id and distances into two vectors instead of a
  // vector of pairs for easy MPI sending/receiving
  std::vector<std::vector<id_t>>       local_edge_ids_by_rank(num_ranks);
  std::vector<std::vector<distance_t>> local_edge_distances_by_rank(num_ranks);

  auto group_local_edges_by_processing_rank_lambda =
      [&smallest_local_edge_ids, &local_edge_ids_by_rank,
       &local_edge_distances_by_rank](const edge_id_with_dist_t &edge) {
        int rank = 0;
        for (int i = 0; i < smallest_local_edge_ids.size(); ++i) {
          // If rank i has no array elements, move on
          if (0 == smallest_local_edge_ids.at(i)) {
            // pass
          } else if (smallest_local_edge_ids.at(i) < edge.first) {
            rank = i;
          } else {
            break;
          }
        }
        local_edge_ids_by_rank.at(rank).push_back(edge.first);
        local_edge_distances_by_rank.at(rank).push_back(edge.second);
      };
  root_chain_non_alpha_edge_set.for_all(
      group_local_edges_by_processing_rank_lambda);
  comm.barrier();

  // Send the root chain non-alpha edges to the appropriate rank for
  // processing

  // Each vector in local_edges_by_rank has variable size, so we need to
  // send counts first
  std::vector<int> send_counts(num_ranks, 0);
  for (int i = 0; i < num_ranks; ++i) {
    send_counts.at(i) = static_cast<int>(local_edge_ids_by_rank.at(i).size());
  }

  // All ranks receive the counts they will get from other ranks
  std::vector<int> recv_counts(num_ranks, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               mpi_comm);

  // Build displacements
  std::vector<int> send_displs(num_ranks, 0), recv_displs(num_ranks, 0);
  for (int i = 1; i < num_ranks; ++i) {
    send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
  }

  // Flatten send data (MPI_Alltoallv needs a flat data buffer)
  int total_send = std::accumulate(send_counts.begin(), send_counts.end(), 0);

  std::vector<id_t>       ids_send_buffer;
  std::vector<distance_t> distances_send_buffer;
  ids_send_buffer.reserve(total_send);
  distances_send_buffer.reserve(total_send);
  for (int i = 0; i < num_ranks; ++i) {
    ids_send_buffer.insert(ids_send_buffer.end(),
                           local_edge_ids_by_rank[i].begin(),
                           local_edge_ids_by_rank[i].end());
    distances_send_buffer.insert(distances_send_buffer.end(),
                                 local_edge_distances_by_rank[i].begin(),
                                 local_edge_distances_by_rank[i].end());
  }

  // Allocate receive buffers
  int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  std::vector<id_t>       received_ids;
  std::vector<distance_t> received_distances;
  received_ids.resize(total_recv);
  received_distances.resize(total_recv);

  // Exchange data
  MPI_Alltoallv(ids_send_buffer.data(), send_counts.data(), send_displs.data(),
                mpi_id_type(), received_ids.data(), recv_counts.data(),
                recv_displs.data(), mpi_id_type(), mpi_comm);
  MPI_Alltoallv(distances_send_buffer.data(), send_counts.data(),
                send_displs.data(), mpi_distance_type(),
                received_distances.data(), recv_counts.data(),
                recv_displs.data(), mpi_distance_type(), mpi_comm);

  // Put edges in pairs again in the final vector to process
  root_chain_non_alpha_edges.reserve(received_ids.size());
  for (int i = 0; i < received_ids.size(); ++i) {
    root_chain_non_alpha_edges.push_back(
        std::make_pair(received_ids.at(i), received_distances.at(i)));
  }

  return root_chain_non_alpha_edges;
}

// Assign root-chain non-alpha edges to their clusters
void assign_root_chain_non_alpha_edges_to_clusters(
    std::vector<edge_id_with_dist_t> &root_chain_non_alpha_edges,
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
        &full_root_chain_cluster_array,
    ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
        &root_chain_cluster_edges_map) {
  ygm::comm &comm = root_chain_cluster_edges_map.comm();

  auto add_single_edge_to_root_chain_cluster_lambda =
      []([[maybe_unused]] const id_t      &cluster_alpha_edge_id,
         std::vector<edge_id_with_dist_t> &edges,
         const edge_id_with_dist_t &new_edge) { edges.push_back(new_edge); };

  if (root_chain_non_alpha_edges.size() > 0) {
    // Sort the non-alpha edges on this rank
    std::sort(root_chain_non_alpha_edges.begin(),
              root_chain_non_alpha_edges.end());

    // For each non-alpha edge, find which alpha edges we fit
    // between and add it to that cluster. A non-alpha edge e belongs to
    // the cluster corresponding to the largest alpha edge smaller than e
    // Walk along both the alpha and non-alpha edge vectors at the
    // same time to do this
    auto alpha_edge_iter = full_root_chain_cluster_array.local_begin();
    id_t alpha_edge_id   = alpha_edge_iter->value.first;
    auto next_iter       = alpha_edge_iter;
    ++next_iter;
    for (id_t i = 0; i < root_chain_non_alpha_edges.size(); ++i) {
      edge_id_with_dist_t edge = root_chain_non_alpha_edges.at(i);

      // Keep walking along the alpha edges until we find the largest one
      // that's smaller than edge
      while (next_iter->value.first < edge.first) {
        // If we're at the end (largest) of the alpha edges, then the
        // alpha edge id will always be this last one
        if (next_iter == full_root_chain_cluster_array.local_end()) {
          break;
        }
        // Otherwise, advance to the next alpha edge until we find the
        // right cluster
        else {
          ++alpha_edge_iter;
          ++next_iter;
          alpha_edge_id = alpha_edge_iter->value.first;
        }
      }

      // Assign the non-alpha edge to its cluster
      root_chain_cluster_edges_map.async_visit(
          alpha_edge_id, add_single_edge_to_root_chain_cluster_lambda, edge);
    }
  }
  comm.barrier();

  // std::cout << "Rank " << comm.rank()
  //           << " done assigning root chain non-alpha edges"
  //           << std::endl; // debug
}

// Process the clusters in the chain map and and fill in missing
// information on birth distance, parent, children
void fill_missing_non_root_chain_cluster_info(
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>> &chain_map,
    ygm::container::map<id_t, alpha_edge_info>      &alpha_edge_map) {
  ygm::comm &comm          = chain_map.comm();
  auto       chain_map_ptr = chain_map.get_ygm_ptr();

  // Go through each non-root chain and for each cluster fill in the birth
  // distance
  auto set_birth_distance_lambda =
      [](const supernode_t &chain_name,
         std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain) {
        // The ith cluster has birth distance equal to the min edge distance of
        // the (i+1)th cluster
        for (auto it = chain.first.begin(); it != chain.first.end(); ++it) {
          auto next_it = std::next(it);
          if (next_it == chain.first.end()) {
            break;  // no next element
          }

          it->second.birth_distance = next_it->second.min_edge.second;
        }
      };

  chain_map.for_all(set_birth_distance_lambda);
  comm.barrier();

  // Get the parent chain and birth distance for the top cluster for each
  // (non-root) chain
  auto get_parent_chain_lambda =
      [&alpha_edge_map, chain_map_ptr](
          const supernode_t &chain_name,
          std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
              &chain) {
        id_t parent_edge_id = chain.second.parent_edge_id;
        // Visit the parent edge in the alpha edge map and get its chain
        auto visit_alpha_edge_lambda = [](const id_t            &edge_id,
                                          const alpha_edge_info &edge_info,
                                          const supernode_t     &chain_name,
                                          auto chain_map_ptr) {
          auto set_parent_chain_and_top_birth_dist =
              [](const supernode_t &chain_name,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                                   &chain,
                 const supernode_t &parent_chain,
                 const distance_t  &parent_edge_distance) {
                // Set the chain parent
                chain.second.parent_chain = parent_chain;

                // Set the birth distance for the top cluster in the chain
                auto it                   = chain.first.rbegin();
                it->second.birth_distance = parent_edge_distance;
              };
          chain_map_ptr->async_visit(
              chain_name, set_parent_chain_and_top_birth_dist,
              edge_info.chain_supernode, edge_info.distance);
        };
        alpha_edge_map.async_visit(parent_edge_id, visit_alpha_edge_lambda,
                                   chain_name, chain_map_ptr);
      };
  chain_map.for_all(get_parent_chain_lambda);
  comm.barrier();

  // Fill in the missing child clusters for all chains
  auto get_child_clusters_lambda = [&alpha_edge_map, chain_map_ptr](
                                       const supernode_t &chain_name,
                                       std::pair<
                                           std::map<id_t, full_cluster_info>,
                                           full_chain_info> &chain) {
    // The lowest cluster in the chain has two children
    auto get_bottom_children_lambda = []([[maybe_unused]] const id_t &edge_id,
                                         const alpha_edge_info       &edge_info,
                                         const supernode_t &chain_name,
                                         auto               chain_map_ptr) {
      if (edge_info.dendrogram_children[0] == BLANK_SUPERNODE ||
          edge_info.dendrogram_children[1] == BLANK_SUPERNODE) {
        std::cout << "Warning! Visiting edge " << edge_id
                  << "at the bottom of a chain. It doesn't have 2 "
                     "dendrogram children and instead has children: "
                  << edge_info.dendrogram_children[0] << " and "
                  << edge_info.dendrogram_children[1] << std::endl;  // debug
      }

      auto set_bottom_children_lambda =
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                                              &chain,
             const std::array<supernode_t, 2> &children) {
            chain.second.children[0] = children[0];
            chain.second.children[1] = children[1];
          };

      chain_map_ptr->async_visit(chain_name, set_bottom_children_lambda,
                                 edge_info.dendrogram_children);
    };

    // All other clusters have one non-chain child
    auto get_child_lambda = [](const id_t            &edge_id,
                               const alpha_edge_info &edge_info,
                               const supernode_t     &chain_name,
                               auto                   chain_map_ptr) {
      if (edge_info.dendrogram_children[0] == BLANK_SUPERNODE &&
          edge_info.dendrogram_children[1] == BLANK_SUPERNODE) {
        std::cout
            << "Warning! Visiting edge " << edge_id
            << "in a chain, but it doesn't have have any dendrogram children."
            << std::endl;  // debug
      }

      auto set_child_lambda =
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                        &chain,
             const id_t &cluster_edge_id, const supernode_t &child) {
            chain.first[cluster_edge_id].child = child;
          };

      if (edge_info.dendrogram_children[0] != BLANK_SUPERNODE) {
        chain_map_ptr->async_visit(chain_name, set_child_lambda, edge_id,
                                   edge_info.dendrogram_children[0]);
      } else {
        chain_map_ptr->async_visit(chain_name, set_child_lambda, edge_id,
                                   edge_info.dendrogram_children[1]);
      }
    };

    for (auto it = chain.first.begin(); it != chain.first.end(); ++it) {
      // The lowest cluster in the chain has two children
      if (it == chain.first.begin()) {
        alpha_edge_map.async_visit(it->first, get_bottom_children_lambda,
                                   chain_name, chain_map_ptr);
      }
      // All other clusters have one non-chain child
      else {
        alpha_edge_map.async_visit(it->first, get_child_lambda, chain_name,
                                   chain_map_ptr);
      }
    }
  };
  chain_map.for_all(get_child_clusters_lambda);
  comm.barrier();
}

// Process the leaf cluster map and fill in missing
// information on birth distance and parent
void fill_missing_leaf_cluster_info(
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    ygm::container::map<id_t, alpha_edge_info>               &alpha_edge_map) {
  ygm::comm &comm                 = leaf_cluster_map.comm();
  auto       leaf_cluster_map_ptr = leaf_cluster_map.get_ygm_ptr();

  // Get the parent chain and birth distance
  auto get_parent_chain_lambda = [&alpha_edge_map, leaf_cluster_map_ptr](
                                     const supernode_t      &cluster_name,
                                     full_leaf_cluster_info &cluster_info) {
    // Visit the parent edge in the alpha edge map and get its chain
    auto visit_alpha_edge_lambda =
        [](const id_t &edge_id, const alpha_edge_info &edge_info,
           const supernode_t &cluster_name, auto leaf_cluster_map_ptr) {
          auto set_parent_chain_and_top_birth_dist =
              [](const supernode_t      &cluster_name,
                 full_leaf_cluster_info &cluster_info,
                 const supernode_t      &parent_chain,
                 const distance_t       &parent_edge_distance) {
                // Set the chain parent
                cluster_info.parent_chain = parent_chain;

                // Set the birth distance for the last cluster in the chain
                cluster_info.birth_distance = parent_edge_distance;
              };
          leaf_cluster_map_ptr->async_visit(
              cluster_name, set_parent_chain_and_top_birth_dist,
              edge_info.chain_supernode, edge_info.distance);
        };
    alpha_edge_map.async_visit(cluster_info.parent_edge_id,
                               visit_alpha_edge_lambda, cluster_name,
                               leaf_cluster_map_ptr);
  };
  leaf_cluster_map.for_all(get_parent_chain_lambda);
  comm.barrier();
}

// Split the root chain into clusters at its alpha edges
// Return root_chain_second_child_supernode
supernode_t fill_missing_root_chain_cluster_info(
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
                                               &full_root_chain_cluster_array,
    ygm::container::map<id_t, alpha_edge_info> &alpha_edge_map,
    ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
        &root_chain_cluster_edges_map) {
  static supernode_t root_chain_second_child_supernode;
  root_chain_second_child_supernode = BLANK_SUPERNODE;

  ygm::comm &comm = alpha_edge_map.comm();
  auto       full_root_chain_cluster_array_ptr =
      full_root_chain_cluster_array.get_ygm_ptr();

  /* Get the missing parent edge id and lambda birth by looking at next
   * highest cluster*/

  id_t array_size = full_root_chain_cluster_array.size();

  // Iterate through our local root-chunk alpha edge chunk and fill in
  // cluster information
  if (full_root_chain_cluster_array.local_size() > 0) {
    id_t first_local_index = full_root_chain_cluster_array.local_begin()->index;
    id_t last_local_index =
        first_local_index + full_root_chain_cluster_array.local_size() - 1;

    for (auto it = full_root_chain_cluster_array.local_begin();
         it != full_root_chain_cluster_array.local_end(); ++it) {
      // If we're not at the top of this local chunk of array, peek at the
      // next item to get the parent info
      if (it->index != last_local_index) {
        auto temp = it;
        ++temp;
        it->value.second.parent_edge_id = temp->value.first;
        it->value.second.lambda_birth   = temp->value.second.lambda_min_edge;
      }
    }

    // If we're not the top of the root chain, the top local array
    // item can visit the item above on another rank to get its parent
    // info
    if (last_local_index < array_size - 1) {
      auto visit_above_lambda =
          []([[maybe_unused]] const id_t &above_index,
             const std::pair<id_t, root_chain_cluster_info>
                 &above_cluster_edge_and_info,
             auto full_root_chain_cluster_array_ptr) {
            full_root_chain_cluster_array_ptr->async_visit(
                above_index - 1,
                []([[maybe_unused]] const id_t &below_index,
                   std::pair<id_t, root_chain_cluster_info>
                              &below_cluster_edge_and_info,
                   const id_t &parent_edge_id, const distance_t &lambda_birth) {
                  below_cluster_edge_and_info.second.parent_edge_id =
                      parent_edge_id;
                  below_cluster_edge_and_info.second.lambda_birth =
                      lambda_birth;
                },
                above_cluster_edge_and_info.first,
                above_cluster_edge_and_info.second.lambda_min_edge);
          };
      full_root_chain_cluster_array_ptr->async_visit(
          last_local_index + 1, visit_above_lambda,
          full_root_chain_cluster_array_ptr);
    }
  }
  comm.barrier();

  /* Get the num points added and sum lambda edges added from the
   * root_chain_cluster_edges_map */

  auto set_num_points_and_sum_lambda_edge_added =
      [&root_chain_cluster_edges_map, full_root_chain_cluster_array_ptr](
          const id_t &index, std::pair<id_t, root_chain_cluster_info> &value) {
        auto visit_cluster_edge_map =
            [](const id_t                             &cluster_edge_id,
               const std::vector<edge_id_with_dist_t> &edges,
               const id_t                             &array_index,
               ygm::ygm_ptr<ygm::container::array<
                   std::pair<id_t, root_chain_cluster_info>>>
                   full_root_chain_cluster_array_ptr) {
              // Calculate num points added and sum lambda edges added for this
              // cluster
              id_t       num_points_added       = edges.size() - 1;
              distance_t sum_lambda_edges_added = 0.0;
              for (auto &edge : edges) {
                sum_lambda_edges_added += lambda_from_dist(edge.second);
              }

              // Set the values in the cluster array
              full_root_chain_cluster_array_ptr->async_visit(
                  array_index,
                  [](const id_t                               &index,
                     std::pair<id_t, root_chain_cluster_info> &value,
                     const id_t                               &num_points_added,
                     const distance_t &sum_lambda_edges_added) {
                    value.second.num_points_added = num_points_added;
                    value.second.sum_lambda_edges_added =
                        sum_lambda_edges_added;
                  },
                  num_points_added, sum_lambda_edges_added);
            };

        root_chain_cluster_edges_map.async_visit(
            value.first, visit_cluster_edge_map, index,
            full_root_chain_cluster_array_ptr);
      };

  full_root_chain_cluster_array.for_all(
      set_num_points_and_sum_lambda_edge_added);
  comm.barrier();

  /* Fill in non-root-chain children */
  auto set_non_root_chain_child_lambda =
      [&alpha_edge_map, full_root_chain_cluster_array_ptr](
          [[maybe_unused]] const id_t              &index,
          std::pair<id_t, root_chain_cluster_info> &value) {
        auto visit_alpha_edge_map =
            [](const id_t &edge_id, const alpha_edge_info &edge_info,
               const id_t &array_index,
               ygm::ygm_ptr<ygm::container::array<
                   std::pair<id_t, root_chain_cluster_info>>>
                   full_root_chain_cluster_array_ptr) {
              // All root chain alpha edges should have at least one dendrogram
              // child
              if (edge_info.dendrogram_children[0] == BLANK_SUPERNODE) {
                std::cout << "Warning! Edge " << edge_id
                          << ", which is an alpha edge in the root chain, has "
                             "no dendrogram child in position 1. In the second "
                             "position is : "
                          << edge_info.dendrogram_children[1] << std::endl;
              }

              // Other than the bottom alpha edge, no root chain alpha edges
              // should have 2 dendrogram children
              if (edge_info.dendrogram_children[0] != BLANK_SUPERNODE &&
                  edge_info.dendrogram_children[1] != BLANK_SUPERNODE) {
                if (array_index == 0) {
                  root_chain_second_child_supernode =
                      edge_info.dendrogram_children[1];
                  // std::cout << "Setting 2nd root chain child supernode to: "
                  //           << edge_info.dendrogram_children[1]
                  //           << std::endl;  // debug
                } else {
                  std::cout << "Warning! Edge " << edge_id
                            << ", which is an alpha edge in the root chain "
                               "that's not at the bottom, has "
                               "two dendrogram children: "
                            << edge_info.dendrogram_children[0] << " and "
                            << edge_info.dendrogram_children[1]
                            << std::endl;  // debug
                }
              }

              // Get this alpha edge's child
              supernode_t child = edge_info.dendrogram_children[0];

              // Set the child in the root chain cluster array
              full_root_chain_cluster_array_ptr->async_visit(
                  array_index,
                  [](const id_t                               &index,
                     std::pair<id_t, root_chain_cluster_info> &value,
                     const supernode_t &child) { value.second.child = child; },
                  child);
            };

        alpha_edge_map.async_visit(value.first, visit_alpha_edge_map, index,
                                   full_root_chain_cluster_array_ptr);
      };

  full_root_chain_cluster_array.for_all(set_non_root_chain_child_lambda);
  comm.barrier();

  return root_chain_second_child_supernode;
}

}  // namespace clams::clustering