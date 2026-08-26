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

/**
 * @brief Calculate cluster size, stability, and stability traversing up
 * from its child cluster information for a chain once all children have been
 * received.
 *
 * @param chain_cluster_map Map of cluster edge id -> cluster info for this
 * chain.
 * @param chain_info full_chain_info for this chain.
 * @param min_cluster_size HDBSCAN min cluster size parameter.
 * @param chain_map_ptr Pointer for YGM map of chain name -> (cluster map, chain
 * info)
 * @param leaf_cluster_map_ptr Pointer for YGM map of leaf cluster name
 * (supernode) -> leaf cluster info.
 * @param chain_name Name of this chain. Used only for debugging printouts.
 *
 * @return The number of valid clusters found when processing this chain. If a
 * chain cluster has two children with size at least min_cluster_size, we say
 * both those children are valid, set them as such, and increase the number of
 * valid clusters found by 2.
 */
std::size_t calculate_chain_cluster_size_stability_from_children(
    std::map<id_t, full_cluster_info> &chain_cluster_map,
    full_chain_info &chain_info, uint32_t min_cluster_size,
    ygm::ygm_ptr<ygm::container::map<
        supernode_t,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
        chain_map_ptr,
    ygm::ygm_ptr<ygm::container::map<supernode_t, full_leaf_cluster_info>>
                leaf_cluster_map_ptr,
    supernode_t chain_name = BLANK_SUPERNODE) {
  std::size_t chain_num_valid_clusters = 0;

  // Keep track of the chain's total size and stability traversing up so far
  // Initialize it with the info from the second min edge child
  id_t       chain_size      = chain_info.min_edge_child_size2;
  distance_t chain_stability = chain_info.min_edge_child_stability2;
  distance_t chain_stability_traversing_up =
      chain_info.min_edge_child_stability_traversing_up2;

  // Traverse up the chain, calculating size and stability for each cluster
  // along the way
  for (auto it = chain_cluster_map.begin(); it != chain_cluster_map.end();
       ++it) {
    // Get cluster info for easier referencing
    full_cluster_info cluster_info = it->second;

    // Calculate this cluster's size: sum of child sizes
    // plus any single nodes from edges added. All edges added
    // other than the min edge contribute 1 to the size.
    cluster_info.size =
        chain_size + cluster_info.child_size + cluster_info.edges.size() - 1;

    // If both the child clusters have at least min cluster size, they are
    // valid, set them as valid
    if (chain_size >= min_cluster_size &&
        cluster_info.child_size >= min_cluster_size) {
      chain_num_valid_clusters += 2;

      auto set_top_chain_cluster_as_valid =
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                 &chain) {
            auto top_cluster_it                  = chain.first.rbegin();
            top_cluster_it->second.valid_cluster = true;
          };
      auto set_leaf_cluster_as_valid =
          [](const supernode_t      &cluster_name,
             full_leaf_cluster_info &cluster_info) {
            cluster_info.valid_cluster = true;
          };

      // If we're at the bottom of the chain, set both chain children as valid
      if (it == chain_cluster_map.begin()) {
        for (supernode_t &chain_child : chain_info.children) {
          if (chain_child.second == 1) {
            leaf_cluster_map_ptr->async_visit(chain_child,
                                              set_leaf_cluster_as_valid);
          } else {
            chain_map_ptr->async_visit(chain_child,
                                       set_top_chain_cluster_as_valid);
          }
        }
      }
      // Otherwise, set the child chain cluster and the non-chain child as valid
      else {
        auto child_cluster_it                  = std::prev(it);
        child_cluster_it->second.valid_cluster = true;
        if (cluster_info.child.second == 1) {
          leaf_cluster_map_ptr->async_visit(cluster_info.child,
                                            set_leaf_cluster_as_valid);
        } else {
          chain_map_ptr->async_visit(cluster_info.child,
                                     set_top_chain_cluster_as_valid);
        }
      }
    }

    // Calculate our own stability
    distance_t stability = 0;

    // Also determine the stability we should send up for the purposes of
    // selecting clusters
    distance_t stability_traversing_up = 0;

    distance_t lambda_min_edge = lambda_from_dist(cluster_info.min_edge.second);
    distance_t lambda_birth    = lambda_from_dist(cluster_info.birth_distance);

    // If we are too small, then just keep stability = 0
    // Calculate our stability if we have at least min cluster size
    if (cluster_info.size >= min_cluster_size) {
      // If both child clusters are too small, annex them together
      // into our cluster
      if (cluster_info.child_size < min_cluster_size &&
          chain_size < min_cluster_size) {
        id_t sum_child_sizes = chain_size + cluster_info.child_size;
        if (sum_child_sizes >= min_cluster_size) {
          // In this case, at min_edge, sum_child_sizes >=
          // min_cluster_size points fall out of the cluster
          // We subtract 1 since we will count lambda_min_edge one more
          // time below since min_edge is in edges
          stability = (sum_child_sizes - 1) * (lambda_min_edge - lambda_birth);

          // All other points fall out at their edge
          for (edge_id_with_dist_t &edge : cluster_info.edges) {
            stability += lambda_from_dist(edge.second) - lambda_birth;
          }

        } else {
          // In this case, at some the k = (min_cluster_size -
          // sum_child_sizes + 1)th edge in this cluster,
          // min_cluster_size points fall out of the cluster

          // Rearrange the edges vector so that the first k edges are
          // less than or equal to all subsequent edges
          id_t k           = min_cluster_size - sum_child_sizes + 1;
          auto cutoff_edge = cluster_info.edges.begin() + k - 1;
          std::nth_element(cluster_info.edges.begin(), cutoff_edge,
                           cluster_info.edges.end());

          // min_cluster_size points fall out at the cutoff edge
          stability = min_cluster_size *
                      (lambda_from_dist(cutoff_edge->second) - lambda_birth);

          // All other points fall out at their edge
          for (int i = k; i < static_cast<int>(cluster_info.edges.size());
               ++i) {
            edge_id_with_dist_t edge = cluster_info.edges.at(i);
            stability += lambda_from_dist(edge.second) - lambda_birth;
          }
        }

        // In this case, the stability to send up is our own stability
        stability_traversing_up = stability;
      }
      // If one child cluster is too small, but the other is large
      // enough, annex the small cluster to ourselves
      else if ((chain_size >= min_cluster_size &&
                cluster_info.child_size < min_cluster_size) ||
               (chain_size < min_cluster_size &&
                cluster_info.child_size >= min_cluster_size)) {
        bool chain_child_is_large_cluster = true;
        if (cluster_info.child_size >= min_cluster_size) {
          chain_child_is_large_cluster = false;
        }

        // 1. Start out with the stability of the large enough cluster
        // 2. Correct the birth distance for the points in the large-enough
        // cluster to be lambda_birth instead of lambda_min_edge (what we
        // thought was the birth distance before annexing)
        // 3. Add the stability contribution from the too-small cluster
        // we annexed
        if (chain_child_is_large_cluster) {
          stability = chain_stability;
          stability += chain_size * (lambda_min_edge - lambda_birth);
          stability +=
              cluster_info.child_size * (lambda_min_edge - lambda_birth);

        } else {
          stability = cluster_info.child_stability;
          stability +=
              cluster_info.child_size * (lambda_min_edge - lambda_birth);
          stability += chain_size * (lambda_min_edge - lambda_birth);
        }

        // Add the stability contribution of other edges added
        for (edge_id_with_dist_t &edge : cluster_info.edges) {
          stability += lambda_from_dist(edge.second) - lambda_birth;
        }
        // Since min_edge is in cluster_info.edges, we've over counted its
        // contribution. Subtract the extra instance
        stability -= (lambda_min_edge - lambda_birth);

        // In this case, the stability to send up the greater of our own
        // stability and the stability of the large child cluster
        if (chain_child_is_large_cluster) {
          stability_traversing_up =
              std::max(stability, chain_stability_traversing_up);
        } else {
          stability_traversing_up =
              std::max(stability, cluster_info.child_stability_traversing_up);
        }
      }
      // If both child clusters are large enough to be their own
      // cluster
      else {
        // Points from the two child clusters fall out at min_edge
        stability = (chain_size + cluster_info.child_size) *
                    (lambda_min_edge - lambda_birth);

        // Add the stability contributions from the points added to
        // this cluster, which come from all edges added other than
        // min_edge
        for (edge_id_with_dist_t &edge : cluster_info.edges) {
          stability += lambda_from_dist(edge.second) - lambda_birth;
        }
        // Since min_edge is in cluster_info.edges, we've over counted its
        // contribution. Subtract the extra instance
        stability -= (lambda_min_edge - lambda_birth);

        // In this case, the stability to send up is the greater of our own
        // stability and the sum of our child stabilities traversing up
        stability_traversing_up =
            std::max(stability, chain_stability_traversing_up +
                                    cluster_info.child_stability_traversing_up);
      }
    }

    // Set size and stability for this cluster
    it->second.size                    = cluster_info.size;
    it->second.stability               = stability;
    it->second.stability_traversing_up = stability_traversing_up;

    // Update the chain size and stability to this cluster's then move on to the
    // next one
    chain_size                    = it->second.size;
    chain_stability               = it->second.stability;
    chain_stability_traversing_up = it->second.stability_traversing_up;
  }

  return chain_num_valid_clusters;
}

/**
 * @brief Traverse up the cluster hierarchy from leaf clusters and fill in
 * cluster sizes and stabilities. Stop when we reach the root chain, which will
 * be processed separately
 *
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info)
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info. This map contains all root chain clusters regardless of
 * if they are valid.
 * @param _min_cluster_size HDBSCAN min cluster size parameter
 * @param _root_chain_supernode Name of the root chain supernode.
 *
 * @return The number of valid clusters found when processing clusters below the
 * root chain.
 */
std::size_t traverse_up_cluster_hierarchy_until_root_chain(
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>          &chain_map,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    const int _min_cluster_size, const supernode_t _root_chain_supernode) {
  // Get ygm comm and pointer to chain map
  ygm::comm &comm                       = chain_map.comm();
  auto       chain_map_ptr              = chain_map.get_ygm_ptr();
  auto       leaf_cluster_map_ptr       = leaf_cluster_map.get_ygm_ptr();
  auto       root_chain_cluster_map_ptr = root_chain_cluster_map.get_ygm_ptr();

  // Define static variables for the (local) number of clusters seen within this
  // function.
  static std::size_t num_valid_clusters;
  num_valid_clusters = 0;

  // Make min_cluster_size and root_chain_supernode static within this function
  // for easy access
  static uint32_t min_cluster_size;
  min_cluster_size = _min_cluster_size;
  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;

  // Lambda to send child cluster information to root chain cluster
  static auto send_child_to_root_chain_cluster =
      [](const id_t &cluster_edge_id, root_chain_cluster_info &cluster_info,
         const supernode_t &child_chain_name, id_t child_size,
         distance_t child_stability, distance_t child_stability_traversing_up) {
        if (cluster_info.child == child_chain_name) {
          cluster_info.child_size      = child_size;
          cluster_info.child_stability = child_stability;
          cluster_info.child_stability_traversing_up =
              child_stability_traversing_up;
        }
      };

  // Walking up the cluster hierarchy and sending calculated cluster info
  // to the cluster parent can be expressed as a recursive operation
  struct send_cluster_to_parent_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<
            supernode_t,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                           chain_map_ptr,
        const supernode_t &chain_name,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
        const supernode_t &child_chain_name, id_t cluster_edge_id,
        id_t child_size, distance_t child_stability,
        distance_t child_stability_traversing_up,
        ygm::ygm_ptr<ygm::container::map<supernode_t, full_leaf_cluster_info>>
            leaf_cluster_map_ptr,
        ygm::ygm_ptr<ygm::container::map<id_t, root_chain_cluster_info>>
            root_chain_cluster_map_ptr) {
      // Increment the number of children received by this chain
      chain.second.num_child_messages_received += 1;

      // The bottom cluster edge id for this chain
      id_t min_edge_id = chain.first.begin()->first;

      // If we received child info for the bottom cluster, check which location
      // to store it
      if (cluster_edge_id == min_edge_id) {
        // For the first bottom child, store info in the chain's cluster map
        if (child_chain_name == chain.second.children[0]) {
          chain.first[cluster_edge_id].child           = child_chain_name;
          chain.first[cluster_edge_id].child_size      = child_size;
          chain.first[cluster_edge_id].child_stability = child_stability;
          chain.first[cluster_edge_id].child_stability_traversing_up =
              child_stability_traversing_up;
        }
        // For the second bottom child, store the info in full_chain_info
        else if (child_chain_name == chain.second.children[1]) {
          chain.second.min_edge_child_size2      = child_size;
          chain.second.min_edge_child_stability2 = child_stability;
          chain.second.min_edge_child_stability_traversing_up2 =
              child_stability_traversing_up;
        } else {
          std::cout << "Error! Chain " << chain_name
                    << " has supernode children " << chain.second.children[0]
                    << " and " << chain.second.children[1]
                    << ", but received child " << child_chain_name << std::endl;
        }
      }
      // Otherwise, just directly store the child info in the corresponding
      // parent cluster
      else {
        chain.first[cluster_edge_id].child           = child_chain_name;
        chain.first[cluster_edge_id].child_size      = child_size;
        chain.first[cluster_edge_id].child_stability = child_stability;
        chain.first[cluster_edge_id].child_stability_traversing_up =
            child_stability_traversing_up;
      }

      // If we received all child messages (one per cluster plus an extra one
      // for the bottom cluster), then calculate all sizes and stabilities in
      // the chain.
      if (chain.second.num_child_messages_received == chain.first.size() + 1) {
        id_t chain_num_valid_clusters =
            calculate_chain_cluster_size_stability_from_children(
                chain.first, chain.second, min_cluster_size, chain_map_ptr,
                leaf_cluster_map_ptr, chain_name);
        num_valid_clusters += chain_num_valid_clusters;

        auto top_cluster_it = chain.first.rbegin();

        // Send the info of the top cluster to the parent chain
        if (chain.second.parent_chain != root_chain_supernode) {
          chain_map_ptr->async_visit(
              chain.second.parent_chain, send_cluster_to_parent_functor(),
              chain_name, chain.second.parent_edge_id,
              top_cluster_it->second.size, top_cluster_it->second.stability,
              top_cluster_it->second.stability_traversing_up,
              leaf_cluster_map_ptr, root_chain_cluster_map_ptr);
        } else {
          root_chain_cluster_map_ptr->async_visit(
              chain.second.parent_edge_id, send_child_to_root_chain_cluster,
              chain_name, top_cluster_it->second.size,
              top_cluster_it->second.stability,
              top_cluster_it->second.stability_traversing_up);
        }
      }
    }
  };

  auto process_leaf_clusters_lambda =
      [chain_map_ptr, leaf_cluster_map_ptr, root_chain_cluster_map_ptr](
          [[maybe_unused]] const supernode_t &cluster_name,
          full_leaf_cluster_info             &cluster_info) {
        // Cluster size (number of nodes) = number of edges + 1
        id_t cluster_size = cluster_info.edges.size() + 1;

        // If this cluster is too small, it has stability 0
        if (cluster_size < min_cluster_size) {
          cluster_info.stability = 0;
        }
        // Otherwise, calculate the stability
        else {
          distance_t lambda_birth =
              lambda_from_dist(cluster_info.birth_distance);

          // Rearrange the edges vector so that the first
          // (min_cluster_size - 1) edges are less than or equal to all
          // subsequent edges
          // Note: .begin() gives the first element
          auto cutoff_edge = cluster_info.edges.begin() + min_cluster_size - 2;
          std::nth_element(cluster_info.edges.begin(), cutoff_edge,
                           cluster_info.edges.end());

          // The min_cluster_size points attached to the bottom
          // (min_cluster_size - 1) edges all fall out at the
          // (min_cluster_size - 1)th-smallest edge
          distance_t stability =
              min_cluster_size *
              (lambda_from_dist(cutoff_edge->second) - lambda_birth);

          // All other points fall out at their edge
          for (int i = min_cluster_size - 1;
               i < static_cast<int>(cluster_info.edges.size()); ++i) {
            edge_id_with_dist_t edge = cluster_info.edges.at(i);
            stability += lambda_from_dist(edge.second) - lambda_birth;
          }

          // Set the stability
          cluster_info.stability = stability;
        }

        // Send this cluster's size and stability to its parent chain if its not
        // the root chain For a leaf cluster, the stability traversing up is its
        // own stability
        if (cluster_info.parent_chain != root_chain_supernode) {
          chain_map_ptr->async_visit(
              cluster_info.parent_chain, send_cluster_to_parent_functor(),
              cluster_name, cluster_info.parent_edge_id, cluster_size,
              cluster_info.stability, cluster_info.stability,
              leaf_cluster_map_ptr, root_chain_cluster_map_ptr);
        } else {
          root_chain_cluster_map_ptr->async_visit(
              cluster_info.parent_edge_id, send_child_to_root_chain_cluster,
              cluster_name, cluster_size, cluster_info.stability,
              cluster_info.stability);
        }
      };
  leaf_cluster_map.for_all(process_leaf_clusters_lambda);
  comm.barrier();

  return num_valid_clusters;
}

/**
 * @brief Make sure bottom root chain cluster has two children with at least min
 * cluster size. We need this assumption for size/stability processing.
 *
 * @details If the bottom root chain cluster does not have two children with at
 * least min cluster size, then split off the bottom of the root chain into a
 * separate chain and start from a cluster with two valid children. This
 * function also gets the root chain second child info and sets the top cluster
 * of the second root chain child as valid.
 *
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info. This map contains all root chain clusters regardless of
 * if they are valid.
 * @param root_chain_cluster_edges_map YGM map of root chain cluster edge id ->
 * vector of edges added to the cluster.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param root_chain_min_edge_id Edge id for the bottom cluster of the root
 * chain (i.e., the smallest alpha-edge id in the root chain). We visit its
 * corresponding cluster to check if it has two valid children. If not, we start
 * traversing up the root chain from here.
 * @param _min_cluster_size HDBSCAN min cluster size parameter
 * @param _root_chain_supernode Name of the root chain supernode.
 */
void make_sure_root_chain_bottom_is_valid_and_get_second_child(
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
        &root_chain_cluster_edges_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>          &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    id_t                      root_chain_min_edge_id,
    extra_child_cluster_info &root_chain_second_child,
    const int _min_cluster_size, const supernode_t _root_chain_supernode) {
  // Get ygm comm and pointer to chain map
  ygm::comm &comm                       = root_chain_cluster_map.comm();
  auto       root_chain_cluster_map_ptr = root_chain_cluster_map.get_ygm_ptr();

  // Make min_cluster_size and root_chain_supernode
  // static within this function for easy access
  static uint32_t min_cluster_size;
  min_cluster_size = _min_cluster_size;
  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;

  // Get the root chain second child info and copy it on all ranks
  {
    id_t              size;
    distance_t        stability, stability_traversing_up;
    static id_t       local_size;
    static distance_t local_stability, local_stability_traversing_up;
    local_size                    = 0;
    local_stability               = 0.0;
    local_stability_traversing_up = 0.0;
    if (comm.rank() == 0) {
      chain_map.async_visit(
          root_chain_second_child.name,
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                 &chain) {
            auto top_cluster_it = chain.first.rbegin();
            local_size          = top_cluster_it->second.size;
            local_stability     = top_cluster_it->second.stability;
            local_stability_traversing_up =
                top_cluster_it->second.stability_traversing_up;
          });
    }
    comm.barrier();

    MPI_Allreduce(&local_size, &size, 1, mpi_id_type(), MPI_MAX,
                  comm.get_mpi_comm());
    MPI_Allreduce(&local_stability, &stability, 1, mpi_distance_type(), MPI_MAX,
                  comm.get_mpi_comm());
    MPI_Allreduce(&local_stability_traversing_up, &stability_traversing_up, 1,
                  mpi_distance_type(), MPI_MAX, comm.get_mpi_comm());
    root_chain_second_child.size                    = size;
    root_chain_second_child.stability               = stability;
    root_chain_second_child.stability_traversing_up = stability_traversing_up;
  }

  // Check if the bottom root chain cluster is valid, and if not, keep sending
  // sizes and stabilities up the root chain until we get a valid cluster
  static id_t local_root_chain_min_valid_cluster_id = 0;
  struct walk_up_root_chain_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<id_t, root_chain_cluster_info>>
                    root_chain_cluster_map_ptr,
        const id_t &cluster_edge_id, root_chain_cluster_info &cluster_info,
        id_t chain_child_size) {
      // std::cout << "Visiting root chain cluster " << cluster_edge_id
      //           << " with child size " << cluster_info.child_size
      //           << " and chain child size = " << chain_child_size
      //           << std::endl;  // debug

      // If this is a valid root chain cluster (both children have min cluster
      // size), set this as the min valid cluster id and stop
      if (chain_child_size >= min_cluster_size &&
          cluster_info.child_size >= min_cluster_size) {
        local_root_chain_min_valid_cluster_id = cluster_edge_id;
      } else {
        // Update the chain child size and traverse up to the next root chain
        // cluster
        id_t new_chain_child_size = chain_child_size + cluster_info.child_size +
                                    cluster_info.num_points_added;
        root_chain_cluster_map_ptr->async_visit(cluster_info.parent_edge_id,
                                                walk_up_root_chain_functor(),
                                                new_chain_child_size);
      }
    }
  };
  if (comm.rank() == 0) {
    root_chain_cluster_map.async_visit(root_chain_min_edge_id,
                                       walk_up_root_chain_functor(),
                                       root_chain_second_child.size);
  }
  comm.barrier();

  // Get the min valid cluster id on each rank
  id_t _root_chain_min_valid_cluster_id;
  MPI_Allreduce(&local_root_chain_min_valid_cluster_id,
                &_root_chain_min_valid_cluster_id, 1, mpi_id_type(), MPI_MAX,
                comm.get_mpi_comm());
  static id_t root_chain_min_valid_cluster_id;
  root_chain_min_valid_cluster_id = _root_chain_min_valid_cluster_id;

  // If the min valid cluster id isn't the min cluster id for the root chain,
  // take the chunk of invalid root chain clusters and add them to the chain map
  if (root_chain_min_edge_id < root_chain_min_valid_cluster_id) {
    comm.cout0() << "Need to correct the bottom of the root chain"
                 << std::endl;  // debug

    auto chain_map_ptr        = chain_map.get_ygm_ptr();
    auto leaf_cluster_map_ptr = leaf_cluster_map.get_ygm_ptr();
    auto root_chain_cluster_edges_map_ptr =
        root_chain_cluster_edges_map.get_ygm_ptr();

    // The root chain supernode is (0, final_round + 1), name its new offshoot
    // chain (1, final_round + 1)
    static supernode_t root_chain_offshoot_supernode =
        std::make_pair(1, root_chain_supernode.second);

    comm.cout0() << "Root chain offshoot name: "
                 << root_chain_offshoot_supernode << std::endl;  // debug

    static auto set_new_parent_lambda =
        [](const supernode_t &chain_name,
           std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
           const supernode_t &parent_chain) {
          chain.second.parent_chain = parent_chain;
        };

    // Make a new chain for this root chain chunk and fill in the extra chain
    // info (e.g. from root_chain_second_child)
    // Also set the two old root chain child chains to have the new
    // root_chain_offshoot_supernode as their parent
    if (comm.rank() == 0) {
      full_chain_info chain_info = {
          .parent_chain              = root_chain_supernode,
          .parent_edge_id            = root_chain_min_valid_cluster_id,
          .min_edge_child_size2      = root_chain_second_child.size,
          .min_edge_child_stability2 = root_chain_second_child.stability,
          .min_edge_child_stability_traversing_up2 =
              root_chain_second_child.stability_traversing_up,
          .children = {BLANK_SUPERNODE, root_chain_second_child.name}};

      chain_map.async_visit(
          root_chain_offshoot_supernode,
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                                   &chain,
             const full_chain_info &chain_info) { chain.second = chain_info; },
          chain_info);

      chain_map.async_visit(root_chain_second_child.name, set_new_parent_lambda,
                            root_chain_offshoot_supernode);

      auto get_first_child_lambda =
          [](const id_t                    &cluster_edge_id,
             const root_chain_cluster_info &cluster_info, auto chain_map_ptr) {
            chain_map_ptr->async_visit(
                root_chain_offshoot_supernode,
                [](const supernode_t &chain_name,
                   std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                                     &chain,
                   const supernode_t &child) {
                  chain.second.children[0] = child;
                },
                cluster_info.child);
            chain_map_ptr->async_visit(cluster_info.child,
                                       set_new_parent_lambda,
                                       root_chain_offshoot_supernode);
          };
      root_chain_cluster_map_ptr->async_visit(
          root_chain_min_edge_id, get_first_child_lambda, chain_map_ptr);
    }
    comm.barrier();

    static std::vector<id_t> local_root_chain_clusters_to_remove;

    struct move_root_chain_clusters_to_chain_map_functor {
      void operator()(
          ygm::ygm_ptr<ygm::container::map<id_t, root_chain_cluster_info>>
                                   root_chain_cluster_map_ptr,
          const id_t              &cluster_edge_id,
          root_chain_cluster_info &root_chain_cluster_info,
          ygm::ygm_ptr<ygm::container::map<
              supernode_t,
              std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
              chain_map_ptr,
          ygm::ygm_ptr<
              ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>>
              root_chain_cluster_edges_map_ptr) {
        if (cluster_edge_id < root_chain_min_valid_cluster_id) {
          // Add this cluster to the root chain offshoot chain
          full_cluster_info chain_cluster_info = {
              .child           = root_chain_cluster_info.child,
              .child_stability = root_chain_cluster_info.child_stability,
              .child_stability_traversing_up =
                  root_chain_cluster_info.child_stability_traversing_up,
              .child_size = root_chain_cluster_info.child_size,
              .min_edge =
                  std::make_pair(cluster_edge_id,
                                 1.0 / root_chain_cluster_info.lambda_min_edge),
              .birth_distance = 1.0 / root_chain_cluster_info.lambda_birth,
          };
          auto add_cluster_to_chain_map_lambda =
              [](const supernode_t &chain_name,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                                         &chain,
                 const id_t              &cluster_edge_id,
                 const full_cluster_info &cluster_info) {
                chain.first[cluster_edge_id].child = cluster_info.child;
                chain.first[cluster_edge_id].child_stability =
                    cluster_info.child_stability;
                chain.first[cluster_edge_id].child_stability_traversing_up =
                    cluster_info.child_stability_traversing_up;
                chain.first[cluster_edge_id].child_size =
                    cluster_info.child_size;
                chain.first[cluster_edge_id].min_edge = cluster_info.min_edge;
                chain.first[cluster_edge_id].birth_distance =
                    cluster_info.birth_distance;
              };
          chain_map_ptr->async_visit(root_chain_offshoot_supernode,
                                     add_cluster_to_chain_map_lambda,
                                     cluster_edge_id, chain_cluster_info);

          // Set the child cluster to have the root_chain_offshoot_supernode as
          // its parent
          chain_map_ptr->async_visit(root_chain_cluster_info.child,
                                     set_new_parent_lambda,
                                     root_chain_offshoot_supernode);

          // Transfer over the cluster edges
          auto transfer_edges_lambda =
              [](const id_t                       &cluster_edge_id,
                 std::vector<edge_id_with_dist_t> &edges, auto chain_map_ptr) {
                chain_map_ptr->async_visit(
                    root_chain_offshoot_supernode,
                    [](const supernode_t                      &chain_name,
                       std::pair<std::map<id_t, full_cluster_info>,
                                 full_chain_info>             &chain,
                       const id_t                             &cluster_edge_id,
                       const std::vector<edge_id_with_dist_t> &edges) {
                      chain.first[cluster_edge_id].edges = edges;
                    },
                    cluster_edge_id, edges);
              };
          root_chain_cluster_edges_map_ptr->async_visit(
              cluster_edge_id, transfer_edges_lambda, chain_map_ptr);
          local_root_chain_clusters_to_remove.push_back(cluster_edge_id);

          root_chain_cluster_map_ptr->async_visit(
              root_chain_cluster_info.parent_edge_id,
              move_root_chain_clusters_to_chain_map_functor(), chain_map_ptr,
              root_chain_cluster_edges_map_ptr);
        }
      }
    };
    root_chain_cluster_map.async_visit(
        root_chain_min_edge_id, move_root_chain_clusters_to_chain_map_functor(),
        chain_map_ptr, root_chain_cluster_edges_map_ptr);
    comm.barrier();

    // Delete the invalid clusters from root_chain_cluster_edges_map
    // after transfering edges
    for (id_t &cluster_edge_id : local_root_chain_clusters_to_remove) {
      root_chain_cluster_edges_map.async_erase(cluster_edge_id);
    }
    comm.barrier();
    local_root_chain_clusters_to_remove.clear();

    // Calculate the size and stability in the root chain offshoot
    if (comm.rank() == 0) {
      chain_map.async_visit(
          root_chain_offshoot_supernode,
          [](const supernode_t &chain_name,
             std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                 &chain,
             ygm::ygm_ptr<ygm::container::map<
                 supernode_t,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                 chain_map_ptr,
             ygm::ygm_ptr<
                 ygm::container::map<supernode_t, full_leaf_cluster_info>>
                 leaf_cluster_map_ptr) {
            calculate_chain_cluster_size_stability_from_children(
                chain.first, chain.second, min_cluster_size, chain_map_ptr,
                leaf_cluster_map_ptr);
          },
          chain_map_ptr, leaf_cluster_map_ptr);
    }
    comm.barrier();

    // Get the corrected root_chain_second_child information
    // Get the root chain second child info and copy it on all ranks
    {
      root_chain_second_child.name = root_chain_offshoot_supernode;

      id_t              size;
      distance_t        stability, stability_traversing_up;
      static id_t       local_size;
      static distance_t local_stability, local_stability_traversing_up;
      local_size                    = 0;
      local_stability               = 0.0;
      local_stability_traversing_up = 0.0;
      if (comm.rank() == 0) {
        chain_map.async_visit(
            root_chain_second_child.name,
            [](const supernode_t &chain_name,
               std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                   &chain) {
              auto top_cluster_it = chain.first.rbegin();
              local_size          = top_cluster_it->second.size;
              local_stability     = top_cluster_it->second.stability;
              local_stability_traversing_up =
                  top_cluster_it->second.stability_traversing_up;
            });
      }
      comm.barrier();
      MPI_Allreduce(&local_size, &size, 1, mpi_id_type(), MPI_MAX,
                    comm.get_mpi_comm());
      MPI_Allreduce(&local_stability, &stability, 1, mpi_distance_type(),
                    MPI_MAX, comm.get_mpi_comm());
      MPI_Allreduce(&local_stability_traversing_up, &stability_traversing_up, 1,
                    mpi_distance_type(), MPI_MAX, comm.get_mpi_comm());
      root_chain_second_child.size                    = size;
      root_chain_second_child.stability               = stability;
      root_chain_second_child.stability_traversing_up = stability_traversing_up;
    }

  }  // End adding to chain map

  // Set the top cluster of the second root chain child as valid
  if (comm.rank() == 0) {
    chain_map.async_visit(
        root_chain_second_child.name,
        [](const supernode_t &chain_name,
           std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
               &chain) {
          auto it                  = chain.first.rbegin();
          it->second.valid_cluster = true;
        });
  }
  comm.barrier();
}

/**
 * @brief Some root chain clusters have non-root-chain child with size less than
 * min_cluster_size, resulting in invalid root chain clusters (instead the
 * clusters annex the small child into the large child)
 */

/**
 * @brief Make sure bottom root chain cluster has two children with at least min
 * cluster size. We need this assumption for size/stability processing.
 *
 * @detail If the bottom root chain cluster does not have two children with at
 * least min cluster size, then split off the bottom of the root chain into a
 * separate chain and start from a cluster with two valid children. This
 * function also gets the root chain second child info and sets the top cluster
 * of the second root chain child as valid.
 *
 * @param full_root_chain_array Sorted YGM array of (cluster edge id,
 * root chain cluster info).
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info. This map contains all root chain clusters regardless of
 * if they are valid. This function will update the map by merging clusters so
 * that we only keep valid root chain clusters.
 * @param root_chain_cluster_edges_map YGM map of root chain cluster edge id ->
 * vector of edges added to the cluster. This map contains all root chain
 * clusters regardless of if they are valid. This function will update the map
 * by merging clusters so that we only keep valid root chain clusters.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param min_cluster_size HDBSCAN min cluster size parameter
 */
void merge_invalid_root_chain_clusters(
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
                                                       &full_root_chain_array,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
        &root_chain_cluster_edges_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>          &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    const int min_cluster_size) {
  ygm::comm &comm          = chain_map.comm();
  auto       chain_map_ptr = chain_map.get_ygm_ptr();
  auto       root_chain_cluster_edges_map_ptr =
      root_chain_cluster_edges_map.get_ygm_ptr();
  auto leaf_cluster_map_ptr = leaf_cluster_map.get_ygm_ptr();

  // Store the full_root_chain_array indicies that contain root chain
  // clusters to keep after merge
  std::vector<id_t> local_index_to_keep;

  // Initialize a map to root chain cluster (represented as just its edge)
  // to merge into -> vector of clusters that merge into it
  std::map<id_t, std::vector<id_t>> local_merge_clusters_into_map;
  bool                              merge_to_rank_below = false;

  // Process local cluster merges
  auto cluster_below_it = full_root_chain_array.local_begin();
  id_t cluster_below_edge_id, first_edge_id;
  if (full_root_chain_array.local_size() > 0) {
    cluster_below_it      = full_root_chain_array.local_begin();
    cluster_below_edge_id = cluster_below_it->value.first;
    first_edge_id         = cluster_below_edge_id;
    for (auto it = full_root_chain_array.local_begin();
         it != full_root_chain_array.local_end(); ++it) {
      root_chain_cluster_info cluster_info = it->value.second;

      // For first cluster, check if we need to merge below. We can't do
      // the merging yet since the cluster below lives on another rank
      if (it == full_root_chain_array.local_begin()) {
        if (cluster_info.child_size < min_cluster_size) {
          merge_to_rank_below = true;
        } else {
          local_index_to_keep.push_back(it->index);
        }
      } else {
        // Do the merging locally
        // If the child size is large enough, then this cluster becomes the
        // new cluster to merge into
        if (cluster_info.child_size >= min_cluster_size) {
          cluster_below_it      = it;
          cluster_below_edge_id = it->value.first;
          local_index_to_keep.push_back(it->index);
        }
        // Otherwise, annex this cluster to the cluster to merge into below
        else {
          // Store which clusters merged in the map, we will use this to
          // update edges added later
          local_merge_clusters_into_map[cluster_below_edge_id].push_back(
              it->value.first);

          // Update the num points and sum lambda edges of the cluster we're
          // merging into
          cluster_below_it->value.second.num_points_added +=
              cluster_info.child_size + cluster_info.num_points_added;
          cluster_below_it->value.second.sum_lambda_edges_added +=
              cluster_info.child_size * cluster_info.lambda_min_edge +
              cluster_info.sum_lambda_edges_added;

          // Update the parent of the cluster below
          cluster_below_it->value.second.parent_edge_id =
              it->value.second.parent_edge_id;
          cluster_below_it->value.second.lambda_birth =
              it->value.second.lambda_birth;
        }
      }
    }
  }
  comm.barrier();

  // Handle clusters needing to merge to one stored on a rank below. Do
  // this from highest rank to lowest rank sequentially
  auto mpi_comm  = comm.get_mpi_comm();
  int  comm_size = comm.size();

  if (full_root_chain_array.size() > 0) {
    // All gather to see which ranks need to merge down
    // Using int instead of bool since the packed std::vector<bool>
    // container causes issues
    int              local_val = merge_to_rank_below ? 1 : 0;
    std::vector<int> all_merge_to_rank_below_bools(comm_size);
    MPI_Allgather(&local_val, 1, MPI_INT, all_merge_to_rank_below_bools.data(),
                  1, MPI_INT, mpi_comm);

    // Iterate from highest to lowest rank and merge cluster down if needed
    // Send info as 3 messages and use tags to keep them apart
    int tag_size                  = 1;
    int tag_vector_of_cluster_ids = 2;
    int tag_cluster_info          = 3;
    for (int current_rank = comm_size - 1; current_rank > 0; --current_rank) {
      // If we have a merge, current ranks sends information down to rank below
      if (comm.rank() == current_rank &&
          all_merge_to_rank_below_bools.at(current_rank) == 1) {
        // Get the clusters to merge to rank below
        std::vector<id_t> clusters_to_merge_to_rank_below =
            local_merge_clusters_into_map[first_edge_id];
        // Make sure to also include to the first cluster that we're sending
        clusters_to_merge_to_rank_below.push_back(first_edge_id);
        // Erase from the local merge map since they'll appear on the local
        // merge map on the rank below instead
        local_merge_clusters_into_map.erase(first_edge_id);

        // Send size of clusters_to_merge_to_rank_below vector
        int num_cluster_ids =
            static_cast<int>(clusters_to_merge_to_rank_below.size());
        MPI_Send(&num_cluster_ids, 1, MPI_INT, current_rank - 1, tag_size,
                 mpi_comm);

        // Send vector of cluster ids to merge in
        MPI_Send(clusters_to_merge_to_rank_below.data(), num_cluster_ids,
                 MPI_INT, current_rank - 1, tag_vector_of_cluster_ids,
                 mpi_comm);

        // Send custom root-chain cluster info struct
        root_chain_cluster_info bottom_cluster_to_send =
            full_root_chain_array.local_begin()->value.second;
        MPI_Send(&bottom_cluster_to_send, sizeof(root_chain_cluster_info),
                 MPI_BYTE, current_rank - 1, tag_cluster_info, mpi_comm);
      }

      // The rank below receives the merge cluster information
      if (comm.rank() == current_rank - 1 &&
          all_merge_to_rank_below_bools.at(current_rank) == 1) {
        // Receive size of clusters_to_merge_to_rank_below vector
        int num_cluster_ids = 0;
        MPI_Recv(&num_cluster_ids, 1, MPI_INT, current_rank, tag_size, mpi_comm,
                 MPI_STATUS_IGNORE);

        // Receive vector of cluster ids to merge in
        std::vector<id_t> received_cluster_ids_to_merge(num_cluster_ids);
        MPI_Recv(received_cluster_ids_to_merge.data(), num_cluster_ids, MPI_INT,
                 current_rank, tag_vector_of_cluster_ids, mpi_comm,
                 MPI_STATUS_IGNORE);

        // Receive cluster info from rank above
        root_chain_cluster_info received_cluster_info;
        MPI_Recv(&received_cluster_info, sizeof(root_chain_cluster_info),
                 MPI_BYTE, current_rank, tag_cluster_info, mpi_comm,
                 MPI_STATUS_IGNORE);

        // Merge in the received cluster info to the top valid cluster
        // on this rank, which is at cluster_below_it and
        // cluster_below_edge_id
        local_merge_clusters_into_map[cluster_below_edge_id].insert(
            local_merge_clusters_into_map[cluster_below_edge_id].end(),
            received_cluster_ids_to_merge.begin(),
            received_cluster_ids_to_merge.end());

        cluster_below_it->value.second.num_points_added +=
            received_cluster_info.child_size +
            received_cluster_info.num_points_added;
        cluster_below_it->value.second.sum_lambda_edges_added +=
            received_cluster_info.child_size *
                received_cluster_info.lambda_min_edge +
            received_cluster_info.sum_lambda_edges_added;
        cluster_below_it->value.second.parent_edge_id =
            received_cluster_info.parent_edge_id;
        cluster_below_it->value.second.lambda_birth =
            received_cluster_info.lambda_birth;
      }

      // Finish this merge and update before checking the next rank
      MPI_Barrier(mpi_comm);
    }
  }
  comm.barrier();

  // Update the root chain cluster map with the new post-merge info
  auto        root_chain_cluster_map_ptr = root_chain_cluster_map.get_ygm_ptr();
  static auto update_cluster_info_lambda =
      []([[maybe_unused]] const id_t   &cluster_edge_id,
         root_chain_cluster_info       &cluster_info,
         const root_chain_cluster_info &new_cluster_info) {
        cluster_info = new_cluster_info;
      };

  for (id_t &index : local_index_to_keep) {
    full_root_chain_array.local_visit(
        index, [root_chain_cluster_map_ptr](
                   [[maybe_unused]] const id_t                    &index,
                   const std::pair<id_t, root_chain_cluster_info> &value) {
          root_chain_cluster_map_ptr->async_visit(
              value.first, update_cluster_info_lambda, value.second);
        });
  }
  comm.barrier();

  // Now that we have the final indicies and updated cluster information of the
  // merged root chain clusters, clear the full_root_chain_array to free up
  // space
  full_root_chain_array.clear();

  // Update the edges added for clusters that we merged into and delete
  // the merged root-chain clusters to save space
  static auto add_edges_lambda =
      []([[maybe_unused]] const id_t            &cluster_edge_id,
         std::vector<edge_id_with_dist_t>       &cluster_edges,
         const std::vector<edge_id_with_dist_t> &new_edges_added) {
        cluster_edges.insert(cluster_edges.end(), new_edges_added.begin(),
                             new_edges_added.end());
      };

  static auto get_leaf_cluster_edges_lambda =
      []([[maybe_unused]] const supernode_t &cluster_name,
         full_leaf_cluster_info             &cluster_info,
         const id_t                         &merge_into_cluster_edge_id,
         ygm::ygm_ptr<
             ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>>
             root_chain_cluster_edges_map_ptr) {
        root_chain_cluster_edges_map_ptr->async_visit(
            merge_into_cluster_edge_id, add_edges_lambda, cluster_info.edges);
      };

  struct merge_edges_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<
            supernode_t,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                                            chain_map_ptr,
        [[maybe_unused]] const supernode_t &chain_name,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
        const id_t &merge_into_cluster_edge_id,
        ygm::ygm_ptr<
            ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>>
            root_chain_cluster_edges_map_ptr,
        ygm::ygm_ptr<ygm::container::map<supernode_t, full_leaf_cluster_info>>
            leaf_cluster_map_ptr) {
      // For each cluster in the chain
      for (auto &[cluster_edge_id, cluster_info] : chain.first) {
        // add its edges to the root chain cluster merged into
        root_chain_cluster_edges_map_ptr->async_visit(
            merge_into_cluster_edge_id, add_edges_lambda, cluster_info.edges);

        // Visit its non chain child and do the same
        if (cluster_info.child.second == 1) {
          leaf_cluster_map_ptr->async_visit(
              cluster_info.child, get_leaf_cluster_edges_lambda,
              merge_into_cluster_edge_id, root_chain_cluster_edges_map_ptr);
        } else {
          chain_map_ptr->async_visit(cluster_info.child, merge_edges_functor(),
                                     merge_into_cluster_edge_id,
                                     root_chain_cluster_edges_map_ptr,
                                     leaf_cluster_map_ptr);
        }
      }

      // Also visit the second chain child
      if (chain.second.children[1].second == 1) {
        leaf_cluster_map_ptr->async_visit(
            chain.second.children[1], get_leaf_cluster_edges_lambda,
            merge_into_cluster_edge_id, root_chain_cluster_edges_map_ptr);
      } else {
        chain_map_ptr->async_visit(
            chain.second.children[1], merge_edges_functor(),
            merge_into_cluster_edge_id, root_chain_cluster_edges_map_ptr,
            leaf_cluster_map_ptr);
      }
    }
  };

  for (auto &[merge_to, vector_of_merge_from] : local_merge_clusters_into_map) {
    for (id_t merge_from : vector_of_merge_from) {
      // Merge in this root chain cluster's edges
      auto get_merge_from_edges =
          []([[maybe_unused]] const id_t            &cluster_edge_id,
             const std::vector<edge_id_with_dist_t> &edges,
             const id_t                             &merge_into_cluster_edge_id,
             ygm::ygm_ptr<
                 ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>>
                 root_chain_cluster_edges_map_ptr) {
            root_chain_cluster_edges_map_ptr->async_visit(
                merge_into_cluster_edge_id, add_edges_lambda, edges);
          };
      root_chain_cluster_edges_map.async_visit(
          merge_from, get_merge_from_edges, merge_to,
          root_chain_cluster_edges_map_ptr);

      // Visit the merge from cluster's child
      auto visit_merge_from_cluster_child =
          []([[maybe_unused]] const id_t   &cluster_edge_id,
             const root_chain_cluster_info &cluster_info,
             const id_t                    &merge_into_cluster_edge_id,
             ygm::ygm_ptr<ygm::container::map<
                 supernode_t,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                 chain_map_ptr,
             ygm::ygm_ptr<
                 ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>>
                 root_chain_cluster_edges_map_ptr,
             ygm::ygm_ptr<
                 ygm::container::map<supernode_t, full_leaf_cluster_info>>
                 leaf_cluster_map_ptr) {
            if (cluster_info.child.second == 1) {
              leaf_cluster_map_ptr->async_visit(
                  cluster_info.child, get_leaf_cluster_edges_lambda,
                  merge_into_cluster_edge_id, root_chain_cluster_edges_map_ptr);
            } else {
              chain_map_ptr->async_visit(
                  cluster_info.child, merge_edges_functor(),
                  merge_into_cluster_edge_id, root_chain_cluster_edges_map_ptr,
                  leaf_cluster_map_ptr);
            }
          };
      root_chain_cluster_map.async_visit(
          merge_from, visit_merge_from_cluster_child, merge_to, chain_map_ptr,
          root_chain_cluster_edges_map_ptr, leaf_cluster_map_ptr);
    }
  }
  comm.barrier();

  // Delete the first child of the merge from clusters to save space (if that
  // child has further children, we don't recurse further)
  for (auto &[merge_to, vector_of_merge_from] : local_merge_clusters_into_map) {
    for (id_t merge_from : vector_of_merge_from) {
      auto erase_child =
          []([[maybe_unused]] const id_t   &cluster_edge_id,
             const root_chain_cluster_info &cluster_info,
             ygm::ygm_ptr<ygm::container::map<
                 supernode_t,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                 chain_map_ptr,
             ygm::ygm_ptr<
                 ygm::container::map<supernode_t, full_leaf_cluster_info>>
                 leaf_cluster_map_ptr) {
            if (cluster_info.child.second == 1) {
              leaf_cluster_map_ptr->async_erase(cluster_info.child);
            } else {
              chain_map_ptr->async_erase(cluster_info.child);
            }
          };
      root_chain_cluster_map.async_visit(merge_from, erase_child, chain_map_ptr,
                                         leaf_cluster_map_ptr);
    }
  }
  comm.barrier();

  // Delete the merge from root chain clusters
  for (auto &[merge_to, vector_of_merge_from] : local_merge_clusters_into_map) {
    for (id_t merge_from : vector_of_merge_from) {
      root_chain_cluster_map.async_erase(merge_from);
    }
  }

  // Set the children of the post-merged root chain clusters as valid
  auto set_child_cluster_as_valid_lambda =
      [leaf_cluster_map_ptr, chain_map_ptr](
          const id_t &cluster_edge_id, root_chain_cluster_info &cluster_info) {
        if (cluster_info.child.second == 1) {
          leaf_cluster_map_ptr->async_visit(
              cluster_info.child, [](const supernode_t      &cluster_name,
                                     full_leaf_cluster_info &cluster_info) {
                cluster_info.valid_cluster = true;
              });
        } else {
          chain_map_ptr->async_visit(
              cluster_info.child,
              [](const supernode_t &chain_name,
                 std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                     &chain) {
                auto it                  = chain.first.rbegin();
                it->second.valid_cluster = true;
              });
        }
      };
  root_chain_cluster_map.for_all(set_child_cluster_as_valid_lambda);
  comm.barrier();
}

/**
 * @brief Get cluster size, stability, and first attempt at
 * stability_traversing_up for root chain by processing each chunk of the root
 * chain array locally and then doing a prefix sum return the root chain
 * clusters that have stability. This is a first pass of the calculation for
 * stability traversing up (without correction accounting for selectable
 * root-chain clusters).
 *
 * @param root_chain_array Sorted YGM array of (cluster edge id,
 * root chain cluster info) with valid root chain clusters only.
 * @param root_chain_second_child Information of the second child for the
 * cluster at the bottom of the root chain.
 * @param comm YGM comm
 * @param root_chain_supernode Supernode name of the root chain.
 *
 * @return Vector on each rank of the edge ids for root chain clusters that are
 * candidates for selection. These clusters are ones that have stability >
 * stability traversing up in the first pass calculation. They will need to be
 * further checked/processed to see if they are selected and if their stability
 * traversing up needs to be updated.
 */
std::vector<std::pair<id_t, root_chain_cluster_info>>
calculate_root_chain_size_stability(
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
                             &root_chain_array,
    extra_child_cluster_info &root_chain_second_child, ygm::comm &comm,
    const supernode_t root_chain_supernode) {
  // cluster size = sum of child cluster sizes + num points added
  // stability traversing up = max{stability, sum child stabilities
  // traversing up}

  // We expect few if any root chain clusters to have their own
  // stability be larger than the sum of child stabilities, so we first
  // calculate stability traversing up = sum child stabilities traversing up
  // and then correct this at the end

  // Local sums to get root chain cluster sizes and expected stability
  // traversing up for each array chunk before corrections
  id_t       size = 0;  // local aggregated root chain cluster size
  distance_t stability_traversing_up =
      0.0;  // local aggregated stability traversing up
  for (auto it = root_chain_array.local_begin();
       it != root_chain_array.local_end(); ++it) {
    root_chain_cluster_info cluster_info = it->value.second;

    // If we're at the bottom of the root chain, the starting size and
    // stability_traversing_up is the sum of both child values
    if (it->index == 0) {
      size = root_chain_second_child.size + cluster_info.child_size;
      stability_traversing_up =
          root_chain_second_child.stability_traversing_up +
          cluster_info.child_stability_traversing_up;
    }
    // Otherwise, one of our children is in the root chain
    // Add the non-root chain child size to the aggregate size
    else {
      size += cluster_info.child_size;
      stability_traversing_up += cluster_info.child_stability_traversing_up;
    }

    // Add the number of points added into our cluster size
    size += cluster_info.num_points_added;
    // Update our cluster info with the new size, child size, and initial
    // stability_traversing_up
    it->value.second.size                    = size;
    it->value.second.stability_traversing_up = stability_traversing_up;
  }

  // MPI exclusive scan to get partial sums to correct sizes and initial
  // stability_traversing_up values in each local array chunk
  id_t       correction_size                    = 0;
  distance_t correction_stability_traversing_up = 0.0;
  MPI_Exscan(&size, &correction_size, 1, mpi_id_type(), MPI_SUM,
             comm.get_mpi_comm());
  MPI_Exscan(&stability_traversing_up, &correction_stability_traversing_up, 1,
             mpi_distance_type(), MPI_SUM, comm.get_mpi_comm());

  // Correct all the size and initial stability_traversing_up values on
  // ranks > 0 by adding the Exscan values from the root chain below
  if (comm.rank() > 0) {
    for (auto it = root_chain_array.local_begin();
         it != root_chain_array.local_end(); ++it) {
      it->value.second.size += correction_size;
      it->value.second.stability_traversing_up +=
          correction_stability_traversing_up;
    }
  }

  // Calculate the stability for each root chain cluster
  // Collect any clusters with stability >= stability_traversing_up for
  // possible selection after correcting stability_traversing_up
  // (array index, edge id, cluster info))
  std::vector<std::pair<id_t, root_chain_cluster_info>>
      local_possible_clusters_for_selection;

  for (auto it = root_chain_array.local_begin();
       it != root_chain_array.local_end(); ++it) {
    root_chain_cluster_info cluster_info = it->value.second;
    float                   stability    = cluster_info.lambda_min_edge *
                          (cluster_info.size - cluster_info.num_points_added) +
                      cluster_info.sum_lambda_edges_added -
                      cluster_info.lambda_birth * cluster_info.size;
    it->value.second.stability = stability;
    if (it->value.second.stability >=
        it->value.second.stability_traversing_up) {
      local_possible_clusters_for_selection.push_back(it->value);
    }
  }
  comm.barrier();

  return local_possible_clusters_for_selection;
}

/**
 * @brief Update the root chain cluster map with new root chain info since the
 * root chain array will go out of scope and it will be easier to work with the
 * root_chain_cluster_map directly after this point.
 *
 * @param root_chain_array Sorted YGM array of (cluster edge id,
 * root chain cluster info) with valid root chain clusters only.
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 */
void update_root_chain_cluster_info(
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
        &root_chain_array,
    ygm::container::map<id_t, root_chain_cluster_info>
        &root_chain_cluster_map) {
  ygm::comm &comm = root_chain_cluster_map.comm();

  auto overwrite_cluster_info =
      []([[maybe_unused]] const id_t   &cluster_edge_id,
         root_chain_cluster_info       &cluster_info,
         const root_chain_cluster_info &new_cluster_info) {
        cluster_info = new_cluster_info;
      };
  for (auto it = root_chain_array.local_begin();
       it != root_chain_array.local_end(); ++it) {
    root_chain_cluster_map.async_visit(it->value.first, overwrite_cluster_info,
                                       it->value.second);
  }
  comm.barrier();
}

/**
 * @brief Iteratively corrects the stability traversing up for root chain
 * clusters that are candidates for selection and finds the root chain cluster
 * to select.
 *
 * @details If there are root chain clusters that are candidates for selection,
 * then we will find exactly one root chain cluster to select. This function
 * iteratively corrects the stability traversing up for root chain clusters that
 * are candidates for selection and finds that root chain cluster to select. To
 * save time, we do not correct the stability traversing up for root chain
 * clusters that cannot be selected. Furthermore, we don't bother correcting the
 * stability_traversing_up values in the root chain cluster map.
 *
 * @param possible_clusters_for_selection_array Sorted YGM array of possible
 * root chain clusters (cluster edge id, root chain cluster info) for selection.
 * @param comm YGM comm.
 * @param max_cluster_edge_id The maximum cluster edge id in the root chain.
 * This cluster at the top of the root chain represents assigning all points to
 * the same cluster and we do not consider this case. We stop calculation if we
 * reach this cluster.
 *
 * @returns Pair (selected_root_chain_cluster_edge_id,
 * num_correction_iterations). The first entry is the cluster edge id of the
 * selected root chain cluster. The second entry is the number of iteration
 * passes we needed to correct the stability traversing up and find the root
 * chain cluster to select.
 */
std::pair<id_t, int> correct_root_chain_stability_traversing_up(
    ygm::container::array<std::pair<id_t, root_chain_cluster_info>>
              &possible_clusters_for_selection_array,
    ygm::comm &comm, id_t max_cluster_edge_id) {
  id_t selected_root_chain_cluster_edge_id = 0;
  int  num_correction_iterations           = 0;
  while (num_correction_iterations <
         possible_clusters_for_selection_array.size()) {
    // Placeholder value larger than any possible edge id in root chain
    id_t lowest_cluster_to_correct_idx =
        max_cluster_edge_id;  // index in
                              // possible_clusters_for_selection_array
    id_t lowest_cluster_to_correct_edge_id =
        max_cluster_edge_id;  // edge id of the lowest root chain cluster
                              // to correct

    // For each rank, get the smallest cluster ID we still need to correct
    for (auto it = possible_clusters_for_selection_array.local_begin();
         it != possible_clusters_for_selection_array.local_end(); ++it) {
      root_chain_cluster_info cluster_info = it->value.second;
      if (cluster_info.stability > cluster_info.stability_traversing_up) {
        lowest_cluster_to_correct_idx     = it->index;
        lowest_cluster_to_correct_edge_id = it->value.first;
        break;
      }
    }

    // Get overall min index for cluster to correct
    MPI_Allreduce(MPI_IN_PLACE, &lowest_cluster_to_correct_idx, 1,
                  mpi_id_type(), MPI_MIN, comm.get_mpi_comm());
    MPI_Allreduce(MPI_IN_PLACE, &lowest_cluster_to_correct_edge_id, 1,
                  mpi_id_type(), MPI_MIN, comm.get_mpi_comm());

    // If there's no more clusters to correct, or we have the top cluster
    // to correct, then stop. We don't want to select the top root chain
    // cluster, which puts all points into a single cluster
    if (lowest_cluster_to_correct_edge_id >= max_cluster_edge_id) {
      break;
    }

    // Otherwise, proceed in correcting the stability traversing up for
    // the cluster at lowest_cluster_to_correct and propagate this change
    // up the root chain
    ++num_correction_iterations;
    selected_root_chain_cluster_edge_id = lowest_cluster_to_correct_edge_id;

    // Correction to stability traversing up to propagate to clusters
    // above
    static distance_t stability_correction;
    stability_correction = 0.0;

    // Visit the lowest correct and set its stability_traversing_up to its
    // stability
    // Also calculate the correction its (stability - old
    // stability_traversing_up) to propagate up to clusters above
    if (comm.rank() == 0) {
      possible_clusters_for_selection_array.async_visit(
          lowest_cluster_to_correct_idx,
          []([[maybe_unused]] const id_t              &index,
             std::pair<id_t, root_chain_cluster_info> &value) {
            stability_correction =
                value.second.stability - value.second.stability_traversing_up;
            // std::cout << "Visiting root chain cluster " << value.first
            //           << " with size = " << value.second.size
            //           << ", stability = " << value.second.stability
            //           << " and stability traversing up = "
            //           << value.second.stability_traversing_up
            //           << ", giving correction factor = " <<
            //           stability_correction
            //           << std::endl;  // debug
            value.second.stability_traversing_up = value.second.stability;
          });
    }
    comm.barrier();

    // Make sure all ranks have the correction factor
    MPI_Allreduce(MPI_IN_PLACE, &stability_correction, 1, mpi_distance_type(),
                  MPI_MAX, comm.get_mpi_comm());

    // Correct the stability traversing up of all clusters that we have to
    // check above the lowest_cluster_to_correct_idx
    id_t first_local_index =
        possible_clusters_for_selection_array.local_begin()->index;
    id_t last_local_index = first_local_index +
                            possible_clusters_for_selection_array.local_size() -
                            1;
    if (lowest_cluster_to_correct_idx + 1 < last_local_index) {
      for (auto it = possible_clusters_for_selection_array.local_begin();
           it != possible_clusters_for_selection_array.local_end(); ++it) {
        if (it->index > lowest_cluster_to_correct_idx) {
          it->value.second.stability_traversing_up += stability_correction;
        }
      }
    }
    comm.barrier();
  }

  return std::make_pair(selected_root_chain_cluster_edge_id,
                        num_correction_iterations);
}

/**
 * @brief Start at the root chain and traverse down the cluster hierarchy. Mark
 * selected clusters in their respective map (chain_map or leaf_cluster_map).
 *
 * @details For each root chain cluster with edge id greater than the
 * selected_root_chain_edge_id (if there is one), visit its non-root chain child
 * cluster. A cluster is selected if it is valid and its stability is >= the sum
 * of its child stabilities traversing up. If a cluster is not selected, then
 * visit its two children and continue traversal. A valid leaf cluster will
 * always have stability = stability traversing up (since it has no children)
 * and will be selected if reached.
 *
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param root_chain_second_child Information on the second child for the bottom
 * root chain cluster.
 * @param _min_cluster_size HDBSCAN min cluster size parameter
 * @param _root_chain_supernode Name of the root chain supernode.
 * @param _selected_root_chain_cluster_edge_id Edge ID of the selected root
 * chain cluster (if there is one). If none is selected, this should be 0.
 *
 * @returns Vector of clusters (represented as (supernode, edge_id)) selected on
 * each rank.
 */
std::vector<cluster_name_t> traverse_down_hierarchy_and_select_clusters(
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>    &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    extra_child_cluster_info root_chain_second_child,
    const int _min_cluster_size, const supernode_t _root_chain_supernode,
    const id_t _selected_root_chain_cluster_edge_id) {
  ygm::comm &comm                 = root_chain_cluster_map.comm();
  auto       chain_map_ptr        = chain_map.get_ygm_ptr();
  auto       leaf_cluster_map_ptr = leaf_cluster_map.get_ygm_ptr();

  // Make some variables static within this function for easy access
  static uint32_t min_cluster_size;
  min_cluster_size = _min_cluster_size;
  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;
  static id_t selected_root_chain_cluster_edge_id;
  selected_root_chain_cluster_edge_id = _selected_root_chain_cluster_edge_id;

  // Collect names of local selected clusters to return
  static std::vector<cluster_name_t> local_selected_clusters;
  local_selected_clusters.clear();

  // Walking down the cluster hierarchy and visiting children until we
  // have all selected clusters can be expressed as a recursive
  // operation
  struct select_clusters_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<
            supernode_t,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                           chain_map_ptr,
        const supernode_t &chain_name,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
        ygm::ygm_ptr<ygm::container::map<supernode_t, full_leaf_cluster_info>>
            leaf_cluster_map_ptr) {
      // Traverse down the chain (traverse the cluster map in reverse)
      for (auto it = chain.first.rbegin(); it != chain.first.rend(); ++it) {
        id_t              cluster_edge_id = it->first;
        full_cluster_info cluster_info    = it->second;

        // If the cluster's stability is >= the sum of child stabilities
        // (traversing up), then select this cluster
        if (cluster_info.valid_cluster &&
            cluster_info.stability >= cluster_info.stability_traversing_up) {
          it->second.selected = true;
          cluster_name_t cluster_name =
              std::make_pair(chain_name, cluster_edge_id);
          local_selected_clusters.push_back(cluster_name);
          break;  // stop traversing the chain
        }
        // Otherwise, visit this cluster's non-chain child and also
        // continue traversing down the chain
        else {
          if (cluster_info.child_size >= min_cluster_size) {
            // Select the child if it's a leaf cluster
            if (cluster_info.child.second == 1) {
              leaf_cluster_map_ptr->async_visit(
                  cluster_info.child, [](const supernode_t      &cluster_name,
                                         full_leaf_cluster_info &cluster_info) {
                    cluster_info.selected = true;
                  });
              local_selected_clusters.push_back(
                  std::make_pair(cluster_info.child, 0));

            }
            // Otherwise visit the child chain and continue traversal
            else {
              chain_map_ptr->async_visit(cluster_info.child,
                                         select_clusters_functor(),
                                         leaf_cluster_map_ptr);
            }
          }

          // If we're at the bottom of the chain, visit the second chain
          // child also
          auto it_next = std::next(it);
          if (it_next == chain.first.rend()) {
            supernode_t child = chain.second.children[1];
            if (chain.second.min_edge_child_size2 >= min_cluster_size) {
              // Select the child if it's a leaf cluster
              if (child.second == 1) {
                leaf_cluster_map_ptr->async_visit(
                    child, [](const supernode_t      &cluster_name,
                              full_leaf_cluster_info &cluster_info) {
                      cluster_info.selected = true;
                    });
              }
              // Otherwise visit the child chain and continue traversal
              else {
                chain_map_ptr->async_visit(child, select_clusters_functor(),
                                           leaf_cluster_map_ptr);
              }
            }
          }
        }
      }
    }
  };

  // Traverse down the child chains for cluster selection from all root
  // chain clusters above the selected one (if there is one)
  auto traverse_from_root_chain_lambda =
      [chain_map_ptr, leaf_cluster_map_ptr](
          const id_t                    &cluster_edge_id,
          const root_chain_cluster_info &cluster_info) {
        if (cluster_edge_id > selected_root_chain_cluster_edge_id) {
          // Select the child if it's a leaf cluster - note that leaf
          // clusters off the root chain are all valid since we merged
          // away invalid root chain clusters
          if (cluster_info.child.second == 1) {
            leaf_cluster_map_ptr->async_visit(
                cluster_info.child, [](const supernode_t      &cluster_name,
                                       full_leaf_cluster_info &cluster_info) {
                  cluster_info.selected = true;
                });
            local_selected_clusters.push_back(
                std::make_pair(cluster_info.child, 0));
          }
          // Otherwise visit the child chain and continue traversal
          else {
            chain_map_ptr->async_visit(cluster_info.child,
                                       select_clusters_functor(),
                                       leaf_cluster_map_ptr);
          }
        }
      };
  root_chain_cluster_map.for_all(traverse_from_root_chain_lambda);

  // If no root chain cluster is selected, also traverse down to its
  // second child
  if (selected_root_chain_cluster_edge_id == 0) {
    if (comm.rank() == 0) {
      if (root_chain_second_child.name.second == 1) {
        leaf_cluster_map_ptr->async_visit(
            root_chain_second_child.name,
            [](const supernode_t      &cluster_name,
               full_leaf_cluster_info &cluster_info) {
              cluster_info.selected = true;
            });
        local_selected_clusters.push_back(
            std::make_pair(root_chain_second_child.name, 0));
      }
      // Otherwise visit the child chain and continue traversal
      else {
        chain_map_ptr->async_visit(root_chain_second_child.name,
                                   select_clusters_functor(),
                                   leaf_cluster_map_ptr);
      }
    }
  }

  comm.barrier();

  return local_selected_clusters;
}

/**
 * @brief Assign consecutive ids to selected clusters and then label points with
 * their cluster id. Points that are not in a selected cluster are labeled as
 * noise.
 *
 * @details Takes in the local_selected_clusters and labels points in those
 * clusters on this rank. An MPI prefix sum call should be used first to provide
 * a start_cluster_id for each rank to start enumerating clusters so that
 * clusters get consecutive non-overlapping labels in the processing.
 *
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 * @param root_chain_cluster_edges_map YGM map of root chain cluster edge id ->
 * vector of edges added to this cluster.
 * @param point_to_cluster_id_map An empty YGM map of point id -> cluster label.
 * @param edge_endpoints_map YGM map of edge id -> pair of point ids of its
 * endpoints.
 * @param local_selected_clusters Vector of selected clusters to process on this
 * rank.
 * @param start_cluster_id The first id to enumerate clusters on this rank.
 * @param root_chain_supernode Name of the root chain supernode. We use this to
 * figure out if a selected cluster is in the root chain or not.
 * @param root_chain_second_child_name Name of the second child of the bottom
 * root chain cluster. We use this if traversing down from a selected root chain
 * cluster and assigning all points below its cluster id.
 */
void assign_points_cluster_ids(
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>          &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<id_t, std::vector<edge_id_with_dist_t>>
                                            &root_chain_cluster_edges_map,
    ygm::container::map<id_t, cluster_id_t> &point_to_cluster_id_map,
    ygm::container::map<id_t, std::pair<id_t, id_t>> &edge_endpoints_map,
    const std::vector<cluster_name_t>                &local_selected_clusters,
    const cluster_id_t start_cluster_id, const supernode_t root_chain_supernode,
    const supernode_t root_chain_second_child_name) {
  ygm::comm &comm                   = chain_map.comm();
  auto       edge_endpoints_map_ptr = edge_endpoints_map.get_ygm_ptr();
  auto point_to_cluster_id_map_ptr  = point_to_cluster_id_map.get_ygm_ptr();
  auto leaf_cluster_map_ptr         = leaf_cluster_map.get_ygm_ptr();
  auto chain_map_ptr                = chain_map.get_ygm_ptr();
  auto root_chain_cluster_edges_map_ptr =
      root_chain_cluster_edges_map.get_ygm_ptr();

  // All points in the data set start out as noise points and when we
  // pick clusters, this will be over written with the cluster id
  edge_endpoints_map.for_all(
      [&point_to_cluster_id_map]([[maybe_unused]] const id_t &edge_id,
                                 const std::pair<id_t, id_t> &edge_endpoints) {
        point_to_cluster_id_map.async_insert(edge_endpoints.first,
                                             NOISE_POINT_LABEL);
        point_to_cluster_id_map.async_insert(edge_endpoints.second,
                                             NOISE_POINT_LABEL);
      });
  comm.barrier();

  // Labels the endpoints of an edge with the provided cluster id
  static auto label_edge_endpoints_lambda =
      []([[maybe_unused]] const id_t &edge_id,
         const std::pair<id_t, id_t> &edge_endpoints,
         const cluster_id_t &cluster_id, auto point_to_cluster_id_map_ptr) {
        auto set_point_label_lambda = []([[maybe_unused]] const id_t &point_id,
                                         cluster_id_t       &cluster_id,
                                         const cluster_id_t &new_cluster_id) {
          cluster_id = new_cluster_id;
        };

        point_to_cluster_id_map_ptr->async_visit(
            edge_endpoints.first, set_point_label_lambda, cluster_id);
        point_to_cluster_id_map_ptr->async_visit(
            edge_endpoints.second, set_point_label_lambda, cluster_id);
      };

  // Goes through all edges in a leaf cluster and labels their points with the
  // provided cluster id
  static auto label_leaf_cluster_edge_endpoints_lambda =
      [](const supernode_t            &cluster_name,
         const full_leaf_cluster_info &cluster_info,
         const cluster_id_t &cluster_id, auto edge_endpoints_map_ptr,
         auto point_to_cluster_id_map_ptr) {
        for (const edge_id_with_dist_t &edge : cluster_info.edges) {
          edge_endpoints_map_ptr->async_visit(
              edge.first, label_edge_endpoints_lambda, cluster_id,
              point_to_cluster_id_map_ptr);
        }
      };

  // Goes through all edges added to a root chain cluster and labels their
  // points with the provided cluster id
  static auto label_root_chain_cluster_edge_endpoints_lambda =
      [](const id_t &cluster_edge_id, std::vector<edge_id_with_dist_t> &edges,
         const cluster_id_t &cluster_id, auto edge_endpoints_map_ptr,
         auto point_to_cluster_id_map_ptr) {
        for (const edge_id_with_dist_t &edge : edges) {
          edge_endpoints_map_ptr->async_visit(
              edge.first, label_edge_endpoints_lambda, cluster_id,
              point_to_cluster_id_map_ptr);
        }
      };

  // Labeling the points in a cluster involves visiting the cluster and
  // all its children. During these visits, all edges in each cluster
  // get their endpoints assigned the desired cluster label. We express
  // this as a recursive operation
  struct label_points_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<
            supernode_t,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                           chain_map_ptr,
        const supernode_t &chain_name,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
        const id_t &starting_cluster_edge_id, const cluster_id_t &cluster_id,
        ygm::ygm_ptr<ygm::container::map<supernode_t, full_leaf_cluster_info>>
            leaf_cluster_map_ptr,
        ygm::ygm_ptr<ygm::container::map<id_t, std::pair<id_t, id_t>>>
            edge_endpoints_map_ptr,
        ygm::ygm_ptr<ygm::container::map<id_t, cluster_id_t>>
            point_to_cluster_id_map_ptr) {
      // For all clusters in the chain below or equal to the provided starting
      // cluster edge id. If the whole chain should be labeled, provide starting
      // cluster edge id equal to std::numeric_limits<id_t>::max()
      for (auto it = chain.first.begin(); it != chain.first.end(); ++it) {
        if (it->first > starting_cluster_edge_id) {
          break;
        } else {
          full_cluster_info cluster_info = it->second;

          // Set this cluster's label as the provided label
          it->second.cluster_id = cluster_id;

          // Label each edge in the cluster with the provided label
          for (const edge_id_with_dist_t &edge : cluster_info.edges) {
            edge_endpoints_map_ptr->async_visit(
                edge.first, label_edge_endpoints_lambda, cluster_id,
                point_to_cluster_id_map_ptr);
          }

          // Visit this cluster's non-chain child and propagate the cluster
          // label
          if (cluster_info.child.second == 1) {
            leaf_cluster_map_ptr->async_visit(
                cluster_info.child, label_leaf_cluster_edge_endpoints_lambda,
                cluster_id, edge_endpoints_map_ptr,
                point_to_cluster_id_map_ptr);
          } else {
            chain_map_ptr->async_visit(
                cluster_info.child, label_points_functor(),
                std::numeric_limits<id_t>::max(), cluster_id,
                leaf_cluster_map_ptr, edge_endpoints_map_ptr,
                point_to_cluster_id_map_ptr);
          }
        }
      }

      // Also visit the second chain child and propagate the label
      if (chain.second.children[1].second == 1) {
        leaf_cluster_map_ptr->async_visit(
            chain.second.children[1], label_leaf_cluster_edge_endpoints_lambda,
            cluster_id, edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);
      } else {
        chain_map_ptr->async_visit(
            chain.second.children[1], label_points_functor(),
            std::numeric_limits<id_t>::max(), cluster_id, leaf_cluster_map_ptr,
            edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);
      }
    }
  };

  for (uint32_t i = 0; i < local_selected_clusters.size(); ++i) {
    const cluster_name_t cluster_name = local_selected_clusters.at(i);
    const cluster_id_t   cluster_id   = start_cluster_id + i;

    // If the selected cluster is a leaf cluster, directly go label its edges
    if (cluster_name.first.second == 1) {
      leaf_cluster_map.async_visit(
          cluster_name.first, label_leaf_cluster_edge_endpoints_lambda,
          cluster_id, edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);
    }
    // If the selected cluster is in the root chain, handle processing it
    // separately
    else if (cluster_name.first == root_chain_supernode) {
      id_t starting_cluster_edge_id = cluster_name.second;
      auto label_root_chain_cluster_lambda =
          [starting_cluster_edge_id, cluster_id, chain_map_ptr,
           leaf_cluster_map_ptr, root_chain_cluster_edges_map_ptr,
           edge_endpoints_map_ptr,
           point_to_cluster_id_map_ptr](const id_t &cluster_edge_id,
                                        root_chain_cluster_info &cluster_info) {
            // If this root chain cluster is the selected cluster or below it,
            // then label it
            if (cluster_edge_id <= starting_cluster_edge_id) {
              // Set this cluster's label
              cluster_info.cluster_id = cluster_id;

              // Label this cluster's edges added
              root_chain_cluster_edges_map_ptr->async_visit(
                  cluster_edge_id,
                  label_root_chain_cluster_edge_endpoints_lambda, cluster_id,
                  edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);

              // Visit this cluster's non-root-chain child and label it
              // If its a leaf cluster, just label it directly
              if (cluster_info.child.second == 1) {
                leaf_cluster_map_ptr->async_visit(
                    cluster_info.child,
                    label_leaf_cluster_edge_endpoints_lambda, cluster_id,
                    edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);
              }
              // If its a chain, go visit it with the label points functor
              else {
                chain_map_ptr->async_visit(
                    cluster_info.child, label_points_functor(),
                    std::numeric_limits<id_t>::max(), cluster_id,
                    leaf_cluster_map_ptr, edge_endpoints_map_ptr,
                    point_to_cluster_id_map_ptr);
              }
            }
          };

      root_chain_cluster_map.for_all(label_root_chain_cluster_lambda);

      // Also handle labeling the second root chain child
      if (root_chain_second_child_name.second == 1) {
        leaf_cluster_map.async_visit(root_chain_second_child_name,
                                     label_leaf_cluster_edge_endpoints_lambda,
                                     cluster_id, edge_endpoints_map_ptr,
                                     point_to_cluster_id_map_ptr);
      } else {
        chain_map.async_visit(
            root_chain_second_child_name, label_points_functor(),
            std::numeric_limits<id_t>::max(), cluster_id, leaf_cluster_map_ptr,
            edge_endpoints_map_ptr, point_to_cluster_id_map_ptr);
      }
    }
    // If the selected cluster is in a chain, use the label points functor
    else {
      chain_map.async_visit(cluster_name.first, label_points_functor(),
                            cluster_name.second, cluster_id,
                            leaf_cluster_map_ptr, edge_endpoints_map_ptr,
                            point_to_cluster_id_map_ptr);
    }
  }

  comm.barrier();
  return;
}

/**
 * @brief Writes information of all clusters to file, disregarding
 * min_cluster_size.
 *
 * @param comm YGM comm.
 * @param output_dir Output directory for YGM multi_output to write files.
 * @param _cluster_file_name Filename prefix for output files.
 * @param file_number File number to add to the filename for this rank.
 * @param root_chain_min_edge_id The smallest cluster edge id in the root chain.
 * We write the second root chain child info to that cluster.
 * @param root_chain_second_child Info of the second child for the bottom root
 * chain cluster.
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 */
void write_all_clusters_to_file_including_invalid_clusters(
    ygm::comm &comm, std::string output_dir, std::string _cluster_file_name,
    int file_number, supernode_t _root_chain_supernode,
    id_t                                                root_chain_min_edge_id,
    extra_child_cluster_info                           &root_chain_second_child,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>    &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info>
        &leaf_cluster_map) {
  ygm::io::multi_output mo_clusters(comm, output_dir);

  static std::string cluster_file_name;
  cluster_file_name = _cluster_file_name;
  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;

  // Output cluster data
  if (comm.rank() == file_number) {
    std::string header =
        "cluster_name,size,stability,stability_traversing_up,"
        "selected,cluster_id,parent_cluster_name,birth_distance,"
        "min_edge_distance,non_chain_child1,non_chain_child2,valid_cluster";
    mo_clusters.async_write_line(cluster_file_name, header);
  }
  comm.barrier();

  // Write root chain clusters
  auto write_root_chain_clusters_lambda =
      [&mo_clusters, &root_chain_second_child, &root_chain_min_edge_id](
          const id_t                    &cluster_edge_id,
          const root_chain_cluster_info &cluster_info) {
        std::stringstream ss;
        ss << "\"" << std::make_pair(root_chain_supernode, cluster_edge_id)
           << "\"," << cluster_info.size << "," << cluster_info.stability << ","
           << cluster_info.stability_traversing_up << ","
           << cluster_info.selected << "," << cluster_info.cluster_id << ","
           << "\""
           << std::make_pair(root_chain_supernode, cluster_info.parent_edge_id)
           << "\""
           << "," << 1.0 / cluster_info.lambda_birth << ","
           << 1.0 / cluster_info.lambda_min_edge << "," << "\""
           << cluster_info.child << "\"" << ",";
        if (cluster_edge_id == root_chain_min_edge_id) {
          ss << "\"" << root_chain_second_child.name << "\"";
        }
        ss << ",1";
        mo_clusters.async_write_line(cluster_file_name, ss.str());
      };
  root_chain_cluster_map.for_all(write_root_chain_clusters_lambda);
  comm.barrier();

  // Write chain clusters
  auto write_chain_clusters_lambda =
      [&mo_clusters](
          const supernode_t &chain_name,
          const std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
              &chain) {
        for (auto it = chain.first.begin(); it != chain.first.end(); ++it) {
          id_t              cluster_edge_id = it->first;
          full_cluster_info cluster_info    = it->second;

          if (cluster_info.valid_cluster) {
            std::stringstream ss;

            std::pair<supernode_t, id_t> parent_name;
            auto                         next_it = std::next(it);
            if (next_it == chain.first.end()) {
              parent_name = std::make_pair(chain.second.parent_chain,
                                           chain.second.parent_edge_id);
            } else {
              parent_name = std::make_pair(chain_name, next_it->first);
            }

            ss << "\"" << std::make_pair(chain_name, cluster_edge_id) << "\","
               << cluster_info.size << "," << cluster_info.stability << ","
               << cluster_info.stability_traversing_up << ","
               << cluster_info.selected << "," << cluster_info.cluster_id << ","
               << "\"" << parent_name << "\""
               << "," << cluster_info.birth_distance << ","
               << cluster_info.min_edge.second << "," << "\""
               << cluster_info.child << "\"" << ",";
            if (it == chain.first.begin()) {
              ss << "\"" << chain.second.children[1] << "\"";
            }
            ss << "," << cluster_info.valid_cluster;
            mo_clusters.async_write_line(cluster_file_name, ss.str());
          }
        }
      };
  chain_map.for_all(write_chain_clusters_lambda);
  comm.barrier();

  // Write leaf clusters
  auto write_leaf_clusters_lambda =
      [&mo_clusters](const supernode_t            &cluster_name,
                     const full_leaf_cluster_info &cluster_info) {
        if (cluster_info.valid_cluster) {
          std::stringstream ss;
          ss << "\"" << std::make_pair(cluster_name, 0) << "\"" << ","
             << cluster_info.edges.size() + 1 << "," << cluster_info.stability
             << "," << cluster_info.stability << "," << cluster_info.selected
             << "," << cluster_info.cluster_id << ","
             << "\""
             << std::make_pair(cluster_info.parent_chain,
                               cluster_info.parent_edge_id)
             << "\""
             << "," << cluster_info.birth_distance << ",,,,"
             << cluster_info.valid_cluster;
          mo_clusters.async_write_line(cluster_file_name, ss.str());
        }
      };
  leaf_cluster_map.for_all(write_leaf_clusters_lambda);
  comm.barrier();
}

/**
 * @brief Go through all clusters and get valid clusters. Store valid clusters
 * and their information in the valid_clusters_map. Erase invalid clusters from
 * root_chain_cluster_map, chain_map, and leaf_cluster_map.
 *
 * @param valid_cluster_map An empty YGM map of valid cluster name -> valid
 * cluster info.
 * @param root_chain_min_edge_id The smallest cluster edge id in the root chain.
 * We write the second root chain child info to that cluster.
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param _root_chain_supernode Name of the root chain supernode.
 */
void get_and_keep_valid_clusters_only(
    ygm::container::map<cluster_name_t, full_valid_cluster_info>
                                                       &valid_cluster_map,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>    &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    supernode_t _root_chain_supernode) {
  ygm::comm         &comm = valid_cluster_map.comm();
  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;

  // Go through the leaf clusters and add valid ones to the valid cluster
  // map and delete invalid ones
  auto leaf_cluster_map_ptr    = leaf_cluster_map.get_ygm_ptr();
  auto get_valid_leaf_clusters = [&valid_cluster_map, leaf_cluster_map_ptr](
                                     const supernode_t &leaf_cluster_supernode,
                                     full_leaf_cluster_info &cluster_info) {
    if (cluster_info.valid_cluster) {
      cluster_name_t cluster_name = std::make_pair(leaf_cluster_supernode, 0);
      full_valid_cluster_info valid_cluster_info{
          .birth_distance          = cluster_info.birth_distance,
          .stability               = cluster_info.stability,
          .stability_traversing_up = cluster_info.stability,
          .size                    = cluster_info.edges.size() + 1,
          .cluster_id              = cluster_info.cluster_id,
          .selected                = cluster_info.selected,
          .num_points_added        = cluster_info.edges.size() + 1};
      valid_cluster_map.async_insert(cluster_name, valid_cluster_info);
    } else {
      leaf_cluster_map_ptr->async_erase(leaf_cluster_supernode);
    }
  };
  leaf_cluster_map.for_all(get_valid_leaf_clusters);
  comm.barrier();

  // Go through the chains and add valid ones to the valid cluster map
  // and delete invalid ones
  auto get_valid_chain_clusters =
      [&valid_cluster_map](
          const supernode_t &chain_name,
          std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
              &chain) {
        // First erase the invalid clusters
        for (auto it = chain.first.begin(); it != chain.first.end();) {
          if (!it->second.valid_cluster) {
            chain.first.erase(
                it++);  // Safe increment and erase invalid clusters
          } else {
            ++it;
          }
        }

        // Now go through the map again and add the valid clusters to
        // valid_cluster_map, with correct within-chain parent/child
        // relationships
        for (auto it = chain.first.begin(); it != chain.first.end(); ++it) {
          id_t              cluster_edge_id = it->first;
          full_cluster_info cluster_info    = it->second;
          cluster_name_t    cluster_name =
              std::make_pair(chain_name, cluster_edge_id);

          std::array<cluster_name_t, 2> children{BLANK_CLUSTER_NAME,
                                                 BLANK_CLUSTER_NAME};
          if (it != chain.first.begin()) {
            auto prev_it = std::prev(it);
            children[0]  = std::make_pair(chain_name, prev_it->first);
          }

          cluster_name_t parent  = BLANK_CLUSTER_NAME;
          auto           next_it = std::next(it);
          if (next_it != chain.first.end()) {
            parent = std::make_pair(chain_name, next_it->first);
          }

          full_valid_cluster_info valid_cluster_info{
              .children                = children,
              .parent                  = parent,
              .birth_distance          = cluster_info.birth_distance,
              .stability               = cluster_info.stability,
              .stability_traversing_up = cluster_info.stability_traversing_up,
              .size                    = cluster_info.size,
              .cluster_id              = cluster_info.cluster_id,
              .selected                = cluster_info.selected,
              .num_points_added        = cluster_info.edges.size() - 1};
          valid_cluster_map.async_insert(cluster_name, valid_cluster_info);
        }
      };

  chain_map.for_all(get_valid_chain_clusters);
  comm.barrier();

  // Root chain clusters are all valid, just transfer them to the valid cluster
  // map
  auto get_valid_root_chain_clusters =
      [&valid_cluster_map](const id_t                    &cluster_edge_id,
                           const root_chain_cluster_info &cluster_info) {
        cluster_name_t cluster_name =
            std::make_pair(root_chain_supernode, cluster_edge_id);
        full_valid_cluster_info valid_cluster_info{
            .parent                  = std::make_pair(root_chain_supernode,
                                                      cluster_info.parent_edge_id),
            .birth_distance          = 1.0 / cluster_info.lambda_birth,
            .stability               = cluster_info.stability,
            .stability_traversing_up = cluster_info.stability_traversing_up,
            .size                    = cluster_info.size,
            .cluster_id              = cluster_info.cluster_id,
            .selected                = cluster_info.selected,
            .num_points_added        = cluster_info.num_points_added};
        valid_cluster_map.async_insert(cluster_name, valid_cluster_info);
      };
  root_chain_cluster_map.for_all(get_valid_root_chain_clusters);
  comm.barrier();

  // Set the within-root-chain children
  auto set_root_chain_child_clusters =
      [&valid_cluster_map](const id_t                    &cluster_edge_id,
                           const root_chain_cluster_info &cluster_info) {
        // If we're not the top of the root chain
        if (cluster_info.parent_edge_id != 0) {
          cluster_name_t parent_cluster_name =
              std::make_pair(root_chain_supernode, cluster_info.parent_edge_id);
          cluster_name_t child_cluster_name =
              std::make_pair(root_chain_supernode, cluster_edge_id);
          valid_cluster_map.async_visit(
              child_cluster_name,
              []([[maybe_unused]] const cluster_name_t &cluster_name,
                 full_valid_cluster_info               &cluster_info,
                 const cluster_name_t                  &parent) {
                cluster_info.parent = parent;
              },
              parent_cluster_name);
          valid_cluster_map.async_visit(
              parent_cluster_name,
              []([[maybe_unused]] const cluster_name_t &cluster_name,
                 full_valid_cluster_info               &cluster_info,
                 const cluster_name_t                  &child) {
                if (cluster_info.children[0] == BLANK_CLUSTER_NAME) {
                  cluster_info.children[0] = child;
                } else {
                  cluster_info.children[1] = child;
                }
              },
              child_cluster_name);
        }
      };
  root_chain_cluster_map.for_all(set_root_chain_child_clusters);
  comm.barrier();
}

/**
 * @brief Fill in the valid parent/child clusters of clusters in the
 * valid_cluster_map. Do this by going through chains by their contraction round
 * and send valid clusters up the hierarchy to find their valid parent cluster.
 *
 * @param valid_cluster_map An empty YGM map of valid cluster name -> valid
 * cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 * @param _root_chain_supernode Name of the root chain supernode.
 */
void get_valid_cluster_parent_child_relations(
    ygm::container::map<cluster_name_t, full_valid_cluster_info>
                                                             &valid_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>          &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info> &leaf_cluster_map,
    supernode_t _root_chain_supernode) {
  ygm::comm &comm                  = valid_cluster_map.comm();
  auto       valid_cluster_map_ptr = valid_cluster_map.get_ygm_ptr();
  auto       chain_map_ptr         = chain_map.get_ygm_ptr();

  static supernode_t root_chain_supernode;
  root_chain_supernode = _root_chain_supernode;
  static int final_round;
  final_round = root_chain_supernode.second - 1;

  static auto set_parent =
      []([[maybe_unused]] const cluster_name_t &cluster_name,
         full_valid_cluster_info               &cluster_info,
         const cluster_name_t &parent) { cluster_info.parent = parent; };
  static auto set_child =
      []([[maybe_unused]] const cluster_name_t &cluster_name,
         full_valid_cluster_info &cluster_info, const cluster_name_t &child) {
        if (cluster_info.children[0] == BLANK_CLUSTER_NAME) {
          cluster_info.children[0] = child;
        } else {
          cluster_info.children[1] = child;
        }
      };

  // Walking up the valid cluster hierarchy and sending child up to find its
  // valid cluster parent can be expressed as a recursive operation
  struct send_valid_cluster_to_valid_parent_functor {
    void operator()(
        ygm::ygm_ptr<ygm::container::map<
            supernode_t,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>>>
                           chain_map_ptr,
        const supernode_t &chain_name,
        std::pair<std::map<id_t, full_cluster_info>, full_chain_info> &chain,
        const cluster_name_t &child_cluster_name, const id_t &parent_edge_id,
        ygm::ygm_ptr<
            ygm::container::map<cluster_name_t, full_valid_cluster_info>>
            valid_cluster_map_ptr) {
      // Find the first chain map key <= parent edge id
      auto it = chain.first.lower_bound(parent_edge_id);

      // If this chain has no valid clusters, or we didn't find a cluster in the
      // chain above our parent edge id, send ourselves on to the parent chain
      // to continue searching for a parent
      if (chain.first.size() == 0 || it == chain.first.end()) {
        // If the parent chain is the root chain, then we know its valid parent
        // cluster and set it
        if (chain.second.parent_chain == root_chain_supernode) {
          cluster_name_t parent_cluster_name =
              std::make_pair(root_chain_supernode, chain.second.parent_edge_id);
          // Set the parent of the child cluster
          valid_cluster_map_ptr->async_visit(child_cluster_name, set_parent,
                                             parent_cluster_name);
          // Set the child of the parent cluster
          valid_cluster_map_ptr->async_visit(parent_cluster_name, set_child,
                                             child_cluster_name);
        }
        // Otherwise, we are now looking for the parent cluster for this chain,
        // so we visit the parent chain and send along the parent edge id of the
        // chain
        chain_map_ptr->async_visit(
            chain.second.parent_chain,
            send_valid_cluster_to_valid_parent_functor(), child_cluster_name,
            chain.second.parent_edge_id, valid_cluster_map_ptr);

      }
      // Otherwise, if find which parent cluster in this chain, set it as the
      // parent and ourselves as the child
      else {
        id_t           cluster_edge_id = it->first;
        cluster_name_t parent_cluster_name =
            std::make_pair(chain_name, cluster_edge_id);
        // Set the parent of the child cluster
        valid_cluster_map_ptr->async_visit(child_cluster_name, set_parent,
                                           parent_cluster_name);
        // Set the child of the parent cluster
        valid_cluster_map_ptr->async_visit(parent_cluster_name, set_child,
                                           child_cluster_name);
      }
    }
  };

  // Start at valid leaf clusters
  auto send_valid_leaf_clusters_to_parent =
      [valid_cluster_map_ptr, chain_map_ptr](
          const supernode_t            &leaf_cluster_supernode,
          const full_leaf_cluster_info &cluster_info) {
        cluster_name_t child_cluster_name =
            std::make_pair(leaf_cluster_supernode, 0);

        // If the parent chain is the root chain, then we know its valid parent
        // cluster and set it
        if (cluster_info.parent_chain == root_chain_supernode) {
          cluster_name_t parent_cluster_name =
              std::make_pair(root_chain_supernode, cluster_info.parent_edge_id);
          // Set the parent of the child cluster
          valid_cluster_map_ptr->async_visit(child_cluster_name, set_parent,
                                             parent_cluster_name);
          // Set the child of the parent cluster
          valid_cluster_map_ptr->async_visit(parent_cluster_name, set_child,
                                             child_cluster_name);
        }

        // Otherwise, send to the parent chain to look for the valid parent
        // cluster
        else {
          chain_map_ptr->async_visit(
              cluster_info.parent_chain,
              send_valid_cluster_to_valid_parent_functor(), child_cluster_name,
              cluster_info.parent_edge_id, valid_cluster_map_ptr);
        }
      };
  leaf_cluster_map.for_all(send_valid_leaf_clusters_to_parent);
  comm.barrier();

  // Do each chains from each round of contraction in order and get parent/child
  // relations by climbing up the hierarchy
  for (int round = 2; round <= final_round; ++round) {
    auto visit_chains_this_round =
        [valid_cluster_map_ptr, chain_map_ptr, round](
            const supernode_t &chain_name,
            std::pair<std::map<id_t, full_cluster_info>, full_chain_info>
                &chain) {
          if (chain_name.second == round) {
            if (chain.first.size() > 0) {
              // Go to the top cluster in this chain and send it up to find its
              // parent
              auto              it              = chain.first.rbegin();
              id_t              cluster_edge_id = it->first;
              full_cluster_info cluster_info    = it->second;
              cluster_name_t    child_cluster_name =
                  std::make_pair(chain_name, cluster_edge_id);
              // If the parent chain is the root chain, then we know its valid
              // parent cluster and set it
              if (chain.second.parent_chain == root_chain_supernode) {
                cluster_name_t parent_cluster_name = std::make_pair(
                    root_chain_supernode, chain.second.parent_edge_id);
                // Set the parent of the child cluster
                valid_cluster_map_ptr->async_visit(
                    child_cluster_name, set_parent, parent_cluster_name);
                // Set the child of the parent cluster
                valid_cluster_map_ptr->async_visit(
                    parent_cluster_name, set_child, child_cluster_name);
              }

              // Otherwise, send to the parent chain to look for the valid
              // parent cluster
              else {
                chain_map_ptr->async_visit(
                    chain.second.parent_chain,
                    send_valid_cluster_to_valid_parent_functor(),
                    child_cluster_name, chain.second.parent_edge_id,
                    valid_cluster_map_ptr);
              }
            }
          }
        };

    chain_map.for_all(visit_chains_this_round);
    comm.barrier();
  }
}

/**
 * @brief Writes information on valid clusters to file, disregarding
 * min_cluster_size.
 *
 * @param comm YGM comm.
 * @param output_dir Output directory for YGM multi_output to write files.
 * @param _cluster_file_name Filename prefix for output files.
 * @param file_number File number to add to the filename for this rank.
 * @param valid_cluster_map YGM map of valid clusters mapping cluster name ->
 * valid cluster info.
 */
void write_valid_clusters_to_file(
    ygm::comm &comm, std::string output_dir, std::string _cluster_file_name,
    int file_number,
    ygm::container::map<cluster_name_t, full_valid_cluster_info>
        &valid_cluster_map) {
  ygm::io::multi_output mo_clusters(comm, output_dir);

  static std::string cluster_file_name;
  cluster_file_name = _cluster_file_name;

  // Output cluster data
  if (comm.rank() == file_number) {
    std::string header =
        "cluster_name,size,stability,stability_traversing_up,"
        "selected,cluster_id,birth_distance,parent,child1,child2,num_points_"
        "added";
    mo_clusters.async_write_line(cluster_file_name, header);
  }
  comm.barrier();

  // Write valid clusters
  auto write_valid_clusters_lambda =
      [&mo_clusters](const cluster_name_t          &cluster_name,
                     const full_valid_cluster_info &cluster_info) {
        std::stringstream ss;
        ss << "\"" << cluster_name << "\"" << "," << cluster_info.size << ","
           << cluster_info.stability << ","
           << cluster_info.stability_traversing_up << ","
           << cluster_info.selected << "," << cluster_info.cluster_id << ","
           << cluster_info.birth_distance << "," << "\"" << cluster_info.parent
           << "\"" << ","
           << "\"" << cluster_info.children[0] << "\"" << ","
           << "\"" << cluster_info.children[1] << "\"" << ","
           << cluster_info.num_points_added;
        mo_clusters.async_write_line(cluster_file_name, ss.str());
      };
  valid_cluster_map.for_all(write_valid_clusters_lambda);
  comm.barrier();
}

}  // namespace clams::clustering