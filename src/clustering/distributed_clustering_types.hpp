#pragma once

/*
Hash function for std::pair
Syntax taken from: https://lists.isocpp.org/std-discussion/2020/12/0937.php
Using boost's hash_combine:
https://www.boost.org/doc/libs/1_51_0/doc/html/hash/combine.html

*/
namespace std {

template <typename T1, typename T2>
struct hash<pair<T1, T2>> {
  size_t operator()(const pair<T1, T2> &input_pair) const {
    size_t seed = 0;
    boost::hash_combine(seed, input_pair.first);
    boost::hash_combine(seed, input_pair.second);
    return seed;
    // return std::hash<T1>()(input_pair.first) ^
    // std::hash<T2>()(input_pair.second);
  }
};
}  // namespace std

// Types
namespace clams::clustering {

// Point/vertex ID type.
using id_t = clams::id_t;
// Distance type.
using distance_t = clams::distance_t;

// Translates id_t to MPI type
MPI_Datatype mpi_id_type() {
  if constexpr (std::is_same_v<id_t, uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<id_t, uint64_t>) {
    return MPI_UINT64_T;
  } else {
    static_assert(
        std::is_same_v<id_t, uint32_t> || std::is_same_v<id_t, uint64_t>,
        "id_t must be uint32_t or uint64_t");
  }
}

// Translates distance_t to MPI type
MPI_Datatype mpi_distance_type() {
  if constexpr (std::is_same_v<distance_t, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<distance_t, double>) {
    return MPI_DOUBLE;
  } else {
    static_assert(
        std::is_same_v<distance_t, float> || std::is_same_v<distance_t, double>,
        "distance_t must be float or double");
  }
}

// Cluster label type
using cluster_id_t             = int32_t;
cluster_id_t NOISE_POINT_LABEL = -1;  // cluster label for noise points

// Supernode type - Supernodes are identified by a pair of
// (original node contained in the supernode, iteration contracted/formed)
using supernode_t = std::pair<id_t, uint32_t>;
static const supernode_t BLANK_SUPERNODE{0, 0};

// Cluster name type - To have unique cluster names, we make the cluster
// names (supernode, edge_id), where supernode is the contracted supernode in
// the MST that this cluster forms at and edge_id is the edge in that supernode
// that splits of the cluster. If the entire supernode corresponds to a cluster,
// then edge_id = 0
using cluster_name_t = std::pair<supernode_t, id_t>;
static const cluster_name_t BLANK_CLUSTER_NAME{BLANK_SUPERNODE, 0};

using edge_id_with_dist_t = std::pair<id_t, distance_t>;
// id_t ROOT_EDGE_ID         = -1;

/**
 * @brief Calculate lambda = 1/distance from distance.
 **/
static distance_t lambda_from_dist(distance_t distance) {
  if (distance > 0.0) {
    return static_cast<distance_t>(1.0) / distance;
  } else {
    return std::numeric_limits<distance_t>::max();
  }
}

// overload operator<< for pair
template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> my_pair) {
  os << "(" << my_pair.first << ", " << my_pair.second << ")";
  return os;
}

// overload operator<< for vector
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> my_vector) {
  for (const T &entry : my_vector) {
    os << entry << " ";
  }
  return os;
}

/**
 * @brief Holds all edge info needed in MST contraction phase.
 **/
struct edge_contraction_info {
  // Current reps of supernode endpoints (if edge is not contracted yet, i.e.,
  // contraction_round = 0). We don't store the full supernodes since the round
  // is known
  std::pair<id_t, id_t> endpoint_supernode_reps{0, 0};

  // The supernode of the chain (parent dendrogram edge, supernode child) we
  // think this edge is
  supernode_t chain_supernode{BLANK_SUPERNODE};

  // The parent edge id for our chain. This is 0 until we find parent dendrogram
  // edge greater than edge_id (or this edge ends up in the root chain)
  id_t chain_parent_edge_id{0};

  // Distance between the two points in the edge
  distance_t distance{0};

  // Which iteration round the edge was contracted into a supernode
  uint32_t contraction_round{0};  // 0 indicates its not contracted yet

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(endpoint_supernode_reps, chain_supernode, chain_parent_edge_id,
            distance, contraction_round);
  }
#endif
};

// overload operator<< for edge info
std::ostream &operator<<(std::ostream                &os,
                         const edge_contraction_info &edge_info) {
  os << "dist = " << edge_info.distance
     << ", last supernode endpoint reps = " << edge_info.endpoint_supernode_reps
     << ", contraction round = " << edge_info.contraction_round
     << ", current chain supernode = " << edge_info.chain_supernode
     << ", found chain parent = " << edge_info.chain_parent_edge_id;
  return os;
}

/**
 * @brief Holds all info we need to keep for alpha edges, which are edges
 * contracted after the first round. Alpha edges have 2 non-leaf children in the
 * dendrogram.
 **/
struct alpha_edge_info {
  // Supernode children (will have 1 or 2) in the dendrogram
  std::array<supernode_t, 2> dendrogram_children{BLANK_SUPERNODE,
                                                 BLANK_SUPERNODE};

  // The supernode of the chain this edge is in
  supernode_t chain_supernode{BLANK_SUPERNODE};

  // Distance between the two points in the edge
  distance_t distance{0};

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(dendrogram_children, chain_supernode, distance);
  }
#endif
};

// overload operator<< for alpha edge info
std::ostream &operator<<(std::ostream &os, const alpha_edge_info &edge_info) {
  os << "dist = " << edge_info.distance
     << ", chain supernode = " << edge_info.chain_supernode
     << ", dendrogram children: " << edge_info.dendrogram_children[0] << " and "
     << edge_info.dendrogram_children[1];
  return os;
}

/**
 * @brief Holds info for clusters that are leaf clusters (have no child
 * clusters) when we disregard min_cluster_size
 **/
struct full_leaf_cluster_info {
  // Chain the parent edge belongs to
  supernode_t parent_chain{BLANK_SUPERNODE};

  // Parent edge id
  id_t parent_edge_id{0};

  // Birth distance - distance attached to the parent edge of this cluster in
  // the dendrogram
  distance_t birth_distance{0.0};

  // Cluster stability score
  distance_t stability{0.0};

  // ID of cluster - this is its own id if selected
  // If the cluster is not selected, its the id for a selected parent above
  id_t cluster_id{std::numeric_limits<id_t>::max()};

  // Whether the cluster is selected in final flat clustering
  bool selected{false};

  // Whether the cluster a valid cluster in the hierarchy
  // (otherwise its a chunk that get annexed into another cluster)
  bool valid_cluster{false};

  // Edges added at this part of the cluster hierarchy. The set of all edges in
  // this cluster is the union of edges for this cluster, and edges of any
  // child clusters and their child clusters, etc.
  std::vector<edge_id_with_dist_t> edges;

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(parent_chain, parent_edge_id, birth_distance, stability, cluster_id,
            selected, valid_cluster, edges);
  }
#endif
};

// overload operator<< for leaf cluster info
std::ostream &operator<<(std::ostream                 &os,
                         const full_leaf_cluster_info &cluster) {
  os << "stability = " << cluster.stability
     << ", size = " << cluster.edges.size() + 1
     << ", valid = " << cluster.valid_cluster
     << ", selected = " << cluster.selected
     << ", birth distance = " << cluster.birth_distance
     << ", parent = " << cluster.parent_chain
     << ", parent edge id = " << cluster.parent_edge_id;
  os << ", edges added in this cluster: ";
  for (const edge_id_with_dist_t &edge : cluster.edges) {
    os << edge.first << " ";
  }
  return os;
}

/**
 * @brief Holds cluster info
 **/
struct full_cluster_info {
  // All stored values are for the non-chain child

  // Child chain name
  supernode_t child{BLANK_SUPERNODE};

  // Child cluster stability value
  distance_t child_stability{0.0};

  // Child stability traversing up the dendrogram
  // If a cluster has smaller stability than the sum of stabilities of its child
  // clusters, the stability traversing up will be the sum of child stabilities
  distance_t child_stability_traversing_up{0.0};

  // Array containing child cluster sizes
  id_t child_size{0};

  // The minimum edge (by id) in edges
  edge_id_with_dist_t min_edge;

  // Birth distance - distance attached to the parent edge of this cluster in
  // the dendrogram
  distance_t birth_distance{0.0};

  // Cluster stability score
  distance_t stability{0.0};

  // Stability score we sent up to parent when traversing up dendrogram
  distance_t stability_traversing_up{0.0};

  // Cluster size (number of nodes)
  id_t size{0};

  // ID of cluster - this is its own id if selected
  // If the cluster is not selected, its the id for a selected parent above
  id_t cluster_id{std::numeric_limits<id_t>::max()};

  // Whether the cluster is selected in final flat clustering
  bool selected{false};

  // Whether the cluster a valid cluster in the hierarchy
  // (otherwise its a chunk that get annexed into another cluster)
  bool valid_cluster{false};

  // Edges added at this part of the cluster hierarchy. The set of all edges in
  // this cluster is the union of edges for this cluster, and edges of any
  // child clusters and their child clusters, etc.
  std::vector<edge_id_with_dist_t> edges;

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(child, child_stability, child_stability_traversing_up, child_size,
            min_edge, birth_distance, stability, stability_traversing_up, size,
            cluster_id, selected, valid_cluster, edges);
  }
#endif
};

// overload operator<< for cluster info
std::ostream &operator<<(std::ostream &os, const full_cluster_info &cluster) {
  os << "stability = " << cluster.stability
     << ", stability traversing up = " << cluster.stability_traversing_up
     << ", size = " << cluster.size << ", valid = " << cluster.valid_cluster
     << ", selected = " << cluster.selected
     << ", birth distance = " << cluster.birth_distance
     << ", non-chain child: " << cluster.child;
  os << ", edges added in this cluster: ";
  for (const edge_id_with_dist_t &edge : cluster.edges) {
    os << edge.first << " ";
  }
  return os;
}

/**
 * @brief Holds chain info we need in addition to the clusters
 **/
struct full_chain_info {
  // Parent chain
  supernode_t parent_chain{BLANK_SUPERNODE};

  // parent edge id for this chain
  // Note: we need this to get birth distance of the top cluster in the chain
  id_t parent_edge_id{0};

  // // Min alpha edge id in this chain
  // id_t min_alpha_edge_id{0};

  // Unlike the other chain clusters, the min edge cluster will receive messages
  // from 2 children (instead of 1), store the second child's information in
  // these variables
  id_t       min_edge_child_size2{0};
  distance_t min_edge_child_stability2{0.0};
  distance_t min_edge_child_stability_traversing_up2{0.0};

  // Child chains for this chain
  // first entry is the same child stored in min_edge_id in the chain cluster
  // map, and second entry has its info stored in full_chain info in the
  // variables above
  std::array<supernode_t, 2> children{BLANK_SUPERNODE, BLANK_SUPERNODE};

  // Number of messages received
  id_t num_child_messages_received{0};

  // Vector to hold non-alpha edges before we assign them to clusters
  std::vector<edge_id_with_dist_t> non_alpha_edges;

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(parent_chain, parent_edge_id, min_edge_child_size2,
            min_edge_child_stability2, min_edge_child_stability_traversing_up2,
            children, num_child_messages_received, non_alpha_edges);
  }
#endif
};

/**
 * @brief Holds the 2nd root chain child cluster info
 **/
struct extra_child_cluster_info {
  id_t        size{0};
  distance_t  stability{0.0};
  distance_t  stability_traversing_up{0.0};
  supernode_t name{BLANK_SUPERNODE};

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(size, stability, stability_traversing_up, name);
  }
#endif
};

/**
 * @brief Holds reduced cluster info for processing root-chain clusters.
 * Notably, unlike full_cluster_info, the vector of edges is not included and
 * instead num_points_added and sum_lambda_edges_added are precomputed.
 **/
struct root_chain_cluster_info {
  // Non-root child chain name
  supernode_t child{BLANK_SUPERNODE};

  // Non-root child cluster stability value
  distance_t child_stability{0.0};

  // Non-root child stability traversing up the dendrogram
  // If a cluster has smaller stability than the sum of stabilities of its child
  // clusters, the stability traversing up will be the sum of child stabilities
  distance_t child_stability_traversing_up{0.0};

  // Non-root child size
  id_t child_size{0};

  // Cluster stability score
  distance_t stability{0.0};

  // Stability score we sent up to parent when traversing up dendrogram
  distance_t stability_traversing_up{0.0};

  // Cluster size (number of nodes)
  id_t size{0};

  // Number of points added to this root-chain cluster
  id_t num_points_added{0};

  // Sum of lambda contribution for all the edges added to this root-chain
  // cluster not including the min edge
  distance_t sum_lambda_edges_added{0.0};

  // Lambda for birth edge (the longest edge in this cluster)
  distance_t lambda_birth{0.0};

  // Lambda for min edge (the shortest edge added)
  distance_t lambda_min_edge{0.0};

  // Edge ID attached to its parent cluster (which will also be in the root
  // chain, so we don't need to store the supernode part of its name)
  id_t parent_edge_id{0};

  // ID of cluster - this is its own id if selected
  // If the cluster is not selected, its the id for a selected parent above
  id_t cluster_id{std::numeric_limits<id_t>::max()};

  // Whether the cluster is selected in final flat clustering
  bool selected{false};

  bool operator<(const root_chain_cluster_info &other) const {
    // larger lambda birth = smaller distance
    return lambda_birth > other.lambda_birth;
  }

  bool operator>(const root_chain_cluster_info &other) const {
    // larger lambda birth = smaller distance
    return lambda_birth < other.lambda_birth;
  }

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(child, child_stability, child_stability_traversing_up, child_size,
            stability, stability_traversing_up, size, num_points_added,
            sum_lambda_edges_added, lambda_birth, lambda_min_edge,
            parent_edge_id, cluster_id, selected);
  }
#endif
};

// overload operator<< for root-chain cluster info
std::ostream &operator<<(std::ostream                  &os,
                         const root_chain_cluster_info &cluster) {
  os << "stability = " << cluster.stability
     << ", stability traversing up = " << cluster.stability_traversing_up
     << ", size = " << cluster.size << ", selected = " << cluster.selected
     << ", num_points_added = " << cluster.num_points_added
     << ", sum_lambda_edges_added = " << cluster.sum_lambda_edges_added
     << ", lambda birth = " << cluster.lambda_birth
     << ", parent edge id = " << cluster.parent_edge_id
     << ", non-chain child: " << cluster.child << " with size "
     << cluster.child_size;
  return os;
}

/**
 * @brief Holds info for valid clusters to write to file
 **/
struct full_valid_cluster_info {
  // All stored values for printout for valid clusters

  // Valid cluster children (0 or 2)
  std::array<cluster_name_t, 2> children{BLANK_CLUSTER_NAME,
                                         BLANK_CLUSTER_NAME};

  // Parent cluster
  cluster_name_t parent{BLANK_CLUSTER_NAME};

  // Birth distance - distance attached to the parent edge of this cluster in
  // the dendrogram
  distance_t birth_distance{0.0};

  // Cluster stability score
  distance_t stability{0.0};

  // Stability score we sent up to parent when traversing up dendrogram
  distance_t stability_traversing_up{0.0};

  // Cluster size (number of nodes)
  id_t size{0};

  // ID of cluster - this is its own id if selected
  // If the cluster is not selected, its the id for a selected parent above
  id_t cluster_id{std::numeric_limits<id_t>::max()};

  // Whether the cluster is selected in final flat clustering
  bool selected{false};

  // Number of points added to this root-chain cluster -DEBUG
  id_t num_points_added{0};

#if __has_include(<cereal/types/base_class.hpp>)
  template <class Archive>
  void serialize(Archive &archive) {
    archive(children, parent, birth_distance, stability,
            stability_traversing_up, size, cluster_id, selected,
            num_points_added);
  }
#endif
};

}  // namespace clams::clustering