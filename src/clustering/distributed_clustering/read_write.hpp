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
 * @brief Get the local PM datastore paths. Taken from
 * clams::dist_pm_knng_t::get_local_datastore_paths
 *
 * @param datastore_path Path to PM datastore
 * @param comm MPI communicator
 */
std::vector<std::filesystem::path> get_local_datastore_paths(
    const std::filesystem::path &datastore_path, const MPI_Comm comm) {
  // Get the number of partitions in the old datastore
  const int n_partitions = metall::utility::metall_mpi_adaptor::partitions(
      datastore_path.c_str(), comm);
  const auto root_path =
      metall::utility::mpi_datastore::make_root_dir_path(datastore_path);

  std::vector<std::filesystem::path> paths;
  for (int r = 0; r < n_partitions; ++r) {
    const auto local_path =
        metall::utility::mpi_datastore::make_local_dir_path(root_path, r);
    paths.push_back(local_path);
  }
  return paths;
}

/**
 * @brief Read MST edges from a Metall datastore. The input format should be
 * clams::weighted_edge_list_t and the edges are read into pairs (distance,
 * (point_id1, point_id2)). This function can read a distributed PM weighted
 * edge list using a different number of mpi ranks than the one used to create
 * it.
 *
 * @param datastore_path Path to PM datastore
 * @param comm MPI communicator
 **/
std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> read_pm_mst(
    std::filesystem::path &dstore_path, ygm::comm &comm) {
  std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> mst_edges;
  mst_edges.clear();

  // Iterate over local datastore paths and read kNNG from each local manager
  const auto local_paths =
      get_local_datastore_paths(dstore_path, comm.get_mpi_comm());

  // If there's a single file, see if we can parallelize reading it by accessing
  // the same file from multiple ranks and having each rank read a chunk.
  bool done_reading_mst_file = false;
  if (local_paths.size() == 1) {
    try {
      metall::manager metall_manager(metall::open_read_only, local_paths.at(0));
      auto           *input_mst_edges =
          metall_manager
              .find<clams::weighted_edge_list_t>(metall::unique_instance)
              .first;

      uint64_t num_edges  = (*input_mst_edges).size();
      uint64_t chunk_size = num_edges / comm.size() + 1;
      uint64_t start_idx  = chunk_size * comm.rank();
      uint64_t end_idx =
          std::min(chunk_size * (comm.rank() + 1) - 1, num_edges - 1);

      for (uint64_t i = start_idx; i <= end_idx; ++i) {
        std::pair<distance_t, std::pair<id_t, id_t>> array_item =
            std::make_pair((*input_mst_edges)[i].distance,
                           std::make_pair((*input_mst_edges)[i].ids[0],
                                          (*input_mst_edges)[i].ids[1]));
        mst_edges.push_back(array_item);
      }
      done_reading_mst_file = true;
    } catch (...) {
      mst_edges.clear();
    }
  }

  // If we didn't successfully read a single file with multi-rank access, then
  // each rank reads its own file(s)
  if (!done_reading_mst_file) {
    for (std::size_t r = 0; r < local_paths.size(); ++r) {
      if (r % comm.size() == comm.rank()) {
        if (!metall::manager::consistent(local_paths[r])) {
          comm.cerr() << "Inconsistent local managers at rank " << comm.rank()
                      << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }
        metall::manager local_manager(metall::open_read_only, local_paths[r]);
        if (!local_manager.check_sanity()) {
          comm.cerr() << "Failed to open local manager at rank " << comm.rank()
                      << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }
        auto *input_mst_edges =
            local_manager
                .find<clams::weighted_edge_list_t>(metall::unique_instance)
                .first;

        if (!input_mst_edges) {
          comm.cerr() << "Failed to find MST edges" << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }

        for (auto edge : *input_mst_edges) {
          std::pair<distance_t, std::pair<id_t, id_t>> array_item =
              std::make_pair(edge.distance,
                             std::make_pair(edge.ids[0], edge.ids[1]));
          mst_edges.push_back(array_item);
        }
      }
    }
  }
  comm.barrier();

  return mst_edges;
}

/**
 * @brief Read MST edges from text file(s).
 *
 * @param input_paths Vector of input files to read.
 * @param comm YGM communicator to define a line parser
 **/
std::vector<std::pair<distance_t, std::pair<id_t, id_t>>>
read_mst_from_text_file(std::vector<std::string> &input_paths,
                        ygm::comm                &comm) {
  ygm::io::line_parser mst_line_parser(comm, input_paths);

  std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> mst_edges;

  auto line_parser_lambda = [&mst_edges](const std::string &line) {
    if (std::isdigit(line[0])) {
      try {
        std::stringstream ss(line);
        id_t              node1, node2;
        distance_t        dist;
        ss >> node1 >> node2 >> dist;
        std::pair<distance_t, std::pair<id_t, id_t>> array_item =
            std::make_pair(dist, std::make_pair(node1, node2));
        mst_edges.push_back(array_item);
      } catch (...) {
        std::cout << "Error reading mst line: " << line << std::endl;
      }
    } else {
      std::cout << "Read comment line in mst file: " << line << std::endl;
    }
  };
  mst_line_parser.for_all(line_parser_lambda);
  comm.barrier();

  return mst_edges;
}

/**
 * @brief Count the number of valid MST edges in text file(s). This is used when
 * we want read MST edges with with edge id already assigned.
 *
 * @param input_paths Vector of input files to read.
 * @param comm YGM communicator to define a line parser
 **/
uint64_t read_num_mst_edges_from_text_file(
    std::vector<std::string> &input_paths, ygm::comm &comm) {
  ygm::io::line_parser mst_line_parser(comm, input_paths);

  uint64_t num_edges_read;

  auto line_parser_lambda = [&num_edges_read](const std::string &line) {
    if (std::isdigit(line[0])) {
      try {
        std::stringstream ss(line);
        id_t              node1, node2;
        distance_t        dist;
        ss >> node1 >> node2 >> dist;
        ++num_edges_read;
      } catch (...) {
        // do nothing
      }
    }
  };
  mst_line_parser.for_all(line_parser_lambda);
  comm.barrier();

  return num_edges_read;
}

/**
 * @brief Read MST edges with ids already assigned from text file(s).
 *
 * @param input_paths Vector of paths to input files to read.
 * @param edge_endpoints_array A YGM array of edge id -> (edge_endpoint1,
 * edge_endpoint2). This array is not filled with values yet, but needs to have
 * size equal to the number of edges when initialized.
 * @param edge_contraction_array An YGM array of edge id ->
 * edge_contraction_info. This array is not filled with values yet, but needs to
 * have size equal to the number of edges when initialized.
 **/
void read_mst_edges_with_id_from_text_file(
    std::vector<std::string>                     &input_paths,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array,
    ygm::container::array<edge_contraction_info> &edge_contraction_array) {
  ygm::comm &comm = edge_endpoints_array.comm();

  ygm::io::line_parser mst_line_parser(comm, input_paths);

  auto line_parser_lambda = [&edge_endpoints_array,
                             &edge_contraction_array](const std::string &line) {
    if (std::isdigit(line[0])) {
      try {
        std::stringstream ss(line);
        id_t              edge_id;
        id_t              node1, node2;
        distance_t        dist;
        ss >> edge_id >> node1 >> node2 >> dist;
        std::pair<id_t, id_t> edge_endpoints = std::make_pair(node1, node2);
        edge_endpoints_array.async_insert(edge_id, edge_endpoints);
        edge_contraction_array.async_insert(
            edge_id,
            edge_contraction_info{.endpoint_supernode_reps = edge_endpoints,
                                  .distance                = dist});

      } catch (...) {
        std::cout << "Error reading mst line: " << line << std::endl;
      }
    } else {
      if (line[0] != '#') {
        std::cout << "Read comment line in mst file: " << line << std::endl;
      }
    }
  };
  mst_line_parser.for_all(line_parser_lambda);
  comm.barrier();
}

/**
 * @brief Sort and process MST edges.
 *
 * @param mst_edge_vector Vector of MST edges (distance, (endpoint1,
 * endpoint2)). Each rank has its own vector of edges and edges should not be
 * duplicated between ranks.
 * @param edge_endpoints_array A YGM array of edge id -> (edge_endpoint1,
 * edge_endpoint2). This array is not filled with values yet, but needs to have
 * size equal to the number of edges when initialized.
 * @param edge_contraction_array An YGM array of edge id ->
 * edge_contraction_info. This array is not filled with values yet, but needs to
 * have size equal to the number of edges when initialized.
 */
void sort_and_process_mst_edges_into_maps(
    std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> &mst_edge_vector,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array,
    ygm::container::array<edge_contraction_info> &edge_contraction_array) {
  ygm::comm &comm = edge_endpoints_array.comm();

  // Create a YGM array with the MST edges
  ygm::container::array<std::pair<distance_t, std::pair<id_t, id_t>>>
      edge_array(comm, mst_edge_vector);
  comm.barrier();

  // Sort the edges
  edge_array.sort();
  comm.barrier();

  // Fill in the edge endpoints and contraction map
  auto visit_sorted_array_lambda =
      [&edge_endpoints_array, &edge_contraction_array](
          const std::size_t                                  &index,
          const std::pair<distance_t, std::pair<id_t, id_t>> &value) {
        // Add this edge to the edge map with its sorted position as its id
        distance_t            distance       = value.first;
        std::pair<id_t, id_t> edge_endpoints = value.second;
        edge_endpoints_array.local_insert(index, edge_endpoints);
        edge_contraction_array.local_insert(
            index,
            edge_contraction_info{.endpoint_supernode_reps = edge_endpoints,
                                  .distance                = distance});
      };
  edge_array.for_all(visit_sorted_array_lambda);
  comm.barrier();
}

/**
 * @brief Writes cluster labels to text files, writing one file per rank.
 * Files are in the form point_id \t cluster_label in each row.
 *
 * @param point_to_cluster_id_map YGM map of point id -> cluster label.
 * @param output_path File to write cluster data to. Each rank must
 * receive its own file name (e.g., path_to_cluster_labels/rank.txt).
 */
void write_cluster_labels_one_txt_file_per_rank(
    ygm::container::map<id_t, cluster_id_t> &point_to_cluster_id_map,
    std::filesystem::path                    output_path) {
  ygm::comm &comm = point_to_cluster_id_map.comm();

  int num_files = comm.size();
  if (comm.rank() == 0) {
    std::filesystem::create_directories(output_path);
  }
  comm.barrier();

  // Output points and cluster labels
  std::string           label_file_name = std::to_string(comm.rank()) + ".txt";
  std::filesystem::path label_file_path = output_path / label_file_name;
  std::ofstream         label_ofs(label_file_path);
  if (!label_ofs.is_open()) {
    std::cout << "Error! Unable to open label file: "
              << label_file_path.string() << std::endl;
  }

  label_ofs << "# point_id\tcluster_id\n";
  for (const auto &[point_id, cluster_id] : point_to_cluster_id_map) {
    label_ofs << point_id << "\t" << cluster_id << "\n";
  }
  label_ofs.flush();
  label_ofs.close();
  comm.barrier();
}

/**
 * @brief Writes cluster labels to a single-rank Metall datastore. Labels
 * are saved as a vector of pairs (point_id, cluster_label).
 *
 * @param point_to_cluster_id_map YGM map of point id -> cluster label.
 * @param output_path Path (as a string) to output Metall datastore.
 */
void write_cluster_labels_single_pm(
    ygm::container::map<id_t, cluster_id_t> &point_to_cluster_id_map,
    const std::string                       &output_path) {
  ygm::comm &comm = point_to_cluster_id_map.comm();

  // Get local point ids and cluster labels
  std::vector<id_t>         point_ids;
  std::vector<cluster_id_t> cluster_labels;
  point_ids.reserve(point_to_cluster_id_map.local_size());
  cluster_labels.reserve(point_to_cluster_id_map.local_size());

  for (const auto &[id, label] : point_to_cluster_id_map) {
    point_ids.push_back(id);
    cluster_labels.push_back(label);
  }

  const size_t num_points = point_to_cluster_id_map.size();

  MPI_Comm mpi_comm = comm.get_mpi_comm();

  if (comm.rank() == 0) {
    metall::manager manager(metall::create_only, output_path);
    auto            cluster_id_table =
        manager.construct<clams::pm_point_to_cluster_id_list_t>(
            "point_to_cluster_id_list")(manager.get_allocator());
    cluster_id_table->reserve(num_points);

    // Write own edges
    for (size_t i = 0; i < point_ids.size(); ++i) {
      // cluster_id_table->emplace(point_ids.at(i), cluster_labels.at(i));
      cluster_id_table->emplace_back(
          std::make_pair(point_ids.at(i), cluster_labels.at(i)));
    }

    // Write edges for all other ranks
    for (int r = 1; r < comm.size(); ++r) {
      std::vector<id_t> remote_ids =
          comm.mpi_recv<std::vector<id_t>>(r, 0, mpi_comm);
      std::vector<cluster_id_t> remote_labels =
          comm.mpi_recv<std::vector<cluster_id_t>>(r, 1, mpi_comm);
      for (size_t i = 0; i < remote_ids.size(); ++i) {
        cluster_id_table->emplace_back(
            std::make_pair(remote_ids.at(i), remote_labels.at(i)));
      }
    }

  } else {
    comm.mpi_send(point_ids, 0, 0, mpi_comm);
    comm.mpi_send(cluster_labels, 0, 1, mpi_comm);
  }
  comm.barrier();
}

/**
 * @brief Writes information for all clusters to file, disregarding
 * min_cluster_size.
 *
 * @param cluster_file_path File to write cluster data to. Each rank must
 * receive its own file name (e.g., path_to_cluster_data/rank.csv).
 * @param root_chain_min_edge_id The smallest cluster edge id in the root
 * chain. We write the second root chain child info to that cluster.
 * @param root_chain_second_child Info of the second child for the bottom root
 * chain cluster.
 * @param root_chain_cluster_map YGM map of root chain cluster edge id -> root
 * chain cluster info.
 * @param chain_map YGM map of chain name -> (cluster map, chain info).
 * @param leaf_cluster_map YGM map of leaf cluster name -> leaf cluster info.
 */
void write_all_clusters_to_file_including_invalid_clusters(
    std::filesystem::path cluster_file_path, supernode_t root_chain_supernode,
    id_t                                                root_chain_min_edge_id,
    extra_child_cluster_info                           &root_chain_second_child,
    ygm::container::map<id_t, root_chain_cluster_info> &root_chain_cluster_map,
    ygm::container::map<supernode_t,
                        std::pair<std::map<id_t, full_cluster_info>,
                                  full_chain_info>>    &chain_map,
    ygm::container::map<supernode_t, full_leaf_cluster_info>
        &leaf_cluster_map) {
  ygm::comm &comm = chain_map.comm();

  std::ofstream ofs(cluster_file_path);

  // Output cluster data
  // Header
  ofs << "cluster_name,size,stability,stability_traversing_up,"
         "selected,cluster_id,parent_cluster_name,birth_distance,"
         "min_edge_distance,non_chain_child1,non_chain_child2,valid_cluster"
      << "\n";

  // Write root chain clusters
  for (auto &[cluster_edge_id, cluster_info] : root_chain_cluster_map) {
    std::stringstream ss;
    ss << "\"" << std::make_pair(root_chain_supernode, cluster_edge_id) << "\","
       << cluster_info.size << "," << cluster_info.stability << ","
       << cluster_info.stability_traversing_up << "," << cluster_info.selected
       << "," << cluster_info.cluster_id << ","
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
    ofs << ss.str() << "\n";
  }

  // Write chain clusters
  for (auto &[chain_name, chain] : chain_map) {
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
           << cluster_info.min_edge.second << "," << "\"" << cluster_info.child
           << "\"" << ",";
        if (it == chain.first.begin()) {
          ss << "\"" << chain.second.children[1] << "\"";
        }
        ss << "," << cluster_info.valid_cluster;
        ofs << ss.str() << "\n";
      }
    }
  }

  // Write leaf clusters
  for (auto &[cluster_name, cluster_info] : leaf_cluster_map) {
    if (cluster_info.valid_cluster) {
      std::stringstream ss;
      ss << "\"" << std::make_pair(cluster_name, 0) << "\"" << ","
         << cluster_info.edges.size() + 1 << "," << cluster_info.stability
         << "," << cluster_info.stability << "," << cluster_info.selected << ","
         << cluster_info.cluster_id << ","
         << "\""
         << std::make_pair(cluster_info.parent_chain,
                           cluster_info.parent_edge_id)
         << "\""
         << "," << cluster_info.birth_distance << ",,,,"
         << cluster_info.valid_cluster;
      ofs << ss.str() << "\n";
    }
  }

  comm.barrier();
}

/**
 * @brief Writes information on valid clusters to file.
 *
 * @param cluster_file_path File to write cluster data to. Each rank must
 * receive its own file name (e.g., path_to_cluster_data/rank.csv).
 * @param valid_cluster_map YGM map of valid clusters mapping cluster name ->
 * valid cluster info.
 *
 * @return The sum of sizes of all selected clusters.
 * This is a sanity check against the number of points clustered.
 */
id_t write_valid_clusters_to_file(
    std::filesystem::path cluster_file_path,
    ygm::container::map<cluster_name_t, full_valid_cluster_info>
        &valid_cluster_map) {
  ygm::comm &comm = valid_cluster_map.comm();

  id_t num_points_clustered = 0;

  // Create output file stream
  std::ofstream ofs(cluster_file_path);
  if (!ofs.is_open()) {
    std::cout << "Unable to open cluster info file: "
              << cluster_file_path.string();
  }

  // Write header
  ofs << "cluster_name,size,stability,stability_traversing_up,"
         "selected,cluster_id,birth_distance,parent,child1,child2,num_points_"
         "added"
      << "\n";

  // Write valid clusters
  for (auto &[cluster_name, cluster_info] : valid_cluster_map) {
    std::stringstream ss;
    ss << "\"" << cluster_name << "\""
       << "," << cluster_info.size << "," << cluster_info.stability << ","
       << cluster_info.stability_traversing_up << "," << cluster_info.selected
       << "," << cluster_info.cluster_id << "," << cluster_info.birth_distance
       << ","
       << "\"" << cluster_info.parent << "\""
       << ","
       << "\"" << cluster_info.children[0] << "\""
       << ","
       << "\"" << cluster_info.children[1] << "\""
       << "," << cluster_info.num_points_added;
    ofs << ss.str() << "\n";

    if (cluster_info.selected) {
      num_points_clustered += cluster_info.size;
    }
  }
  ofs.flush();
  ofs.close();
  comm.barrier();

  id_t total_num_points_clustered = ygm::sum(num_points_clustered, comm);

  return total_num_points_clustered;
}

}  // namespace clams::clustering