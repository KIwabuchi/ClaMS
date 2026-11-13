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
 * @brief Read MST edges from Metall datastore.
 *
 **/
std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> read_pm_mst(
    std::filesystem::path &dstore_path, ygm::comm &comm) {
  std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> mst_edges;

  // Iterate over local datastore paths and read kNNG from each local manager
  const auto local_paths =
      get_local_datastore_paths(dstore_path, comm.get_mpi_comm());

  // If there's a single file, see if we can parallelize reading it by accessing
  // the same file from multiple ranks.
  bool done_reading_mst_file = false;
  if (local_paths.size() == 1) {
    try {
      metall::manager metall_manager(metall::open_read_only, dstore_path);
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
      // do nothing
    }
  }

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

        for (uint64_t i = 0; i <= (*input_mst_edges).size(); ++i) {
          std::pair<distance_t, std::pair<id_t, id_t>> array_item =
              std::make_pair((*input_mst_edges)[i].distance,
                             std::make_pair((*input_mst_edges)[i].ids[0],
                                            (*input_mst_edges)[i].ids[1]));
          mst_edges.push_back(array_item);
        }
      }
    }
  }

  comm.cf_barrier();
  return mst_edges;
}

/**
 * @brief Read MST edges from text file(s).
 *
 * @param input_paths Vector of paths to input files to read.
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
 * @brief Count the number of valid MST edges with edge id assigned in text
 * file(s).
 *
 * @param input_paths Vector of paths to input files to read.
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

}  // namespace clams::clustering