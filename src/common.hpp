// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

// MEMO: some OMP related macros are duplicated in saltatlas and metall
// Use saltatlas's one if CLAMS_USE_SALTATLAS is defined
#ifdef CLAMS_USE_SALTATLAS
#include <saltatlas/dnnd/detail/utilities/omp.hpp>
#else
#include <metall/utility/open_mp.hpp>
#endif

#include "details/data_types.hpp"
#include "details/time.hpp"
#include "details/utility.hpp"

namespace clams {

static constexpr id_t k_noise_cluster_id = static_cast<id_t>(-1);

inline std::vector<std::filesystem::path> find_files(
    const std::filesystem::path &path) {
  std::vector<std::filesystem::path> files;

  // If dir is a file, return it as a single element vector
  if (std::filesystem::is_regular_file(path)) {
    files.push_back(path);
    return files;
  }

  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path());
    }
  }

  return files;
}

inline std::vector<std::filesystem::path> find_files(
    const std::vector<std::filesystem::path> &paths) {
  std::vector<std::filesystem::path> files;
  for (const auto &path : paths) {
    const auto fs = find_files(path);
    files.insert(files.end(), fs.begin(), fs.end());
  }
  return files;
}

/// \brief Read edges.
/// \param path A path to an edge file or a directory that contains edge files.
inline void read_edges(const std::filesystem::path &path,
                       weighted_edge_list_t        &edges) {
  const auto files = find_files(path);
  for (const auto &file : files) {
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
      std::cerr << "Cannot open file: " << file << std::endl;
      std::exit(1);
    }

    std::string line;
    while (true) {
      id_t       src;
      id_t       dst;
      distance_t dist;
      ifs >> src >> dst >> dist;
      if (ifs.eof()) {
        break;
      }
      edges.emplace_back(src, dst, dist);
    }
  }
}

/// \brief Read a k-nearest-neighbor graph (kNNG) from files and store it as
/// an edge list.
/// \param knng_files A list of knng files.
/// \param graph A graph to store the knng.
inline void read_knng_edges(
    const std::vector<std::filesystem::path> &knng_files,
    weighted_edge_list_t                     &edges) {
  std::size_t num_edges = 0;
  OMP_DIRECTIVE(parallel for reduction(+ : num_edges))
  for (std::size_t fno = 0; fno < knng_files.size(); ++fno) {
    const auto   &file = knng_files[fno];
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
      std::cerr << "Cannot open file: " << file << std::endl;
      std::exit(1);
    }

    std::string line;
    while (true) {
      std::getline(ifs, line);
      if (line.empty()) {
        break;  // End of file
      }
      std::istringstream iss(line);
      id_t               buf;
      while (iss >> buf) {
        ++num_edges;
      }
      --num_edges;              // The first ID is the source
      std::getline(ifs, line);  // discard distances
    }
  }

  std::cout << "num_edges: " << num_edges << std::endl;
  edges.resize(num_edges);

  std::atomic<long long> cnt_edges{0};
  OMP_DIRECTIVE(parallel for)
  for (std::size_t fno = 0; fno < knng_files.size(); ++fno) {
    const auto   &file = knng_files[fno];
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
      std::cerr << "Cannot open file: " << file << std::endl;
      std::exit(1);
    }

    std::string line;
    while (true) {
      std::vector<id_t> ids;
      {
        std::getline(ifs, line);
        std::istringstream iss(line);
        id_t               buf;
        while (iss >> buf) {
          ids.push_back(buf);
        }
      }

      std::vector<distance_t> dists;
      {
        std::getline(ifs, line);
        std::istringstream iss(line);
        distance_t         buf;
        while (iss >> buf) {
          dists.push_back(buf);
        }
      }

      if (ids.size() != dists.size()) {
        std::cerr << "Invalid file: " << file << std::endl;
        std::cerr << "#of IDs and distances do not match" << std::endl;
        std::exit(1);
      }

      if (ids.empty()) {
        break;  // End of file
      }

      const id_t src = ids[0];
      for (std::size_t i = 1; i < ids.size(); ++i) {
        const auto index = cnt_edges.fetch_add(1);
        edges[index]     = weighted_edge_t{src, ids[i], dists[i]};
      }
    }
  }
}

/// Reads point-to-cluster assignments from `input_path` into
/// `point_cluster_map`.
///
/// Each nonempty, noncomment input line must contain a point ID followed by its
/// cluster ID, separated by whitespace. Lines whose first character is `#` are
/// ignored. The output maps each point ID to its cluster ID; if an ID occurs
/// more than once, the last assignment wins.
template <typename cluster_id_table_t>
void read_cluster_ids(const std::filesystem::path &input_path,
                      cluster_id_table_t          &point_cluster_map) {
  using id_t         = typename cluster_id_table_t::key_type;
  using cluster_id_t = typename cluster_id_table_t::mapped_type;

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
    id_t               point_id;
    cluster_id_t       cluster_id;
    if (!(iss >> point_id >> cluster_id)) {
      std::cerr << "Error parsing line: " << line << std::endl;
      std::abort();
    }
    point_cluster_map[point_id] = cluster_id;
  }
}

template <typename cluster_id_table_t>
void dump_point_cluster_ids(const cluster_id_table_t    &cluster_id,
                            const std::filesystem::path &output_path) {
  std::ofstream ofs(output_path);
  if (!ofs) {
    std::cerr << "Failed to open " << output_path << std::endl;
    std::abort();
  }

  for (const auto &[i, final_cluster_id] : cluster_id) {
    ofs << i << "\t" << final_cluster_id;
    ofs << "\n";
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to write to " << output_path << std::endl;
    std::abort();
  }
}

}  // namespace clams
