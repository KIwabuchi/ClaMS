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
}  // namespace clams