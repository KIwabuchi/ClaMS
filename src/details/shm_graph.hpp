// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>

#include "data_types.hpp"
#include "multithread_adjacency_list.hpp"

namespace clams {
using shm_graph_t =
    multithread_adjacency_list<id_t, std::pair<id_t, distance_t>>;

/// \brief Read knng files.
/// \param knng_files A list of knng files.
/// \param graph A graph to store the knng.
/// \param has_distance If true, read the distances in the knng files; if false,
/// ignore them.
/// \details Each knng file is expected to have the following format:
///   - Each even numbered line (0-based) contains a source vertex ID followed
///   by its neighbor vertex IDs, separated by whitespace.
///   - Each odd numbered line contains the corresponding distances to the
///   neighbors, separated by whitespace.
///   - The first neighbor ID in each even numbered line is the source vertex ID
///   itself, and the corresponding distance in the odd numbered line is 0.0
///   (dummy distance).
///   - The number of neighbor IDs in each even numbered line must match the
///   number of distances in the corresponding odd numbered line.
void read_knng(const std::vector<std::filesystem::path> &knng_files,
               shm_graph_t &graph, const bool has_distance = true) {
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
      if (has_distance) {
        std::getline(ifs, line);
        std::istringstream iss(line);
        distance_t         buf;
        while (iss >> buf) {
          dists.push_back(buf);
        }
      } else {
        dists.resize(ids.size());
      }

      if (has_distance && ids.size() != dists.size()) {
        std::cerr << "Invalid file: " << file << std::endl;
        std::cerr << "#of IDs and distances do not match" << std::endl;
        std::exit(1);
      }

      if (ids.empty()) {
        break;  // End of file
      }

      const id_t src = ids[0];
      for (std::size_t i = 1; i < ids.size(); ++i) {
        graph.add(src, std::make_pair(ids[i], dists[i]));
      }
    }
  }
}

/// \brief Read knng files, keeping only the rows whose source vertex ID is
/// in `keep_ids`.
/// \param knng_files A list of knng files.
/// \param keep_ids The set of source vertex IDs to keep; rows for other
/// source IDs are skipped.
/// \param graph A graph to store the knng.
/// \param has_distance If true, read the distances in the knng files; if
/// false, ignore them.
/// \details Uses the same file format as \ref read_knng.
void read_knng_filtered(
    const std::vector<std::filesystem::path>              &knng_files,
    const boost::unordered::unordered_flat_set<id_t>       &keep_ids,
    shm_graph_t &graph, const bool has_distance = true) {
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
      if (has_distance) {
        std::getline(ifs, line);
        std::istringstream iss(line);
        distance_t         buf;
        while (iss >> buf) {
          dists.push_back(buf);
        }
      } else {
        dists.resize(ids.size());
      }

      if (has_distance && ids.size() != dists.size()) {
        std::cerr << "Invalid file: " << file << std::endl;
        std::cerr << "#of IDs and distances do not match" << std::endl;
        std::exit(1);
      }

      if (ids.empty()) {
        break;  // End of file
      }

      const id_t src = ids[0];
      if (!keep_ids.contains(src)) {
        continue;  // Source vertex not of interest; skip this row
      }
      for (std::size_t i = 1; i < ids.size(); ++i) {
        graph.add(src, std::make_pair(ids[i], dists[i]));
      }
    }
  }
}

/// \brief Dump a graph to a file.
/// Use the same format as the input knng files.
/// \param graph A graph to dump.
/// \param file A file to dump the graph.
void dump_graph(const shm_graph_t &graph, const std::filesystem::path &file) {
  std::ofstream ofs(file);
  if (!ofs.is_open()) {
    std::cerr << "Cannot open file: " << file << std::endl;
    std::exit(1);
  }

  for (auto vit = graph.keys_begin(); vit != graph.keys_end(); ++vit) {
    const auto vid = vit->first;
    ofs << vid;

    // Neighbor IDs
    for (auto eit = graph.values_begin(vid); eit != graph.values_end(vid);
         ++eit) {
      ofs << " " << eit->first;
    }
    ofs << std::endl;

    // Distances
    ofs << "0.0 ";  // Dummy distance
    for (auto eit = graph.values_begin(vid); eit != graph.values_end(vid);
         ++eit) {
      ofs << eit->second << " ";
    }
    ofs << std::endl;
  }
}

void make_undirected_graph(shm_graph_t &graph) {
  std::vector<id_t> vertices;
  for (auto vit = graph.keys_begin(); vit != graph.keys_end(); ++vit) {
    vertices.push_back(vit->first);
  }

  shm_graph_t r_graph;  // Reversed graph

  OMP_DIRECTIVE(parallel for)
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    const auto vid = vertices[i];
    for (auto eit = graph.values_begin(vid); eit != graph.values_end(vid);
         ++eit) {
      const auto nid  = eit->first;
      const auto dist = eit->second;
      r_graph.add(nid, std::make_pair(vid, dist));
    }
  }

  // Merge the reversed graph to the original graph
  OMP_DIRECTIVE(parallel for)
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    const auto vid = vertices[i];
    if (r_graph.num_values(vid) == 0) {
      continue;
    }
    for (auto eit = r_graph.values_begin(vid); eit != r_graph.values_end(vid);
         ++eit) {
      graph.add(vid, *eit);
    }
  }
}
}  // namespace clams