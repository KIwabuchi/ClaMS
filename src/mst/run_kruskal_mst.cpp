// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// Find the MST using Kruskal's algorithm

// If GCC is used,
#ifdef __GNUC__
#define _GLIBCXX_PARALLEL
#else
#warning "No parallelism library is used."
#endif

#include <unistd.h>
#include <execution>
#include <filesystem>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "../common.hpp"

namespace omp = metall::utility::omp;

bool parse_option(int argc, char *argv[], std::filesystem::path &knng_dir,
                  std::filesystem::path &output_dir) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <knng_dir> <output_dir>"
              << std::endl;
    return false;
  }

  knng_dir   = std::filesystem::path(argv[1]);
  output_dir = std::filesystem::path(argv[2]);

  return true;
}

id_t find_root(const id_t &id, std::unordered_map<id_t, id_t> &parent_tbl) {
  if (parent_tbl[id] == id) {
    return id;
  }
  parent_tbl[id] = find_root(parent_tbl[id], parent_tbl);
  return parent_tbl[id];
}

int main(int argc, char *argv[]) {
  std::filesystem::path knng_dir;
  std::filesystem::path output_dir;

  if (!parse_option(argc, argv, knng_dir, output_dir)) {
    return EXIT_FAILURE;
  }

  const auto knng_files = clams::find_files(knng_dir);

  clams::weighted_edge_list_t edges;
  spdlog::info("Read KNNG edges");
  clams::read_knng_edges(knng_files, edges);
  spdlog::info("#of edges {}", edges.size());

  spdlog::info("Sort edges");
  std::sort(edges.begin(), edges.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    if (lhs.ids[0] != rhs.ids[0]) {
      return lhs.ids[0] < rhs.ids[0];
    }
    return lhs.ids[1] < rhs.ids[1];
  });
  spdlog::info("Finished sorting edges");

  std::unordered_map<id_t, id_t> parent_tbl;
  clams::weighted_edge_list_t    mst_edges;
  spdlog::info("Start Kruskal's algorithm");
  for (const auto &edge : edges) {
    const auto src = edge.ids[0];
    const auto dst = edge.ids[1];

    if (parent_tbl.find(src) == parent_tbl.end()) {
      parent_tbl[src] = src;
    }
    if (parent_tbl.find(dst) == parent_tbl.end()) {
      parent_tbl[dst] = dst;
    }

    const auto src_root = find_root(src, parent_tbl);
    const auto dst_root = find_root(dst, parent_tbl);

    if (src_root != dst_root) {
      if (src_root < dst_root)
        parent_tbl[dst_root] = src_root;
      else
        parent_tbl[src_root] = dst_root;

      mst_edges.push_back(edge);
    }
  }
  spdlog::info("Finished Kruskal's algorithm");

  // Show the sum of distances of the MST edges
  {
    double total_distance = 0.0;
    for (const auto &edge : mst_edges) {
      total_distance += edge.distance;
    }
    spdlog::info("Total distance of MST edges: {}", total_distance);
  }

  spdlog::info("Write MST edges int {}", output_dir.string());
  std::filesystem::create_directories(output_dir);
  OMP_DIRECTIVE(parallel) {
    std::string path =
        output_dir /
        (std::string("mst-") + std::to_string(omp::get_thread_num()) + ".txt");
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
      spdlog::critical("Cannot open file: {}", path);
      std::exit(EXIT_FAILURE);
    }

    auto range = partial_range(mst_edges.size(), omp::get_thread_num(),
                               omp::get_num_threads());
    for (std::size_t i = range.first; i < range.second; ++i) {
      ofs << mst_edges[i] << "\n";
    }

    ofs.close();
    if (!ofs) {
      spdlog::critical("Failed to write to file: {}", path);
      std::exit(EXIT_FAILURE);
    }
  }
  spdlog::info("Done");

  return EXIT_SUCCESS;
}
