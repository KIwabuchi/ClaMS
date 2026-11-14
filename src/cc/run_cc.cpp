// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "../common.hpp"
#include "../details/shm_graph.hpp"
#include "cc.hpp"

namespace mtl_omp = metall::utility::omp;

bool parse_option(int argc, char *argv[], std::filesystem::path &knng_dir,
                  std::filesystem::path &output_dir, bool &detailed_analysis,
                  std::filesystem::path &cc_count_file) {
  knng_dir.clear();
  output_dir.clear();
  cc_count_file.clear();
  detailed_analysis = false;

  int opt;
  while ((opt = ::getopt(argc, argv, "i:o:dc:")) != -1) {
    switch (opt) {
      case 'i': {
        knng_dir = std::filesystem::path(optarg);
        break;
      }
      case 'o': {
        output_dir = std::filesystem::path(optarg);
        break;
      }
      case 'd': {
        detailed_analysis = true;
        break;
      }
      case 'c': {
        cc_count_file = std::filesystem::path(optarg);
        break;
      }
      default: {
        std::cerr << "Unknown option: " << opt << std::endl;
        return false;
      }
    }
  }

  if (knng_dir.empty()) {
    std::cerr << "No input directory is specified" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  std::filesystem::path knng_dir;
  std::filesystem::path output_dir;
  bool                  detailed_analysis = false;
  std::filesystem::path cc_count_file;

  if (!parse_option(argc, argv, knng_dir, output_dir, detailed_analysis,
                    cc_count_file)) {
    return EXIT_FAILURE;
  }

  const auto knng_files = clams::find_files(knng_dir);

  clams::shm_graph_t graph;

  std::cout << "Read knng" << std::endl;
  clams::read_knng(knng_files, graph);
  std::cout << "#of vertices: " << graph.num_keys() << std::endl;
  std::cout << "#of edges: " << graph.num_values() << std::endl;

  std::cout << "Make undirected graph" << std::endl;
  clams::make_undirected_graph(graph);
  std::cout << "#of vertices: " << graph.num_keys() << std::endl;
  std::cout << "#of edges: " << graph.num_values() << std::endl;

  auto [vertices, cc_table] = run_cc(graph);

  std::cout << "Count CC sizes" << std::endl;
  std::unordered_map<id_t, std::size_t> cc_sizes;
  for (const auto cc : cc_table) {
    ++cc_sizes[cc.second];
  }

  std::cout << "Number of CCs: " << cc_sizes.size() << std::endl;
  if (!cc_count_file.empty()) {
    std::ofstream ofs(cc_count_file);
    if (!ofs) {
      std::cerr << "Failed to create: " << cc_count_file << std::endl;
      std::exit(1);
    }
    ofs << cc_sizes.size() << std::endl;
  }
  if (!detailed_analysis) {
    return EXIT_SUCCESS;
  }

  std::map<std::size_t, std::size_t> cc_size_hist;
  for (const auto &cc : cc_sizes) {
    ++cc_size_hist[cc.second];
  }
  std::cout << "CC size histogram" << std::endl;
  std::cout << "CC size, count" << std::endl;
  for (const auto &cc : cc_size_hist) {
    std::cout << cc.first << " " << cc.second << std::endl;
  }

  if (cc_sizes.empty()) {
    return EXIT_SUCCESS;
  }

  std::cout << "Create output dir: " << output_dir << std::endl;
  std::filesystem::create_directory(output_dir);

  OMP_DIRECTIVE(parallel) {
    const auto range = partial_range(vertices.size(), mtl_omp::get_thread_num(),
                                     mtl_omp::get_num_threads());
    std::string name = "cc_table-" + std::to_string(mtl_omp::get_thread_num());
    auto        file_path = output_dir / name;
    std::ofstream ofs(file_path);
    if (!ofs) {
      std::cerr << "Failed to create: " << file_path << std::endl;
      std::exit(1);
    }
    for (std::size_t i = range.first; i < range.second; ++i) {
      const auto vid = vertices[i];
      ofs << vid << " " << cc_table[vid] << "\n";
    }
    ofs.close();
    if (!ofs) {
      std::cerr << "Failed to write: " << file_path << std::endl;
      std::exit(1);
    }
  }

  return EXIT_SUCCESS;
}
