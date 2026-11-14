// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// Read kNNG constructed by DNND and compute mutual reachability distances
// (MRD).

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../common.hpp"
#include "../details/shm_graph.hpp"

bool parse_option(int argc, char *argv[],
                  std::filesystem::path &input_knng_path, int &min_samples,
                  std::filesystem::path &output_knn_path) {
  input_knng_path.clear();
  output_knn_path.clear();
  min_samples = -1;

  int opt;
  while ((opt = ::getopt(argc, argv, "i:m:o:")) != -1) {
    switch (opt) {
      case 'i': {
        input_knng_path = std::filesystem::path(optarg);
        break;
      }
      case 'm': {
        min_samples = std::atoi(optarg);
        break;
      }
      case 'o': {
        output_knn_path = std::filesystem::path(optarg);
        break;
      }
      default: {
        std::cerr << "Unknown option: " << opt << std::endl;
        return false;
      }
    }
  }

  if (input_knng_path.empty()) {
    std::cerr << "No input directory is specified" << std::endl;
    return false;
  }

  if (min_samples < 0) {
    std::cerr << "No minimum samples is specified" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  std::filesystem::path input_knng_path;
  int                   min_samples = 0;
  std::filesystem::path output_knn_path;

  if (!parse_option(argc, argv, input_knng_path, min_samples,
                    output_knn_path)) {
    return EXIT_FAILURE;
  }

  const auto knng_files = clams::find_files(input_knng_path);

  clams::shm_graph_t graph;

  std::cout << "Read knng" << std::endl;
  clams::read_knng(knng_files, graph);
  std::cout << "#of vertices: " << graph.num_keys() << std::endl;
  std::cout << "#of edges: " << graph.num_values() << std::endl;

  clams::shm_graph_t mrd_graph(graph);
  std::vector<id_t>  vertices;
  for (auto vit = graph.keys_begin(); vit != graph.keys_end(); ++vit) {
    vertices.push_back(vit->first);
  }
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    const auto vid = vertices[i];
    for (auto eit = mrd_graph.values_begin(vid);
         eit != mrd_graph.values_end(vid); ++eit) {
      const auto nid = eit->first;
      const auto core_dist_a =
          (graph.values_begin(vid) + min_samples - 1)->second;
      const auto core_dist_b =
          (graph.values_begin(nid) + min_samples - 1)->second;
      auto      &dist = eit->second;  // original distance
      const auto mrd  = std::max<clams::distance_t>(
          std::max<clams::distance_t>(core_dist_a, core_dist_b), dist);
      // std::cout << mrd << " == max " << core_dist_a << ", " << core_dist_b
      //           << ", " << dist << std::endl;
      dist = mrd;
    }
  }

  clams::dump_graph(mrd_graph, output_knn_path);
  std::cout << "Dumped mrd-knng to " << output_knn_path << std::endl;

  return EXIT_SUCCESS;
}
