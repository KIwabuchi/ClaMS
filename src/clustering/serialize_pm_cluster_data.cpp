// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

// This program serializes PM cluster IDs to a text file.

#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <spdlog/spdlog.h>

#include "../common.hpp"

using namespace clams;

struct option {
  std::filesystem::path cluster_data_input_path;
  std::filesystem::path cluster_data_out_path;
};

void show_help(const char* program) {
  std::cout << "Usage: " << program
            << " -i <cluster_ids_metall_path> -o <output_file>\n"
               "  -i  Input Metall datastore containing point-to-cluster IDs\n"
               "  -o  Output text file (point ID and cluster ID per line)\n"
               "  -h  Show this help\n";
}

void parse_option(int argc, char* argv[], option& opt) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:h")) != -1) {
    switch (opt_char) {
      case 'i':
        opt.cluster_data_input_path = optarg;
        break;
      case 'o':
        opt.cluster_data_out_path = optarg;
        break;
      case 'h':
        show_help(argv[0]);
        std::exit(EXIT_SUCCESS);
      default:
        show_help(argv[0]);
        std::exit(EXIT_FAILURE);
    }
  }

  if (optind != argc || opt.cluster_data_input_path.empty() ||
      opt.cluster_data_out_path.empty()) {
    std::cerr << "Both -i and -o are required, with no positional arguments.\n";
    show_help(argv[0]);
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char* argv[]) {
  option opt;
  parse_option(argc, argv, opt);

  pm_point_to_cluster_id_map_t point_cluster_map;

  {
    spdlog::info("Attaching cluster IDs in Metall datastore");
    metall::manager metall_manager(metall::open_read_only,
                                   opt.cluster_data_input_path);
    const auto*     input_cluster_id_list =
        metall_manager
            .find<pm_point_to_cluster_id_list_t>("point_to_cluster_id_list")
            .first;
    if (!input_cluster_id_list) {
      spdlog::critical("Failed to find cluster IDs in Metall datastore at {}",
                       opt.cluster_data_input_path.string());
      std::abort();
    }

    point_cluster_map.reserve(input_cluster_id_list->size());
    point_cluster_map.insert(input_cluster_id_list->begin(),
                             input_cluster_id_list->end());
  }

  dump_point_cluster_ids(point_cluster_map, opt.cluster_data_out_path);
  spdlog::info("Serialized {} point-to-cluster IDs to {}",
               point_cluster_map.size(), opt.cluster_data_out_path.string());
  return EXIT_SUCCESS;
}
