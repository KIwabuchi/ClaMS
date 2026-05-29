// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.


#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <utility>

#include <metall/metall.hpp>

#include "../common.hpp"

using namespace clams;

void parse_option(int argc, char *argv[], std::filesystem::path &pm_knng_path,
                  std::filesystem::path &output_path) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:h")) != -1) {
    switch (opt_char) {
      case 'i':
        pm_knng_path = std::filesystem::path(optarg);
        break;
      case 'o':
        output_path = std::filesystem::path(optarg);
        break;
      case 'h':
        std::cout << "Usage: " << argv[0]
                  << " -i <pm_knng_path> -o <output_path>" << std::endl;
        std::exit(0);
      default:
        std::cerr << "Unknown option: " << static_cast<char>(opt_char)
                  << std::endl;
        std::exit(1);
    }
  }

  if (pm_knng_path.empty()) {
    std::cerr << "No input PM KNNG path is specified (-i)." << std::endl;
    std::exit(1);
  }

  if (output_path.empty()) {
    std::cerr << "No output path is specified (-o)." << std::endl;
    std::exit(1);
  }
}

int main(int argc, char *argv[]) {
  ygm::comm comm(&argc, &argv);

  std::filesystem::path pm_knng_path;
  std::filesystem::path output_path;
  parse_option(argc, argv, pm_knng_path, output_path);

  dist_pm_knng_t pm_knng(comm.get_mpi_comm());
  pm_knng.open_read_only(pm_knng_path);
  const auto &knng = pm_knng.get_knng();

  std::filesystem::create_directories(output_path);
  const auto local_output_file =
      output_path / ("knng-" + std::to_string(comm.rank()) + ".txt");

  std::ofstream ofs(local_output_file);
  if (!ofs) {
    std::cerr << "Failed to open output file: " << local_output_file
              << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto &[source, neighbors] : knng) {
    ofs << source;
    for (const auto &neighbor : neighbors) {
      ofs << " " << neighbor.id;
    }
    ofs << "\n";

    ofs << "0.0";
    for (const auto &neighbor : neighbors) {
      ofs << " " << neighbor.distance;
    }
    ofs << "\n";
  }

  comm.cf_barrier();
  comm.cout0("Finished dumping KNNG to: ", output_path.string());

  return 0;
}
