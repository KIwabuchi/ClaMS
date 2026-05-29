// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <ygm/comm.hpp>

#include "../common.hpp"

namespace clams {

struct option_t {
  int         index_k{20};
  double      r{0.5};
  double      replicate_rate{0.0};  // Used for NEO-DNND only
  double      delta{0.0001};
  std::string distance_name;
  std::vector<std::filesystem::path> point_file_names;
  std::filesystem::path              scratchpath{"/dev/shm/dnnd_datastore"};
  std::filesystem::path              datastorepath{};
  std::string                        point_file_format;
  std::size_t                        batch_size{1ULL << 23};
  int                                num_threads{-1};  // For NEO-DNND
  bool                               verbose{false};
};

inline bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.distance_name.clear();
  opt.point_file_names.clear();
  opt.point_file_format.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:R:d:f:p:s:o:b:T:vh")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::stoi(optarg);
        break;

      case 'r':
        opt.r = std::stod(optarg);
        break;

      case 'R':
        opt.replicate_rate = std::stod(optarg);
        break;

      case 'd':
        opt.delta = std::stod(optarg);
        break;

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 's':
        opt.scratchpath = std::filesystem::path(optarg);
        break;

      case 'o':
        opt.datastorepath = std::filesystem::path(optarg);
        break;

      case 'b':
        opt.batch_size = std::stoul(optarg);
        break;

      case 'T':
        opt.num_threads = std::stoi(optarg);
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.index_k <= 0) {
    return false;
  }

  if (opt.replicate_rate < 0.0 || opt.replicate_rate > 1.0) {
    return false;
  }

  if (opt.distance_name.empty() || opt.point_file_format.empty() ||
      opt.point_file_names.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
inline void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: " << exe_name << " [options] point_file1 [point_file2 ...]"
       << std::endl;
  cout << "Options:" << std::endl;
  cout << "  -k <int>    kNNG k parameter (required)" << std::endl;
  cout << "  -f <string> Distance name (required). l1, l2, sql2, cosine, "
          "altcosine, jaccard, altjaccard, and levenshtein are supported."
       << std::endl;
  cout << "  -p <string> Point file format (required). wsv, wsv-id, csv, "
          "csv-id, str, and str-id are supported"
       << std::endl;
  cout << "  -r <float>  NN-Descent r parameter (default: 0.5)" << std::endl;
  cout << "  -R <float>  Replication rate in [0, 1] (default: 0.0)"
       << std::endl;
  cout << "  -d <float>  NN-Descent delta parameter (default: 0.001)"
       << std::endl;
  cout << "  -o <path>   If specified, copy the PM datastore from the "
          "scratchpad to to this path."
       << std::endl;
  cout << "  -s <path>   Path to datastore scratchpad. Hold a PM datastore "
          "during computation (default: /dev/shm/dnnd_datastore). Recommended "
          "to be on a fast storage, e.g., tmpfs (/dev/shm)."
       << std::endl;
  cout << "  -b <int>    Batch size (default: 1^23)" << std::endl;
  cout << "  -T <int>    Number of threads for NEO-DNND (optional)"
       << std::endl;
  cout << "  -v          Verbose mode" << std::endl;
  cout << "  -h          Show this message" << std::endl;
}

template <typename cout_type>
inline void show_options(const option_t &opt, cout_type &cout) {
  cout << "Options:" << std::endl;
  cout << "  k: " << opt.index_k << std::endl;
  cout << "  r: " << opt.r << std::endl;
  cout << "  replicate rate: " << opt.replicate_rate << std::endl;
  cout << "  delta: " << opt.delta << std::endl;
  cout << "  distance name: " << opt.distance_name << std::endl;
  cout << "  point file format: " << opt.point_file_format << std::endl;
  cout << "  datastore scratch path: " << opt.scratchpath << std::endl;
  cout << "  datastore path: " << opt.datastorepath << std::endl;
  cout << "  batch size: " << opt.batch_size << std::endl;
  cout << "  verbose: " << opt.verbose << std::endl;
}

template <typename id_type, typename feature_element_type,
          typename distance_type>
inline void show_config(ygm::comm &comm) {
  comm.welcome();
}
}  // namespace clams
