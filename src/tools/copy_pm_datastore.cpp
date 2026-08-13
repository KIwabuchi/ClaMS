// Copyright 2026 Lawrence Livermore National Security, LLC and other Metall
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

// This program copies a Metall datastore created by the metall_mpi_adaptor
// utility to a new location. It is useful for copying a large datastore because
// it can copy the datastore in parallel using multiple MPI processes.

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <ygm/comm.hpp>

#include <metall/utility/metall_mpi_adaptor.hpp>

namespace {

bool parse_options(int argc, char **argv, std::filesystem::path &datastore_path,
                   std::filesystem::path &copy_path, bool &overwrite) {
  overwrite = false;
  int option;
  while ((option = ::getopt(argc, argv, "s:t:o")) != -1) {
    switch (option) {
      case 's':
        datastore_path = optarg;
        break;
      case 't':
        copy_path = optarg;
        break;
      case 'o':
        overwrite = true;
        break;
      default:
        return false;
    }
  }

  return !datastore_path.empty() && !copy_path.empty() && optind == argc;
}

}  // namespace

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  {
    std::filesystem::path datastore_path;
    std::filesystem::path copy_path;
    bool                  overwrite = false;

    if (!parse_options(argc, argv, datastore_path, copy_path, overwrite)) {
      comm.cerr0() << "Usage: " << argv[0]
                   << " -s <source_datastore_path> -t <target_copy_path>"
                      " [-o (overwrite)]"
                   << std::endl;
      return EXIT_FAILURE;
    }

    comm.cout0() << "Copying a Metall datastore from " << datastore_path
                 << " to " << copy_path << ", overwrite=" << overwrite
                 << std::endl;

    const bool ret = metall::utility::metall_mpi_adaptor::copy(
        datastore_path, copy_path, comm.get_mpi_comm(), overwrite);
    comm.cf_barrier();

    if (ret) {
      comm.cout0() << "Done." << std::endl;
    } else {
      comm.cerr0() << "Failed to copy the Metall datastore." << std::endl;
    }
  }

  return 0;
}
