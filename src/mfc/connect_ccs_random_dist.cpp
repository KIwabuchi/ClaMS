// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.


// Connect connected components by adding random bridge edges.

#define CLAMS_USE_SALTATLAS

#include <unistd.h>

#include <filesystem>
#include <iostream>

#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>
#include <ygm/utility/timer.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "../common.hpp"
#include "dist_cc.hpp"

namespace cls = clams;

void show_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << " -d <pm_knng_datastore_path> -f <distance_name> [-b "
               "<bridge_edge_dump_file>]"
            << std::endl;
  std::cout << "  -d <path>: Path to the distributed PM kNNG datastore."
            << std::endl;
  std::cout << "  -f <string>: Distance function name (e.g., l2, cosine)."
            << std::endl;
  std::cout << "  -b <path>: (Optional) File to dump the bridge edges."
            << std::endl;
}

bool parse_option(int argc, char *argv[],
                  std::filesystem::path &pm_knng_datastore_path,
                  std::string &distance_name,
                  std::string &bridge_edge_dump_file) {
  int opt;
  pm_knng_datastore_path.clear();
  distance_name.clear();

  while ((opt = getopt(argc, argv, "d:f:b:")) != -1) {
    switch (opt) {
      case 'd':
        pm_knng_datastore_path = std::filesystem::path(optarg);
        break;
      case 'f':
        distance_name = optarg;
        break;
      case 'b':
        bridge_edge_dump_file = optarg;
        break;
      default:
        return false;
    }
  }

  if (pm_knng_datastore_path.empty()) {
    std::cerr << "Datastore path is required (-d)." << std::endl;
    return false;
  }
  if (!std::filesystem::exists(pm_knng_datastore_path)) {
    std::cerr << "Datastore does not exist: " << pm_knng_datastore_path
              << std::endl;
    return false;
  }
  if (distance_name.empty()) {
    std::cerr << "Distance function is required (-f)." << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  std::filesystem::path pm_knng_datastore_path;
  std::string distance_name;
  std::string bridge_edge_dump_file;
  const bool opt_parse_ret = parse_option(argc, argv, pm_knng_datastore_path,
                                          distance_name, bridge_edge_dump_file);
  if (!opt_parse_ret) {
    if (comm.rank0()) {
      show_usage(argv[0]);
    }
    return EXIT_FAILURE;
  }

  comm.cout0() << "PM datastore path\t" << pm_knng_datastore_path << std::endl;
  comm.cout0() << "Distance name\t" << distance_name << std::endl;

  ygm::utility::timer root_timer;
  {
    cls::dist_pm_knng_t pm_knng(comm.get_mpi_comm());
    pm_knng.open(pm_knng_datastore_path);

    static auto &knng   = pm_knng.get_knng();
    static auto &pstore = pm_knng.get_point_store();

    ygm::utility::timer cc_timer;
    dist_cc cc(comm, knng);
    cc.run_cc();
    const auto cc_size_table = cc.count_cc_size();
    assert(comm.rank0() || cc_size_table.empty());
    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << cc_size_table.size() << std::endl;

    id_t largest_cc_id = -1;
    std::size_t largest_cc_size = 0;
    for (const auto &cc : cc_size_table) {
      comm.cout0() << "CC ID: " << cc.first << ", Size: " << cc.second
                   << std::endl;
      if (cc.second > largest_cc_size) {
        largest_cc_size = cc.second;
        largest_cc_id = cc.first;
      }
    }
    comm.cout0() << "Largest CC's size: " << largest_cc_size
                 << ", ID: " << largest_cc_id << std::endl;

    comm.cout0() << "Connect small CCs to the largest CC" << std::endl;
    ygm::utility::timer connect_timer;
    static auto ref_distance_func =
        saltatlas::distance::distance_function<cls::point_t, cls::distance_t>(
            distance_name);
    static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
        local_bridge_edges;
    static const auto &ref_pstore = pstore;
    for (const auto &cc : cc_size_table) {
      const auto &pid = cc.first;
      if (pid == largest_cc_id) {
        continue;  // Skip the largest CC
      }

      comm.async(
          cls::dist_pm_knng_t::get_owner(largest_cc_id, comm.size()),
          [](auto *c, const cls::id_t &large, const cls::id_t &small) {
            c->async(
                cls::dist_pm_knng_t::get_owner(small, c->size()),
                [](const cls::id_t &large, const cls::id_t &small,
                   const auto &large_fv) {
                  const auto point_itr = ref_pstore.find(small);
                  if (point_itr == ref_pstore.end()) {
                    std::cerr << "Cannot find point for ID: " << small
                              << std::endl;
                    std::abort();
                  }
                  const auto &small_fv = point_itr->second;
                  const auto d = ref_distance_func(large_fv, small_fv);
                  knng[small].emplace_back(large, d);
                  // Store the bridge edge
                  local_bridge_edges.emplace_back(large, small, d);
                },
                large, small, ref_pstore.at(large));
          },
          largest_cc_id, pid);
    }
    comm.barrier();
    comm.cout0() << "Connected CCs took (s): " << connect_timer.elapsed()
                 << std::endl;
    comm.cout0() << "Entier algorithm took (s): " << root_timer.elapsed()
                 << std::endl;

    if (!bridge_edge_dump_file.empty()) {
      comm.cout0() << "Gather bridge edges to rank 0" << std::endl;
      // Gather bridge edges from all ranks to rank 0
      static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
          all_bridge_edges;
      comm.cf_barrier();
      comm.async(
          0,
          [](const auto &edges) {
            all_bridge_edges.insert(all_bridge_edges.end(), edges.begin(),
                                    edges.end());
          },
          local_bridge_edges);
      comm.barrier();

      // Print bridge edges
      if (comm.rank0()) {
        comm.cout0() << "Dump bridge edges to " << bridge_edge_dump_file
                     << std::endl;
        std::ofstream ofs(bridge_edge_dump_file);
        if (!ofs.is_open()) {
          comm.cerr0() << "Cannot open file: " << bridge_edge_dump_file
                       << std::endl;
          return EXIT_FAILURE;
        }
        for (const auto &edge : all_bridge_edges) {
          ofs << std::get<0>(edge) << "\t" << std::get<1>(edge) << "\t"
              << std::get<2>(edge) << std::endl;
        }
      }
    }
    comm.cf_barrier();
  }
  comm.cout0() << "Done" << std::endl;

  return EXIT_SUCCESS;
}
