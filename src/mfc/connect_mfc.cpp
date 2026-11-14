// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#include <algorithm>
#include <iterator>
#include <vector>
#define CLAMS_USE_SALTATLAS

#include <unistd.h>

#include <filesystem>
#include <iostream>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/utility/timer.hpp>

#include "../common.hpp"
#include "mfc.h"

namespace cls = clams;

void show_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << " -d <pm_knng_datastore_path> -f <distance_name> -b "
               "<bridge_edge_output_file>"
            << std::endl;
  std::cout << "  -d <path>: Path to the distributed PM kNNG datastore."
            << std::endl;
  std::cout << "  -f <string>: Distance function name (e.g., l2, cosine)."
            << std::endl;
  std::cout << "  -b <path>: Path to text file to write the bridge edges."
            << std::endl;
}

bool parse_option(int argc, char *argv[],
                  std::filesystem::path &pm_knng_datastore_path,
                  std::string           &distance_name,
                  std::string           &bridge_edge_output_file) {
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
        bridge_edge_output_file = optarg;
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

template <typename point_t>
const std::pair<int, int> get_new_owner(const point_t &point,
                                        const int      original_comm_size,
                                        const int      new_comm_size) {
  const int original_owner =
      cls::dist_pm_knng_t::get_owner(point, original_comm_size);
  const int new_owner   = original_owner % new_comm_size;
  const int local_index = original_owner / new_comm_size;
  return {new_owner, local_index};
}

int main(int argc, char **argv) {
  ygm::comm        comm(&argc, &argv);
  static const int comm_size = comm.size();
  static const int comm_rank = comm.rank();

  std::filesystem::path pm_knng_datastore_path;
  std::string           distance_name;
  std::string           bridge_edge_output_file;
  const bool            opt_parse_ret =
      parse_option(argc, argv, pm_knng_datastore_path, distance_name,
                   bridge_edge_output_file);
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
    const auto local_paths = cls::dist_pm_knng_t::get_local_datastore_paths(
        pm_knng_datastore_path, comm.get_mpi_comm());

    std::vector<metall::manager> local_managers;
    for (std::size_t r = 0; r < local_paths.size(); ++r) {
      if (r % comm_size == comm_rank) {
        if (!metall::manager::consistent(local_paths[r])) {
          comm.cerr() << "Inconsistent local managers at rank " << comm_rank
                      << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }
        local_managers.emplace_back(metall::open_read_only, local_paths[r]);
        if (!local_managers.back().check_sanity()) {
          comm.cerr() << "Failed to open local manager at rank " << comm_rank
                      << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }
      }
    }
    comm.cout0() << "Opened PM datastores" << std::endl;
    std::cout.flush();

    ygm::utility::timer cc_timer;

    ygm::container::disjoint_set<cls::id_t> disjoint_set(comm);
    for (auto &local_manager : local_managers) {
      auto *knng = cls::dist_pm_knng_t::find_knng(local_manager);
      if (!knng) {
        comm.cerr() << "Failed to find kNNG" << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }

      for (const auto &[source, neighbors] : *knng) {
        for (const auto &neighbor : neighbors) {
          disjoint_set.async_union(source, neighbor.id);
        }
      }
    }
    comm.barrier();
    comm.cout0() << "Finished reading kNNGs" << std::endl;
    std::cout.flush();

    disjoint_set.all_compress();

    std::set<cls::id_t> rep_set;
    disjoint_set.for_all(
        [&](const auto &p, const auto &rep) { rep_set.insert(rep); });
    rep_set = ygm::all_reduce(
        rep_set,
        [](std::set<cls::id_t> a, std::set<cls::id_t> b) {
          std::set<cls::id_t> res;
          std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                         std::inserter(res, res.end()));
          return res;
        },
        comm);

    comm.cout0() << "Found reps" << std::endl;
    std::cout.flush();

    std::vector<std::shared_ptr<
        ygm::container::bag<std::pair<cls::id_t, cls::point_t>>>>
        bags;
    std::map<cls::id_t, std::shared_ptr<ygm::container::bag<
                            std::pair<cls::id_t, cls::point_t>>>>
        bag_map;
    static std::map<cls::id_t, std::shared_ptr<ygm::container::bag<
                                   std::pair<cls::id_t, cls::point_t>>>>
        *s_bag_map;
    s_bag_map = &bag_map;

    bags.reserve(rep_set.size());
    for (auto &rep : rep_set) {
      bags.emplace_back(
          new ygm::container::bag<std::pair<cls::id_t, cls::point_t>>(comm));
      bag_map.emplace(rep, bags.back());
    }

    static std::vector<cls::dist_pm_knng_t::point_store_type *> pstores;
    for (auto &local_manager : local_managers) {
      auto *pstore = cls::dist_pm_knng_t::find_point_store(local_manager);
      if (!pstore) {
        comm.cerr() << "Failed to find point store" << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
      pstores.push_back(pstore);
    }

    const int original_comm_size = static_cast<int>(local_paths.size());
    disjoint_set.for_all([&](const auto &p, const auto &rep) {
      const auto [new_owner, local_index] =
          get_new_owner(p, original_comm_size, comm_size);
      comm.async(
          new_owner,
          [](const int local_index, cls::id_t p, cls::id_t rep) {
            auto point = pstores[local_index]->find(p);
            if (point == pstores[local_index]->end()) {
              std::cerr << "Cannot find point for ID: " << p << std::endl;
              std::abort();
            }
            // A copy of the PM vector retains its Metall allocator, which
            // cannot allocate when the datastore is open read only.
            cls::point_t heap_point(point->second.begin(), point->second.end());
            s_bag_map->at(rep)->local_insert({p, std::move(heap_point)});
          },
          local_index, p, rep);
    });

    comm.barrier();

    for (auto &b : bags) {
      if (b->size() > (size_t)comm_size) b->rebalance();
    }

    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << bags.size() << std::endl;

    // for (auto &b : bags) {
    //   size_t s = b.size();
    //   comm.cout0() << "    CC: size " << s << " local: " << b.local_size()
    //   << std::endl;
    // }

    static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
        local_bridge_edges;

    ygm::utility::timer connect_timer;
    auto                ref_distance_func =
        saltatlas::distance::distance_function<cls::point_t, cls::distance_t>(
            distance_name);
    mfc(
        comm, bags,
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          return ref_distance_func(a.second, b.second);
        },
        [&](std::pair<cls::id_t, cls::point_t> &a,
            std::pair<cls::id_t, cls::point_t> &b) {
          auto dist = ref_distance_func(a.second, b.second);
          local_bridge_edges.emplace_back(a.first, b.first, dist);
        });

    comm.barrier();
    comm.cout0() << "Connected CCs took (s): " << connect_timer.elapsed()
                 << std::endl;
    comm.cout0() << "Entire algorithm took (s): " << root_timer.elapsed()
                 << std::endl;

    if (!bridge_edge_output_file.empty()) {
      std::ofstream ofs;
      if (comm.rank() == 0) {
        ofs.open(bridge_edge_output_file);
        if (!ofs.is_open()) {
          std::cerr << "Failed to open bridge edge output file: "
                    << bridge_edge_output_file << std::endl;
          MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
        }
      }
      for (auto &edge : local_bridge_edges) {
        comm.async(
            0,
            [&](const cls::id_t a, const cls::id_t b, const cls::distance_t d) {
              ofs << a << " " << b << " " << d << "\n";
            },
            std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
      }
      comm.barrier();
      ofs.close();
      if (comm.rank() == 0 && !ofs) {
        std::cerr << "Failed to write to bridge edge output file: "
                  << bridge_edge_output_file << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
    }
    comm.cout0() << "Wrote bridge edges to " << bridge_edge_output_file
                 << std::endl;
    comm.cf_barrier();
  }
  comm.cout0() << "Done" << std::endl;

  return EXIT_SUCCESS;
}
