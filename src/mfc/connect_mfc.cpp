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
                  std::string           &distance_name,
                  std::string           &bridge_edge_dump_file) {
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
  std::string           distance_name;
  std::string           bridge_edge_dump_file;
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

    comm.cout0() << "Opened" << std::endl;
    std::cout.flush();

    ygm::utility::timer cc_timer;

    ygm::container::disjoint_set<cls::id_t> disjoint_set(comm);
    for (const auto &[source, neighbors] : knng) {
      for (const auto &neighbor : neighbors) {
        disjoint_set.async_union(source, neighbor.id);
      }
    }

    comm.cout0() << "Read all points" << std::endl;
    std::cout.flush();

    comm.barrier();

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

    disjoint_set.for_all([&](const auto &p, const auto &rep) {
      comm.async(
          cls::dist_pm_knng_t::get_owner(p, comm.size()),
          [](cls::id_t p, cls::id_t rep) {
            auto point = pstore.find(p);
            if (point == pstore.end()) {
              std::cerr << "Cannot find point for ID: " << p << std::endl;
              std::abort();
            }
            s_bag_map->at(rep)->local_insert({p, point->second});
          },
          p, rep);
    });

    comm.barrier();

    for (auto &b : bags) {
      if (b->size() > (size_t)comm.size()) b->rebalance();
    }

    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << bags.size() << std::endl;

    // for (auto &b : bags) {
    //   size_t s = b.size();
    //   comm.cout0() << "    CC: size " << s << " local: " << b.local_size() <<
    //   std::endl;
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
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          auto dist = ref_distance_func(a.second, b.second);
          comm.async(
              cls::dist_pm_knng_t::get_owner(a.first, comm.size()),
              [](cls::id_t a, cls::id_t b, cls::distance_t dist) {
                local_bridge_edges.emplace_back(a, b, dist);
                knng[a].emplace_back(b, dist);
              },
              a.first, b.first, dist);
        });

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
