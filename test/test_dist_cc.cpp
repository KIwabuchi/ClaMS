// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.


// Connect connected components by adding random bridge edges.

#define CLAMS_USE_SALTATLAS

#include <filesystem>
#include <iostream>

#include <ygm/comm.hpp>

#include "../src/common.hpp"
#include "../src/mfc/dist_cc.hpp"

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  clams::pm_knng_t knng;

  // Sample graph
  // 10 points, 3 connected components
  // 9 directed edges, 16 undirected edges
  std::vector<std::pair<int, int>> edges = {
      {0, 1}, {1, 2}, {2, 1},  // Component 1
      {3, 4}, {4, 5}, {5, 3},  // Component 2
      {9, 6}, {9, 7}, {9, 8}   // Component 3
  };

  for (const auto &edge : edges) {
    if (clams::dist_pm_knng_t::get_owner(edge.first, world.size()) ==
        world.rank()) {
      knng[edge.first].emplace_back(edge.second, 1.0f);
    }
  }
  world.cf_barrier();

  dist_cc cc(world, knng);
  world.cf_barrier();

  const auto &cc_table = cc.run_cc();
  {
    auto result_checker = [&cc_table](const pid_t pid,
                                      const pid_t expected_cc_id) {
      if (cc_table.at(pid) != expected_cc_id) {
        std::cerr << "CC ID mismatch for point " << pid << ": "
                  << cc_table.at(pid) << " != " << expected_cc_id << std::endl;
        std::abort();
      }
    };

    for (pid_t pid = 0; pid < 10; ++pid) {
      if (clams::dist_pm_knng_t::get_owner(pid, world.size()) == world.rank()) {
        if (pid <= 2) {
          result_checker(pid, 0);  // Component 1
        } else if (pid <= 5) {
          result_checker(pid, 3);  // Component 2
        } else {
          result_checker(pid, 6);  // Component 3
        }
      }
    }
  }

  const auto cc_size_table = cc.count_cc_size();
  if (world.rank0()) {
    auto result_checker = [&cc_size_table](const id_t cc_id,
                                           const size_t expected_size) {
      if (cc_size_table.at(cc_id) != expected_size) {
        std::cerr << "CC ID " << cc_id
                  << " size mismatch: " << cc_size_table.at(cc_id)
                  << " != " << expected_size << std::endl;
        std::abort();
      }
    };

    for (const auto &cc_size : cc_size_table) {
      const auto cc_id = cc_size.first;
      const auto size = cc_size.second;
      if (cc_id == 0) {
        result_checker(cc_id, 3);  // Component 1
      } else if (cc_id == 3) {
        result_checker(cc_id, 3);  // Component 2
      } else if (cc_id == 6) {
        result_checker(cc_id, 4);  // Component 3
      } else {
        std::cerr << "Unexpected CC ID: " << cc_id << std::endl;
        std::abort();
      }
    }
  }

  world.cout0() << "Test succeeded!!" << std::endl;

  return EXIT_SUCCESS;
}
