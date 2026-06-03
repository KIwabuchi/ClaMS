// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>

#include "../common.hpp"

class dist_cc {
 private:
  using self_t = dist_cc;
  using input_knng_t = clams::pm_knng_t;
  using work_knng_t =
      std::unordered_map<clams::id_t, std::vector<clams::neighbor_t>>;

 public:
  using cc_table_t = boost::unordered_flat_map<clams::id_t, clams::id_t>;

  dist_cc(ygm::comm &world, const input_knng_t &knng)
      : m_world(world), m_knng(knng) {}

  const cc_table_t &run_cc() {
    m_world.cout0() << "Running weakly connected components algorithm."
                    << std::endl;
    {
      const auto uknng = priv_make_undirected_graph();
      m_world.cout0() << "Converted to undirected knng with "
                      << m_world.all_reduce_sum(uknng.size()) << " points and "
                      << m_world.all_reduce_sum(count_neighbors(uknng))
                      << " edges." << std::endl;
      priv_run_cc(uknng);
    }
    m_world.cout0() << "Connected components algorithm finished." << std::endl;

    return get_cc_table();
  }

  // Each rank has its own CC table
  const cc_table_t &get_cc_table() const { return m_cc_table; }

  // Only rank 0 has result
  std::unordered_map<clams::id_t, size_t> count_cc_size() {
    return priv_count_cc_size();
  }

 private:
  struct cc_visitor {
    template <typename comm_t>
    void operator()(comm_t *comm, auto this_ptr, const clams::id_t &pid,
                    const clams::id_t &cc_id) {
      auto &cc_table = this_ptr->m_cc_table;
      if (cc_table.count(pid) == 0) {
        comm->cerr() << "CC table does not contain point ID: " << pid
                     << std::endl;
        MPI_Abort(comm->get_mpi_comm(), EXIT_FAILURE);
      }
      if (cc_table.at(pid) <= cc_id) {
        return;
      }
      cc_table.at(pid) = cc_id;
      assert(this_ptr->m_ref_knng != nullptr);
      const auto itr = this_ptr->m_ref_knng->find(pid);
      if (itr == this_ptr->m_ref_knng->end()) {
        return;
      }
      for (const auto &neighbor : itr->second) {
        comm->async(clams::dist_pm_knng_t::get_owner(neighbor.id, comm->size()),
                    cc_visitor{}, this_ptr->m_self, neighbor.id, cc_id);
      }
    }
  };

  static std::size_t count_neighbors(const work_knng_t &knng) {
    std::size_t count = 0;
    for (const auto &[id, neighbors] : knng) {
      (void)id;
      count += neighbors.size();
    }
    return count;
  }

  work_knng_t priv_make_undirected_graph() {
    work_knng_t uknng;
    uknng.reserve(m_knng.size());
    m_world.cf_barrier();
    auto uknng_ptr = m_world.make_ygm_ptr(uknng);

    // Make knng undirected
    for (const auto &[source, neighbors] : m_knng) {
      for (const auto &neighbor : neighbors) {
        m_world.async(
            clams::dist_pm_knng_t::get_owner(neighbor.id, m_world.size()),
            [](const auto ptr, const clams::id_t src, const clams::neighbor_t ngbr) {
              (*ptr)[ngbr.id].emplace_back(src, ngbr.distance);
            },
            uknng_ptr, source, neighbor);
      }
    }
    m_world.barrier();

    for (const auto &[source, neighbors] : m_knng) {
      auto &dst_neighbors = uknng[source];
      for (const auto &neighbor : neighbors) {
        dst_neighbors.emplace_back(neighbor);
      }
    }

    for (auto &[source, neighbors] : uknng) {
      (void)source;
      std::sort(neighbors.begin(), neighbors.end(),
                [](const clams::neighbor_t &lhs, const clams::neighbor_t &rhs) {
                  if (lhs.id != rhs.id) return lhs.id < rhs.id;
                  return lhs.distance < rhs.distance;
                });
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end(),
                                  [](const clams::neighbor_t &lhs,
                                     const clams::neighbor_t &rhs) {
                                    return lhs.id == rhs.id;
                                  }),
                      neighbors.end());
    }

    return uknng;
  }

  void priv_run_cc(const work_knng_t &knng) {
    m_ref_knng = &knng;

    m_world.cout0() << "Init CC table" << std::endl;
    // Init CC table
    m_cc_table.clear();
    m_cc_table.reserve(knng.size());
    for (const auto &[source, neighbors] : knng) {
      (void)neighbors;
      m_cc_table[source] = source;  // Initialize CC table
    }
    m_world.cf_barrier();

    m_world.cout0() << "Start CC algorithm" << std::endl;

    // Launch CC algorithm
    for (const auto &[source, neighbors] : knng) {
      const auto cc_id = m_cc_table.at(source);
      if (cc_id != source) {
        // Already processed
        continue;
      }
      for (const auto &neighbor : neighbors) {
        m_world.async(
            clams::dist_pm_knng_t::get_owner(neighbor.id, m_world.size()),
            cc_visitor{}, m_self, neighbor.id, cc_id);
      }
    }
    m_world.barrier();
    m_world.cout0() << "CC algorithm finished" << std::endl;
  }

  std::unordered_map<clams::id_t, size_t> priv_count_cc_size() {
    std::unordered_map<clams::id_t, size_t> l_cc_size_table;
    for (const auto &cc : m_cc_table) {
      const auto cc_id = cc.second;
      if (l_cc_size_table.count(cc_id) == 0) {
        l_cc_size_table[cc_id] = 0;
      }
      ++l_cc_size_table[cc_id];
    }

    static std::unordered_map<clams::id_t, size_t> g_cc_size_table;
    g_cc_size_table.clear();
    m_world.cf_barrier();

    m_world.async(
        0,
        [](const auto &table) {
          for (const auto &entry : table) {
            const auto cc_id = entry.first;
            const auto size  = entry.second;
            if (g_cc_size_table.count(cc_id) == 0) {
              g_cc_size_table[cc_id] = 0;
            }
            g_cc_size_table[cc_id] += size;
          }
        },
        l_cc_size_table);
    m_world.barrier();

    return std::move(g_cc_size_table);
  }

  ygm::comm           &m_world;
  const input_knng_t  &m_knng;  // Original KNNG
  cc_table_t           m_cc_table{};
  ygm::ygm_ptr<self_t> m_self{this};
  // KNNG used for CC, could be undirected or directed
  const work_knng_t *m_ref_knng{nullptr};
};
