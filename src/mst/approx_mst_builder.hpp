// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <ygm/comm.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/detail/collective.hpp>
#include <ygm/utility/timer.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace amst {

template <typename VertexLabel, typename Weight>
class approx_mst_builder {
 public:
  using vlabel_type        = VertexLabel;
  using wgt_type           = Weight;
  using weighted_edge_type = std::tuple<vlabel_type, vlabel_type, wgt_type>;
  using result_type        = std::vector<weighted_edge_type>;

  approx_mst_builder(ygm::comm &comm, const float approx_bound,
                     const size_t cache_size, const wgt_type min_edge_wgt,
                     const wgt_type max_edge_wgt)
      : m_comm(comm),
        m_dset(comm, cache_size),
        m_approx_bound(approx_bound),
        m_local_result(),
        m_local_result_ptr(&m_local_result),
        m_edges_since_compress(0),
        m_min_edge_wgt(min_edge_wgt),
        m_max_edge_wgt(max_edge_wgt) {
    m_local_result_ptr.check(m_comm);

    if (min_edge_wgt != std::numeric_limits<wgt_type>::max() &&
        max_edge_wgt != std::numeric_limits<wgt_type>::min()) {
      assert(min_edge_wgt <= max_edge_wgt);
      assert(min_edge_wgt >= 0 && max_edge_wgt >= 0);

      m_initialized_limits = true;
      m_iteration          = 0;
      calculate_num_buckets();
      initialize_curr_limits();
    }
  }

  approx_mst_builder(ygm::comm &comm, const float approx_bound,
                     const wgt_type min_edge_wgt, const wgt_type max_edge_wgt)
      : approx_mst_builder(comm, approx_bound, 8192, min_edge_wgt,
                           max_edge_wgt) {}

  approx_mst_builder(ygm::comm &comm, const float approx_bound,
                     size_t cache_size)
      : approx_mst_builder(comm, approx_bound, cache_size,
                           std::numeric_limits<wgt_type>::max(),
                           std::numeric_limits<wgt_type>::min()) {}

  approx_mst_builder(ygm::comm &comm, const float approx_bound)
      : approx_mst_builder(comm, approx_bound, 8192) {}

  ~approx_mst_builder() { m_comm.barrier(); }

  size_t cache_hits() const { return m_dset.cache_hits(); }

  void add_vertex(const vlabel_type &vtx) { m_dset.async_union(vtx, vtx); }

  void try_add_edge(const vlabel_type &vtx1, const vlabel_type &vtx2,
                    const wgt_type w) {
    auto add_amst_edge_lambda = [](const auto &vtx1, const auto &vtx2,
                                   const bool union_result, const auto &wgt,
                                   auto local_result_ptr) {
      if (union_result) {
        local_result_ptr->push_back(weighted_edge_type(vtx1, vtx2, wgt));
      }
    };

    if (w < 0.0) {
      m_comm.cerr() << "Edge weights must be non-negative" << std::endl;
      exit(EXIT_FAILURE);
    }

    m_attempted_insert_edges_local = true;

    if (!m_initialized_limits) {
      // Avoid using a weight of 0 when finding logarithmic size of bins, but
      // allow weights of 0 when edges are being inserted
      if (w != 0.0) {
        update_local_wgt_limits(w);
      }
    } else {
      // Reset the union timer
      if (not m_union_timer_active) {
        m_union_timer_active = true;
        m_union_timer.reset();
      }

      if (m_curr_lower_wgt <= w && w < m_curr_upper_wgt) {
        m_dset.async_union_and_execute(vtx1, vtx2, add_amst_edge_lambda, w,
                                       m_local_result_ptr);
        ++m_edges_since_compress;
        ++m_iteration_num_edges;
      }
    }
  }

  bool is_done() {
    m_comm.barrier();

    m_last_iteration_num_edges = m_iteration_num_edges;

    // Stop union timer
    if (m_union_timer_active) {
      m_last_union_round_time = m_union_timer.elapsed();
      m_union_timer_active    = false;
    }

    if (!m_attempted_insert_edges_global) {
      m_attempted_insert_edges_global =
          logical_or(m_attempted_insert_edges_local, m_comm);
    }

    if (!m_attempted_insert_edges_global) {
      return false;
    } else if (!m_initialized_limits) {
      reduce_wgt_limits();
      calculate_num_buckets();
      initialize_curr_limits();

      m_initialized_limits = true;
      m_iteration          = 0;
      return false;
    } else {
      if (++m_iteration == m_num_buckets) {
        return true;
      } else {
        m_compress_timer.reset();
        if (ygm::sum(m_edges_since_compress, m_comm) / m_comm.size() >
            m_compression_threshold) {
          m_dset.all_compress();
        }
        m_comm.cf_barrier();
        m_last_compress_time = m_compress_timer.elapsed();

        m_curr_lower_wgt = m_curr_upper_wgt;
        m_curr_upper_wgt *= m_approx_bound;
        m_iteration_num_edges = 0;

        return false;
      }
    }
  }

  void dump_to_file(const std::string &file_prefix) {
    m_dset.dump_to_file(file_prefix);
  }

  void stats_print(std::ostream &os) { m_dset.stats_print(os); }

  result_type &get_result() { return m_local_result; }

  size_t num_trees() { return m_dset.num_sets(); }

  double get_last_union_time() const { return m_last_union_round_time; }

  double get_last_compression_time() const { return m_last_compress_time; }

  size_t get_last_iteration_num_edges() const {
    return m_last_iteration_num_edges;
  }

  int get_num_buckets() const { return m_num_buckets; }

  int get_current_bucket_id() const { return m_iteration; }

  size_t num_vertices_seen() { return m_dset.size(); }

 private:
  void update_local_wgt_limits(const wgt_type w) {
    m_min_edge_wgt = std::min(w, m_min_edge_wgt);
    m_max_edge_wgt = std::max(w, m_max_edge_wgt);

    YGM_ASSERT_RELEASE(m_min_edge_wgt <= w);
    YGM_ASSERT_RELEASE(m_max_edge_wgt >= w);
  }

  void reduce_wgt_limits() {
    m_comm.barrier();
    m_min_edge_wgt = ygm::min(m_min_edge_wgt, m_comm);
    m_max_edge_wgt = ygm::max(m_max_edge_wgt, m_comm);
  }

  void calculate_num_buckets() {
    m_num_buckets = (int)ceil((log(m_max_edge_wgt) - log(m_min_edge_wgt)) /
                              log(m_approx_bound));

    if (m_num_buckets < 1) {
      m_num_buckets = 1;
    }
  }

  void initialize_curr_limits() {
    m_curr_lower_wgt = 0.0;
    m_curr_upper_wgt = m_min_edge_wgt * m_approx_bound;
  }

  ygm::comm                                &m_comm;
  ygm::container::disjoint_set<vlabel_type> m_dset;

  bool m_initialized_limits            = false;
  bool m_attempted_insert_edges_local  = false;
  bool m_attempted_insert_edges_global = false;

  wgt_type m_min_edge_wgt = std::numeric_limits<wgt_type>::max();
  wgt_type m_max_edge_wgt = std::numeric_limits<wgt_type>::min();

  int m_num_buckets = 0;
  int m_iteration   = -1;

  float m_approx_bound;

  wgt_type m_curr_lower_wgt;
  wgt_type m_curr_upper_wgt;

  size_t m_iteration_num_edges      = 0;
  size_t m_last_iteration_num_edges = 0;
  size_t m_edges_since_compress;
  size_t m_compression_threshold = 256;

  ygm::utility::timer m_union_timer;
  ygm::utility::timer m_compress_timer;
  double              m_last_union_round_time{0.0};
  double              m_last_compress_time{0.0};
  bool                m_union_timer_active{false};

  result_type               m_local_result;
  ygm::ygm_ptr<result_type> m_local_result_ptr;
};

}  // namespace amst
