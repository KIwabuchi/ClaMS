// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include <mpi.h>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>

#include "global_index.h"
#include "mst.h"

struct TreeEdge {
  size_t      a;
  size_t      b;
  GlobalIndex a_index;
  GlobalIndex b_index;
  double      weight;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(a, b, a_index, b_index, weight);
  }
};

/*
 * Runs the MFC algorithm, edges are returned by calling edge_callback on rank 0
 * with the vertices of each edge and the length of the edge as paramaters
 */
template <typename T, typename Dist, typename EdgeCallback>
std::vector<T> mfc(
    ygm::comm                                            &w,
    std::vector<std::shared_ptr<ygm::container::bag<T>>> &components,
    Dist &&dist, EdgeCallback &&edge_callback) {
  static std::vector<std::shared_ptr<ygm::container::bag<T>>> *s_components;
  s_components = &components;

  std::vector<T>           reps;
  std::vector<GlobalIndex> reps_indices;

  reps.reserve(components.size());
  reps_indices.reserve(components.size());

  w.barrier();

  // Choose a single rep from each component
  {
    uint64_t seed;
    if (w.rank0()) seed = std::random_device{}();

    std::vector<uint64_t> bag_sizes;
    std::vector<uint64_t> bag_local_sizes;
    std::vector<uint64_t> bag_prefix_sum;

    bag_sizes.reserve(components.size());
    bag_local_sizes.reserve(components.size());
    bag_prefix_sum.reserve(components.size());

    for (auto &c : components) {
      bag_sizes.push_back(c->local_size());
      bag_local_sizes.push_back(c->local_size());
      bag_prefix_sum.push_back(c->local_size());
    }

    MPI_Request req[3];
    MPI_Status  stat[3];

    static std::vector<T>           *s_reps;
    static std::vector<GlobalIndex> *s_reps_indices;

    s_reps         = &reps;
    s_reps_indices = &reps_indices;

    reps.resize(components.size());
    reps_indices.resize(components.size());

    YGM_ASSERT_MPI(
        MPI_Ibcast(&seed, 1, MPI_UINT64_T, 0, w.get_mpi_comm(), req + 0));
    YGM_ASSERT_MPI(MPI_Iallreduce(MPI_IN_PLACE, bag_sizes.data(),
                                  components.size(), MPI_UINT64_T, MPI_SUM,
                                  w.get_mpi_comm(), req + 1));
    YGM_ASSERT_MPI(MPI_Iscan(MPI_IN_PLACE, bag_prefix_sum.data(),
                             components.size(), MPI_UINT64_T, MPI_SUM,
                             w.get_mpi_comm(), req + 2));
    YGM_ASSERT_MPI(MPI_Waitall(3, req, stat));

    std::default_random_engine random(seed);

    for (size_t i = 0; i < components.size(); i++) {
      size_t chosen_element = std::uniform_int_distribution<ssize_t>{
          0, static_cast<long>(bag_sizes[i] - 1)}(random);
      if (chosen_element < (bag_prefix_sum[i] - bag_local_sizes[i])) continue;
      chosen_element -= (bag_prefix_sum[i] - bag_local_sizes[i]);
      if (chosen_element >= bag_local_sizes[i]) continue;

      GlobalIndex index = {.rank = (uint16_t)w.rank(), .index = chosen_element};
      T element = *std::next(components[i]->local_begin(), chosen_element);

      reps[i]         = element;
      reps_indices[i] = index;

      w.async_bcast(
          [](GlobalIndex index, T element, size_t pos) {
            (*s_reps)[pos]         = element;
            (*s_reps_indices)[pos] = index;
          },
          index, element, i);
    }

    w.barrier();
  }

  w.barrier();

  if (components.size() == 1) return reps;

  std::vector<TreeEdge> new_edges;
  std::vector<TreeEdge> cur_best_tree;
  std::vector<TreeEdge> merged_tmp;

  auto ingest_new_edges = [&]() {
    std::sort(new_edges.begin(), new_edges.end(),
              [](const TreeEdge &a, const TreeEdge &b) {
                return a.weight < b.weight;
              });
    std::merge(cur_best_tree.begin(), cur_best_tree.end(), new_edges.begin(),
               new_edges.end(), std::back_insert_iterator(merged_tmp),
               [](const TreeEdge &a, const TreeEdge &b) {
                 return a.weight < b.weight;
               });
    cur_best_tree.clear();
    MST_presorted(components.size(), merged_tmp,
                  [&](TreeEdge e) { cur_best_tree.push_back(e); });
    merged_tmp.clear();
    new_edges.clear();
  };

  // Loop over every ordered pair of components
  for (size_t i = 0; i < components.size(); i++) {
    for (size_t j = 0; j < components.size(); j++) {
      if (i == j) continue;

      // If the current tree buffer is too big, run MST on it and keep only the
      // minimal edges
      if (new_edges.size() > (1073741824 / sizeof(TreeEdge) / 4)) {
        ingest_new_edges();
      }

      double min_dist  = std::numeric_limits<double>::max();
      size_t min_index = 0;

      size_t cur_index = 0;
      for (auto ite = components[j]->local_begin();
           ite != components[j]->local_end(); ite++) {
        auto d = dist(reps[i], *ite);
        if (d < min_dist) {
          min_dist  = d;
          min_index = cur_index;
        }
        cur_index++;
      }

      auto new_element = TreeEdge{
          .a       = i,
          .b       = j,
          .a_index = reps_indices[i],
          .b_index = GlobalIndex{.rank = (size_t)w.rank(), .index = min_index},
          .weight  = min_dist,
      };
      // Add the new edge to the potential tree
      new_edges.push_back(new_element);
    }
  }

  // One final local call to MST to make sure only the local minimal edges are
  // reduced
  ingest_new_edges();

  // Reduce by calling MST in the reduce function
  cur_best_tree = ygm::all_reduce(
      cur_best_tree,
      [&](std::vector<TreeEdge> a, std::vector<TreeEdge> b) {
        std::vector<TreeEdge> merged_edges;
        merged_edges.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(merged_edges),
                   [](const TreeEdge &a, const TreeEdge &b) {
                     return a.weight < b.weight;
                   });
        std::vector<TreeEdge> new_tree;
        MST_presorted(components.size(), merged_edges,
                      [&](TreeEdge e) { new_tree.push_back(e); });
        return new_tree;
      },
      w);

  if (w.rank0()) {
    std::vector<std::tuple<T, T, double>>         final_edges;
    static std::vector<std::tuple<T, T, double>> *s_final_edges;
    static size_t                                 s_points_recv;
    s_final_edges = &final_edges;
    s_points_recv = 0;
    final_edges.resize(cur_best_tree.size());

    for (size_t i = 0; i < cur_best_tree.size(); i++) {
      auto &ele = cur_best_tree[i];

      std::get<2>(final_edges[i]) = ele.weight;

      w.async(
          ele.a_index.rank,
          [](ygm::comm *w, size_t bag_index, size_t index, size_t edge_index) {
            w->async(
                0,
                [](T point, size_t edge_index) {
                  std::get<0>((*s_final_edges)[edge_index]) = point;
                  s_points_recv++;
                },
                *std::next((*s_components)[bag_index]->local_begin(), index),
                edge_index);
          },
          ele.a, ele.a_index.index, i);

      w.async(
          ele.b_index.rank,
          [](ygm::comm *w, size_t bag_index, size_t index, size_t edge_index) {
            w->async(
                0,
                [](T point, size_t edge_index) {
                  std::get<1>((*s_final_edges)[edge_index]) = point;
                  s_points_recv++;
                },
                *std::next((*s_components)[bag_index]->local_begin(), index),
                edge_index);
          },
          ele.b, ele.b_index.index, i);
    }

    w.local_wait_until(
        [&]() { return s_points_recv == cur_best_tree.size() * 2; });

    for (auto &e : final_edges) {
      edge_callback(std::get<0>(e), std::get<1>(e));
    }
  }

  w.barrier();

  return reps;
}
