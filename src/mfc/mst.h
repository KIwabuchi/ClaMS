// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <algorithm>
#include <vector>

template <typename T>
concept EdgeType = requires(T t) {
  { t.weight } -> std::convertible_to<float>;
  { t.a } -> std::convertible_to<size_t>;
  { t.b } -> std::convertible_to<size_t>;
};

template <EdgeType T, typename E>
void MST_presorted(size_t num_nodes, std::vector<T> edges, E&& edge_callback) {
  struct Node {
    size_t set_id;
    size_t next_node;
  };

  std::vector<Node> sets;
  sets.reserve(num_nodes);
  for (size_t i = 0; i < num_nodes; i++)
    sets.push_back({
        i,
        i,
    });

  size_t num_edges = 0;

  for (auto& e : edges) {
    if (sets[e.a].set_id == sets[e.b].set_id) {
      continue;
    }

    edge_callback(e);
    num_edges++;
    if (num_edges == num_nodes - 1) return;

    size_t new_id = sets[e.a].set_id;

    size_t cur_id = e.b;

    while (sets[cur_id].next_node != e.b) {
      sets[cur_id].set_id = new_id;
      cur_id              = sets[cur_id].next_node;
    }
    sets[cur_id].set_id = new_id;

    sets[cur_id].next_node = sets[e.a].next_node;
    sets[e.a].next_node    = e.b;
  }
}

template <EdgeType T, typename E>
void MST(size_t num_nodes, std::vector<T> edges, E&& edge_callback) {
  std::sort(edges.begin(), edges.end(),
            [](T& e1, T& e2) { return e1.weight < e2.weight; });
  MST_presorted(num_nodes, edges, edge_callback);
}