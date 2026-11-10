// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <metall/container/unordered_flat_map.hpp>
#include <metall/container/vector.hpp>
#include <metall/metall.hpp>

// CLAMS_USE_SALTATLAS is defined in cpp files
#ifdef CLAMS_USE_SALTATLAS
#include <saltatlas/dnnd/dnnd_adv.hpp>
#include "dist_pm_knng.hpp"
#endif

#include "edge_list.hpp"

namespace clams {

// Point/vertex ID type.
using id_t = uint32_t;
// Distance type.
using distance_t = float;
// Feature element type.
using fe_t = float;
// Cluster ID type.
using cluster_id_t = int32_t;

using weighted_edge_list_t = clams::weighted_edge_list<
    id_t, distance_t,
    metall::manager::scoped_fallback_allocator_type<std::byte>>;
using weighted_edge_t = weighted_edge_list_t::value_type;

// Type to store a list of point ID to cluster ID mappings.
using pm_point_to_cluster_id_list_t =
    metall::container::vector<std::pair<id_t, cluster_id_t>,
                              metall::manager::scoped_fallback_allocator_type<
                                  std::pair<id_t, cluster_id_t>>>;
using pm_point_to_cluster_id_map_t = metall::container::unordered_flat_map<
    id_t, cluster_id_t, std::hash<id_t>, std::equal_to<>,
    metall::manager::scoped_fallback_allocator_type<
        std::pair<const id_t, cluster_id_t>>>;

#ifdef CLAMS_USE_SALTATLAS
using dnnd_t =
    saltatlas::dnnd_adv<id_t, saltatlas::pm_feature_vector<fe_t>, distance_t>;

using dist_pm_knng_t =
    dist_pm_knng<id_t, saltatlas::pm_feature_vector<fe_t>, distance_t>;
using pm_knng_t  = typename dist_pm_knng_t::knng_type;
using neighbor_t = typename dist_pm_knng_t::neighbor_type;
using point_t    = typename dist_pm_knng_t::point_type;
#endif

}  // namespace clams
