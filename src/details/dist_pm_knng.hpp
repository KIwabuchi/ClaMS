// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

#include <metall/container/unordered_node_map.hpp>
#include <metall/container/vector.hpp>
#include <metall/utility/hash.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <metall/utility/mpi.hpp>
#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>

namespace clams {
/// \brief Holds local point store and k-NN graph that are allocated by
/// Metall.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::pm_feature_vector<float>,
          typename Distance = float, typename IdHash = std::hash<Id>>
class dist_pm_knng {
 private:
  using metall_manager_t = metall::utility::metall_mpi_adaptor::manager_type;

  template <typename T>
  using allocator_t = metall_manager_t::fallback_allocator<T>;

  template <typename T>
  using scp_allocator_t = metall_manager_t::scoped_fallback_allocator_type<T>;

  constexpr static unsigned int k_local_hash_seed             = 0xA1B2C3D4;
  constexpr static unsigned int k_point_partitioner_hash_seed = 0x1A2B3C4D;
  static_assert(k_local_hash_seed != k_point_partitioner_hash_seed,
                "k_local_hash_seed and k_point_partitioner_hash_seed must be "
                "different.");

  // Used for local data structures that use point ID as the key
  struct local_hasher {
    inline std::size_t operator()(const Id &id) const {
      return metall::utility::hash<void, k_local_hash_seed>{}(IdHash{}(id));
    }
  };

  // Used for partitioning points across processes
  struct partitioner_hasher {
    inline std::size_t operator()(const Id &id) const {
      return metall::utility::hash<void, k_point_partitioner_hash_seed>{}(
          IdHash{}(id));
    }
  };

 public:
  using id_type       = Id;
  using point_type    = Point;
  using distance_type = Distance;

  using point_store_type = metall::container::unordered_node_map<
      id_type, point_type, local_hasher, std::equal_to<id_type>,
      scp_allocator_t<std::pair<const id_type, point_type>>>;

  using neighbor_type = saltatlas::detail::neighbor<id_type, distance_type>;
  using neighbor_list_type =
      metall::container::vector<neighbor_type, allocator_t<neighbor_type>>;
  using knng_type = metall::container::unordered_node_map<
      id_type, neighbor_list_type, local_hasher, std::equal_to<id_type>,
      scp_allocator_t<std::pair<const id_type, neighbor_list_type>>>;

  /// \brief Copy a distributed PM k-NN graph from src_path to dst_path.
  /// The source datastore must be unopened.
  static bool copy(const std::filesystem::path &src_path,
                   const std::filesystem::path &dst_path, const MPI_Comm comm) {
    if (!metall::utility::metall_mpi_adaptor::consistent(src_path, comm)) {
      if (metall::utility::mpi::comm_rank(comm) == 0) {
        std::cerr << "Source datastore is not consistent across all "
                     "processes."
                  << std::endl;
      }
      return false;
    }

    const auto ret = metall::utility::metall_mpi_adaptor::copy(
        src_path, dst_path, comm, true);

    return ret;
  }

  /// \brief Get the owner rank of a point ID.
  static int get_owner(const id_type &id, const int comm_size) {
    return partitioner_hasher{}(id) % comm_size;
  }

  explicit dist_pm_knng(const MPI_Comm comm) : m_comm(comm) {}

  ~dist_pm_knng() noexcept = default;

  /// \brief Create a new distributed PM k-NN graph.
  void create(const std::filesystem::path &datastore_path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::create_only, datastore_path, m_comm, true);
    auto &localm = m_metall->get_local_manager();
    m_pstore     = localm.construct<point_store_type>(metall::unique_instance)(
        localm.get_allocator<>());
    m_knng = localm.construct<knng_type>(metall::unique_instance)(
        localm.get_allocator<>());
    metall::utility::mpi::barrier(m_comm);
  }

  /// \brief Open an existing distributed PM k-NN graph for read/write.
  void open(const std::filesystem::path &datastore_path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_only, datastore_path.string(), m_comm);
    auto &localm = m_metall->get_local_manager();
    m_pstore     = localm.find<point_store_type>(metall::unique_instance).first;
    m_knng       = localm.find<knng_type>(metall::unique_instance).first;
    if (!m_pstore || !m_knng) {
      if (metall::utility::mpi::comm_rank(m_comm) == 0) {
        std::cerr << "Failed to open distributed PM kNNG datastore at "
                  << datastore_path
                  << ". The datastore likely has a different format."
                  << std::endl;
      }
      std::abort();
    }
    metall::utility::mpi::barrier(m_comm);
  }

  /// \brief Open an existing distributed PM k-NN graph for read-only access.
  void open_read_only(const std::filesystem::path &datastore_path) {
    m_metall = std::make_unique<metall::utility::metall_mpi_adaptor>(
        metall::open_read_only, datastore_path.string(), m_comm);
    auto &localm = m_metall->get_local_manager();
    m_pstore     = localm.find<point_store_type>(metall::unique_instance).first;
    m_knng       = localm.find<knng_type>(metall::unique_instance).first;
    if (!m_pstore || !m_knng) {
      if (metall::utility::mpi::comm_rank(m_comm) == 0) {
        std::cerr << "Failed to open distributed PM kNNG datastore at "
                  << datastore_path
                  << ". The datastore likely has a different format."
                  << std::endl;
      }
      std::abort();
    }
    metall::utility::mpi::barrier(m_comm);
  }

  /// \brief Get a reference to the point store.
  point_store_type &get_point_store() { return *m_pstore; }

  /// \brief Get a const reference to the point store.
  const point_store_type &get_point_store() const { return *m_pstore; }

  /// \brief Get a reference to the k-NN graph.
  knng_type &get_knng() { return *m_knng; }

  /// \brief Get a const reference to the k-NN graph.
  const knng_type &get_knng() const { return *m_knng; }

 private:
  MPI_Comm                                             m_comm{MPI_COMM_NULL};
  std::unique_ptr<metall::utility::metall_mpi_adaptor> m_metall{nullptr};
  point_store_type                                    *m_pstore{nullptr};
  knng_type                                           *m_knng{nullptr};
};
}  // namespace clams
