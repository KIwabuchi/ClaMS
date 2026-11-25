// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#define APPROX_MST_BUILDER_VERBOSE 1
#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/utility/timer.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <metall/metall.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <metall/utility/metall_mpi_datastore.hpp>

#include "../common.hpp"
#include "approx_mst_builder.hpp"

namespace cls = clams;

using id_t       = cls::id_t;
using distance_t = cls::distance_t;

using edge_t = std::tuple<id_t, id_t, distance_t>;

void usage(ygm::comm &comm, const char *program) {
  if (comm.rank0()) {
    std::cout
        << "Usage: " << program
        << " (-i <input_file> [input_file ...] | -d <pm_input_path>)"
           " [-b <bridge_file>] [-p <pm_output_path>] [-e <float>]"
        << std::endl
        << "  -i <input_file>  Text file(s) containing an approximate "
           "nearest-neighbor graph"
        << std::endl
        << "  -d <pm_input_path>  PM datastore path for a distributed kNNG"
        << std::endl
        << "  -b <bridge_file>  Optional text file of bridge edges "
           "(source destination distance)"
        << std::endl
        << "  -p <pm_output_path>  PM datastore path for MST edges" << std::endl
        << "  -e <float>  Approximation bound (default: 1.1)" << std::endl
        << "  -h  Show this help message" << std::endl;
  }
}

void parse_cmd_line(int argc, char **argv, ygm::comm &comm, float &approx_bound,
                    std::vector<std::filesystem::path> &txt_input_filenames,
                    std::filesystem::path              &pm_input_path,
                    std::filesystem::path              &bridge_file_path,
                    std::filesystem::path              &pm_output_path) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  int  c;
  bool inserting_input_filenames = false;
  bool prn_help                  = false;
  while (true) {
    while ((c = getopt(argc, argv, "+e:i:d:b:p:h")) != -1) {
      inserting_input_filenames = false;
      switch (c) {
        case 'h':
          usage(comm, argv[0]);
          exit(EXIT_SUCCESS);
        case 'e':
          approx_bound = atof(optarg);
          break;
        case 'i':
          txt_input_filenames.push_back(optarg);
          inserting_input_filenames = true;
          break;
        case 'd':
          pm_input_path = optarg;
          break;
        case 'b':
          bridge_file_path = optarg;
          break;
        case 'p':
          pm_output_path = optarg;
          break;
        default:
          std::cerr << "Unrecognized option: " << c << ", ignore." << std::endl;
          prn_help = true;
          break;
      }
    }
    if (optind >= argc) break;

    if (inserting_input_filenames) {
      txt_input_filenames.push_back(argv[optind]);
    }

    ++optind;
  }

  // Detect misconfigured options
  if (pm_input_path.empty() && txt_input_filenames.size() < 1) {
    comm.cout0(
        "Must specify input file(s) containing approximate nearest-neighbor "
        "graph");
    prn_help = true;
  }

  if (!txt_input_filenames.empty() && !pm_input_path.empty()) {
    comm.cout0(
        "Cannot specify both input files and PM datastore path, choose one.");
    prn_help = true;
  }

  if (prn_help) {
    usage(comm, argv[0]);
    exit(EXIT_FAILURE);
  }
}

std::vector<edge_t> read_dnnd_output(
    const std::vector<std::filesystem::path> &filenames, ygm::comm &c) {
  std::vector<edge_t> to_return;

  ygm::container::bag<std::string> filenames_bag(c);

  if (c.rank0()) {
    for (const auto &filename : filenames) {
      filenames_bag.async_insert(filename.string());
    }
  }
  c.barrier();

  filenames_bag.for_all([&to_return](const auto &filename) {
    std::ifstream ifs(filename);

    std::string neighbors_str;
    std::string distances_str;

    while (std::getline(ifs, neighbors_str)) {
      std::getline(ifs, distances_str);
      assert(distances_str.size() > 0);

      std::stringstream neighbors_ss(neighbors_str);
      std::stringstream distances_ss(distances_str);

      id_t       src;
      id_t       ngbr;
      distance_t dist;

      neighbors_ss >> src;
      distances_ss >> dist;  // Dummy value, not used

      while (neighbors_ss >> ngbr) {
        distances_ss >> dist;

        to_return.push_back(std::make_tuple(src, ngbr, dist));
      }
    }
  });

  c.barrier();

  return to_return;
}

void read_edge_file(const std::filesystem::path &bridge_file_path,
                    std::vector<edge_t>         &edges) {
  std::ifstream ifs(bridge_file_path);
  id_t          src;
  id_t          dst;
  distance_t    dist;
  while (ifs >> src >> dst >> dist) {
    edges.push_back(std::make_tuple(src, dst, dist));
  }
}

/// Read a distributed PM kNNG from the given datastore path and return the
/// edges as a vector of edge_t. This function can read a distributed PM kNNG
/// using a different number of mpi ranks than the one used to create it.
std::vector<edge_t> read_pm_knng_output(std::filesystem::path &dstore_path,
                                        ygm::comm             &comm) {
  comm.cout0() << "Reading distributed PM kNNG data from: " << dstore_path
               << std::endl;

  std::vector<edge_t> to_return;
  // Lambda function to read kNNG from a local metall manager and append edges
  // to to_return
  auto knng_reader = [&](const cls::dist_pm_knng_t::knng_type &knng) {
    std::size_t local_num_neighbors = 0;
    for (const auto &[source, neighbors] : knng) {
      (void)source;
      local_num_neighbors += neighbors.size();
    }
    to_return.reserve(to_return.size() + local_num_neighbors);

    for (const auto &[source, neighbors] : knng) {
      for (const auto &neighbor : neighbors) {
        if (neighbor.distance < 0.0) {
          comm.cerr0("Negative distance found in distributed kNNG: ", neighbor);
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        to_return.push_back(
            std::make_tuple(source, neighbor.id, neighbor.distance));
      }
    }
  };

  // Iterate over local datastore paths and read kNNG from each local manager
  const auto local_paths = cls::dist_pm_knng_t::get_local_datastore_paths(
      dstore_path, comm.get_mpi_comm());
  for (std::size_t r = 0; r < local_paths.size(); ++r) {
    if (r % comm.size() == comm.rank()) {
      if (!metall::manager::consistent(local_paths[r])) {
        comm.cerr() << "Inconsistent local managers at rank " << comm.rank()
                    << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
      metall::manager local_manager(metall::open_read_only, local_paths[r]);
      if (!local_manager.check_sanity()) {
        comm.cerr() << "Failed to open local manager at rank " << comm.rank()
                    << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
      auto *knng = cls::dist_pm_knng_t::find_knng(local_manager);
      if (!knng) {
        comm.cerr() << "Failed to find kNNG" << std::endl;
        MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
      }
      knng_reader(*knng);
    }
  }
  comm.cf_barrier();

  return to_return;
}

std::vector<edge_t> approx_mst(const std::vector<edge_t> &edges,
                               const float approx_bound, ygm::comm &c) {
  amst::approx_mst_builder<id_t, distance_t> amst_builder(c, approx_bound);

  c.barrier();
  ygm::utility::timer amst_timer;

  while (not amst_builder.is_done()) {
    std::for_each(edges.begin(), edges.end(),
                  [&amst_builder](const auto &edge) {
                    const auto &[src, dest, wgt] = edge;
                    amst_builder.try_add_edge(src, dest, wgt);
                  });
  }

  c.barrier();
  double amst_time = amst_timer.elapsed();

  c.cout0() << "Approx bound: " << approx_bound
            << "\nTrees: " << amst_builder.num_trees()
            << "\nEdges: " << sum(amst_builder.get_result().size(), c)
            << "\nTime: " << amst_time << std::endl;

  return amst_builder.get_result();
}

// Adapted from Reddit comment sorting example
void pivot_sort(std::vector<edge_t> &in_vec, ygm::comm &world) {
  // Use edge weight for sorting
  auto edge_comp_lambda = [](const edge_t &a, const edge_t &b) {
    return std::get<2>(a) < std::get<2>(b);
  };
  auto edge_index_pair_comp_lambda = [](const std::pair<edge_t, size_t> &a,
                                        const std::pair<edge_t, size_t> &b) {
    if (std::get<2>(a.first) != std::get<2>(b.first)) {
      return std::get<2>(a.first) < std::get<2>(b.first);
    }
    return a.second < b.second;
  };

  const size_t        samples_per_pivot = 40;
  std::vector<edge_t> to_sort;
  to_sort.reserve(in_vec.size() * 1.1f);

  //
  //  Choose pivots, uses index as 3rd sorting argument to solve issue with lots
  //  of duplicate items
  std::vector<std::pair<edge_t, size_t>> samples;
  std::vector<std::pair<edge_t, size_t>> pivots;
  static auto                           &s_samples = samples;
  static auto                           &s_to_sort = to_sort;
  samples.reserve(world.size() * samples_per_pivot);

  //
  std::default_random_engine rng;

  size_t my_prefix   = ygm::prefix_sum(in_vec.size(), world);
  size_t global_size = ygm::sum(in_vec.size(), world);
  std::uniform_int_distribution<size_t> uintdist{0, global_size - 1};

  for (size_t i = 0; i < samples_per_pivot * world.size(); ++i) {
    size_t index = uintdist(rng);
    if (index >= my_prefix && index < my_prefix + in_vec.size()) {
      world.async_bcast(
          [](const std::pair<edge_t, size_t> &sample) {
            s_samples.push_back(sample);
          },
          std::make_pair(in_vec[index - my_prefix], index));
    }
  }
  world.barrier();

  YGM_ASSERT_RELEASE(samples.size() == samples_per_pivot * world.size());
  std::sort(samples.begin(), samples.end(), edge_index_pair_comp_lambda);
  for (size_t i = samples_per_pivot - 1, pivot_id = 0;
       pivot_id < world.size() - 1; i += samples_per_pivot, ++pivot_id) {
    pivots.push_back(samples[i]);
  }
  samples.clear();
  samples.shrink_to_fit();

  YGM_ASSERT_RELEASE(pivots.size() == world.size() - 1);

  //
  // Partition using pivots
  for (size_t i = 0; i < in_vec.size(); ++i) {
    auto   itr   = std::lower_bound(pivots.begin(), pivots.end(),
                                    std::make_pair(in_vec[i], my_prefix + i),
                                    edge_index_pair_comp_lambda);
    size_t owner = std::distance(pivots.begin(), itr);

    world.async(
        owner, [](const edge_t &val) { s_to_sort.push_back(val); }, in_vec[i]);
  }
  world.barrier();

  world.cout0("Min partition size: ", ygm::min(s_to_sort.size(), world));
  world.cout0("Max partition size: ", ygm::max(s_to_sort.size(), world));

  if (not to_sort.empty()) {
    std::sort(to_sort.begin(), to_sort.end(), edge_comp_lambda);
  }

  // world.cout0(ygm::max(s_to_sort.size(), world));
  // world.cout0(ygm::min(s_to_sort.size(), world));

  // world.cout(to_sort.size());

  // world.cout("to_sort.size() = ", to_sort.size());

  // //
  // // verify
  // world.barrier();
  // YGM_ASSERT_RELEASE(ygm::sum(in_vec.size(), world) ==
  //                ygm::sum(to_sort.size(), world));
  // if (world.rank() < world.size() - 1 && not to_sort.empty()) {
  //   world.async(
  //       world.rank() + 1,
  //       [](const T &val) {
  //         if (not s_to_sort.empty()) {
  //           YGM_ASSERT_RELEASE(val <= s_to_sort[0]);
  //         }
  //       },
  //       to_sort.back());
  // }

  world.barrier();
  in_vec.swap(to_sort);
}

void write_output_single_file(const std::filesystem::path &output_filename,
                              const std::vector<edge_t> &edges, ygm::comm &c) {
  MPI_Comm mpi_comm = c.get_mpi_comm();

  if (c.rank0()) {
    std::ofstream ofs(output_filename);

    // Write own edges
    for (auto &e : edges) {
      ofs << std::get<0>(e) << "\t" << std::get<1>(e) << "\t" << std::get<2>(e)
          << "\n";
    }

    // Write edges for all other ranks
    for (int i = 1; i < c.size(); ++i) {
      std::vector<edge_t> remote_edges =
          c.mpi_recv<std::vector<edge_t>>(i, 0, mpi_comm);

      for (auto &e : remote_edges) {
        ofs << std::get<0>(e) << "\t" << std::get<1>(e) << "\t"
            << std::get<2>(e) << "\n";
      }
    }
  } else {
    c.mpi_send(edges, 0, 0, mpi_comm);
  }
  c.cout0("Dumped AMST edges to: ", output_filename);
}

void write_output_pm(const std::filesystem::path &output_path,
                     const std::vector<edge_t> &edges, ygm::comm &c) {
  metall::utility::metall_mpi_adaptor manager(metall::create_only, output_path,
                                              c.get_mpi_comm(), true);
  auto &local_manager = manager.get_local_manager();
  auto *amst_edges    = local_manager.construct<cls::weighted_edge_list_t>(
      metall::unique_instance)(local_manager.get_allocator());
  amst_edges->reserve(edges.size());

  // Write own edges
  for (const auto &e : edges) {
    amst_edges->emplace_back(std::get<0>(e), std::get<1>(e), std::get<2>(e));
  }

  c.cf_barrier();
  c.cout0("Created distributed Metall datastore to: ", output_path);
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  std::vector<std::filesystem::path> txt_input_filenames;
  std::filesystem::path              pm_input_path;
  std::filesystem::path              bridge_file_path;
  std::filesystem::path              pm_output_path;
  float                              amst_approx_bound{1.1};

  parse_cmd_line(argc, argv, world, amst_approx_bound, txt_input_filenames,
                 pm_input_path, bridge_file_path, pm_output_path);

  world.cout0("Reading kNNG");
  std::vector<edge_t> edges;
  {
    ygm::utility::timer knng_read_timer;
    edges = (txt_input_filenames.size() > 0)
                ? read_dnnd_output(txt_input_filenames, world)
                : read_pm_knng_output(pm_input_path, world);
    world.cout0("kNNG reading time (s): ", knng_read_timer.elapsed());
  }

  if (!bridge_file_path.empty() && world.rank0()) {
    ygm::utility::timer edge_read_timer;
    read_edge_file(bridge_file_path, edges);
    world.cout0("Edge bridge reading time (s): ", edge_read_timer.elapsed());
  }

  world.cout0() << "Total number of edges read: "
                << ygm::sum(edges.size(), world) << std::endl;
  world.cf_barrier();

  auto amst_edges = approx_mst(edges, amst_approx_bound, world);
  world.cout0() << "AMST edges: " << ygm::sum(amst_edges.size(), world)
                << std::endl;

  world.cout0("Sorting AMST edges");
  {
    ygm::utility::timer sort_timer;
    pivot_sort(amst_edges, world);
    world.cout0("Sorting time (s): ", sort_timer.elapsed());
  }
  world.cf_barrier();

  world.cout0("Writing output");
  {
    ygm::utility::timer output_timer;
    if (!pm_output_path.empty()) {
      write_output_pm(pm_output_path, amst_edges, world);
    }
    world.cout0("Output writing time (s): ", output_timer.elapsed());
  }
  world.cf_barrier();

  return EXIT_SUCCESS;
}
