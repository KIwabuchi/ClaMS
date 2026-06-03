// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY
#define METALL_DISABLE_OBJECT_CACHE

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <saltatlas/neo_dnnd/mpi.hpp>
#include <saltatlas/neo_dnnd/neo_dnnd.hpp>
#include <ygm/comm.hpp>

#include "build_knng.hpp"

using id_t       = clams::id_t;
using fe_t       = clams::fe_t;
using dist_t     = clams::distance_t;
using neo_dnnd_t = saltatlas::neo_dnnd<id_t, fe_t, dist_t>;

int main(int argc, char **argv) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    saltatlas::mpi::communicator comm;

    if (provided < MPI_THREAD_FUNNELED) {
      comm.cerr0()
          << "The threading support level is lesser than that demanded."
          << std::endl;
      comm.abort();
    }

    clams::option_t opt;
    bool            help{false};
    if (!parse_options(argc, argv, opt, help)) {
      comm.cerr0() << "Invalid option" << std::endl;
      clams::usage(argv[0], comm.cerr0());
      return 0;
    }
    if (help) {
      clams::usage(argv[0], comm.cout0());
      return 0;
    }
    show_options(opt, comm.cout0());

    // Set the number of threads per rank
    if (opt.num_threads > 0) {
      comm.cout0() << "Setting #of OpenMP threads to " << opt.num_threads
                   << std::endl;
      saltatlas::utility::omp::set_num_threads(opt.num_threads);
    } else if (std::getenv("OMP_NUM_THREADS")) {
      comm.cout0() << "Using OMP_NUM_THREADS=" << std::getenv("OMP_NUM_THREADS")
                   << std::endl;
    } else {
      comm.cout0() << "Setting #of OpenMP threads to " << 1 << std::endl;
      saltatlas::utility::omp::set_num_threads(1);
    }
    comm.barrier();

    const auto point_files =
        saltatlas::utility::find_file_paths(opt.point_file_names);
    if (point_files.empty()) {
      comm.cerr0() << "No point files found" << std::endl;
      return 1;
    }

    ygm::utility::timer neo_dnnd_const_timer;
    // Build kNNG using neo_dnnd
    neo_dnnd_t neo_dnnd(
        saltatlas::distance::distance_function<typename neo_dnnd_t::point_type,
                                               dist_t>(opt.distance_name),
        comm, opt.verbose);
    comm.cout0() << "\n<<Read Points>>" << std::endl;
    neo_dnnd.load_points(point_files.begin(), point_files.end(),
                         opt.point_file_format);

    comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
    auto knng = neo_dnnd.build(opt.index_k, opt.r, opt.delta,
                               opt.replicate_rate, opt.batch_size);
    comm.barrier();
    comm.cout0() << "\nNEO-DNND kNNG construction took (s)\t"
                 << neo_dnnd_const_timer.elapsed() << std::endl;

    ygm::utility::timer pm_data_const_timer;
    {
      comm.cout0() << "\n<<Construct distributed PM kNNG Datastore>>"
                   << std::endl;

      clams::dist_pm_knng_t pm_knng(comm.comm());
      pm_knng.create(opt.scratchpath);
      ygm::comm ygm_comm(comm.comm());

      static auto &ref_knng = pm_knng.get_knng();
      for (const auto &pair : knng) {
        const auto src = pair.first;
        ygm_comm.async(
            clams::dist_pm_knng_t::get_owner(src, ygm_comm.size()),
            [](const id_t sid, const auto &neighbors) {
              auto &dst = ref_knng[sid];
              dst.reserve(neighbors.size());
              for (const auto &neighbor : neighbors) {
                dst.emplace_back(neighbor);
              }
            },
            src, pair.second);
      }
      knng.clear();
      ygm_comm.barrier();

      static auto &ref_pstore = pm_knng.get_point_store();
      for (const auto &point : neo_dnnd.local_points()) {
        const auto       &id = point.first;
        const auto       &fv = point.second;
        std::vector<fe_t> fv_vec(fv.begin(), fv.end());
        ygm_comm.async(
            clams::dist_pm_knng_t::get_owner(id, comm.size()),
            [](const auto &pid, const auto &fv) {
              const auto [it, inserted] =
                  ref_pstore.try_emplace(pid, fv.begin(), fv.end());
              if (!inserted) {
                std::cerr << "Warning: Duplicate point with id " << pid
                          << " found. Ignoring the duplicate." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            },
            id, fv_vec);
      }
      ygm_comm.barrier();
    }
    comm.barrier();
    comm.cout0() << "\nConstructing distributed PM kNNG datastore took (s)\t"
                 << pm_data_const_timer.elapsed() << std::endl;

    if (opt.scratchpath != opt.datastorepath && !opt.datastorepath.empty()) {
      comm.cout0() << "Copying PM datastore to " << opt.datastorepath
                   << std::endl;
      metall::logger::set_log_level(metall::logger::level_filter::critical);
      ygm::utility::timer copy_timer;
      const auto          ret = clams::dist_pm_knng_t::copy(
          opt.scratchpath, opt.datastorepath, comm.comm());
      if (!ret) {
        comm.cerr0() << "Failed to copy PM datastore" << std::endl;
        comm.abort();
      }
      comm.cout0() << "Copy took (s): " << copy_timer.elapsed() << std::endl;
    }
    comm.barrier();
  }
  ::MPI_Finalize();

  return 0;
}
