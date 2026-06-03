// ClaMS clustering pipeline
// Copyright (C) 2023–2025 Lawrence Livermore National Security (LLNS), LLC

#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "build_knng.hpp"

using id_t   = clams::id_t;
using fe_t   = clams::fe_t;
using dist_t = clams::distance_t;
using dnnd_t =
    saltatlas::dnnd_adv<id_t, saltatlas::pm_feature_vector<fe_t>, dist_t>;
using dist_pm_knng_t = clams::dist_pm_knng_t;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);
  clams::show_config<id_t, fe_t, dist_t>(comm);
  {
    clams::option_t opt;
    bool            help{false};
    if (!clams::parse_options(argc, argv, opt, help)) {
      comm.cerr0() << "Invalid option" << std::endl;
      clams::usage(argv[0], comm.cerr0());
      return 0;
    }
    if (help) {
      clams::usage(argv[0], comm.cout0());
      return 0;
    }
    clams::show_options(opt, comm.cout0());

    comm.cout0() << "Constructing PM datastore" << std::endl;
    {
      dnnd_t g(comm, std::random_device{}(), opt.verbose);
      {
        comm.cout0() << "\n<<Read Points>>" << std::endl;
        const auto paths =
            saltatlas::utility::find_file_paths(opt.point_file_names);
        if (paths.empty()) {
          comm.cerr0() << "No point files found" << std::endl;
          return 1;
        }
        ygm::utility::timer point_read_timer;
        g.load_points(paths.begin(), paths.end(), opt.point_file_format);
        comm.cout0() << "\nReading points took (s)\t"
                     << point_read_timer.elapsed() << std::endl;
        comm.cout0() << "#of points\t" << g.num_points() << std::endl;
      }

      size_t index_id{};
      {
        comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
        ygm::utility::timer const_timer;
        index_id = g.build(
            saltatlas::distance::convert_to_distance_id(opt.distance_name),
            opt.index_k, opt.r, opt.delta);
        comm.cout0() << "\nkNNG construction took (s)\t"
                     << const_timer.elapsed() << std::endl;
      }

      // Copy the DNND data to the PM datastore
      {
        dist_pm_knng_t pm_knng(comm.get_mpi_comm());
        pm_knng.create(opt.scratchpath);

        auto&        pstore     = pm_knng.get_point_store();
        static auto& ref_pstore = pstore;
        for (const auto& point : g.local_points()) {
          const auto& id = point.first;
          const auto& fv = point.second;
          comm.async(
              dist_pm_knng_t::get_owner(id, comm.size()),
              [](const auto& pid, const auto& pfv) { ref_pstore[pid] = pfv; },
              id, fv);
        }

        auto& knng =
            const_cast<typename dnnd_t::knn_index_type&>(g.get_index(index_id));
        auto&        pm_index = pm_knng.get_knng();
        static auto& ref_knng = pm_index;

        for (auto pitr = knng.points_begin(), pend = knng.points_end();
             pitr != pend; ++pitr) {
          const auto& source = pitr->first;
          for (auto nitr = knng.neighbors_begin(source),
                    nend = knng.neighbors_end(source);
               nitr != nend; ++nitr) {
            const auto& neighbor = *nitr;
            comm.async(
                dist_pm_knng_t::get_owner(source, comm.size()),
                [](auto src, auto ngbr) { ref_knng[src].emplace_back(ngbr); },
                source, neighbor);
          }
        }
        comm.barrier();
      }
    }
    comm.cf_barrier();

    if (opt.scratchpath != opt.datastorepath && !opt.datastorepath.empty()) {
      comm.cout0() << "Copying DNND PM datastore to " << opt.datastorepath
                   << std::endl;
      metall::logger::set_log_level(metall::logger::level_filter::critical);
      ygm::utility::timer copy_timer;
      const auto ret = dist_pm_knng_t::copy(opt.scratchpath, opt.datastorepath,
                                            comm.get_mpi_comm());
      if (!ret) {
        comm.cerr0() << "Failed to copy DNND PM datastore" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      comm.cout0() << "Copy took (s): " << copy_timer.elapsed() << std::endl;
    }
  }

  return 0;
}
