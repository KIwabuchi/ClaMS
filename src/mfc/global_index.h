// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <cstdlib>

struct GlobalIndex {
  size_t rank;
  size_t index;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(rank, index);
  }
};