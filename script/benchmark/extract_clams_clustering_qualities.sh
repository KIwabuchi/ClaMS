#!/bin/bash
# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# Extracts clustering parameters and results (ARI/AMI) from ClaMS benchmark
# log files (e.g., out.log). Values are tracked as a running state while
# scanning the log top to bottom, so parameter lines that appear less often
# than ARI/AMI lines are repeated/shared across all following ARI/AMI rows
# until the next occurrence of that parameter updates the state.
#
# Usage: extract_clustering_results.sh <log_file> [<log_file> ...]

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <log_file> [<log_file> ...]" >&2
  exit 1
fi

printf "kNNG_k,min_cluster_size,final_clusters,singleton_to_noise,cluster_coverage,ARI,AMI,file\n"

for file in "$@"; do
  awk -v fname="$file" '
    # Return the last whitespace/colon separated token on the line, i.e. its value
    function lastfield(s,    n, arr) {
      n = split(s, arr, /[[:space:]:]+/)
      return arr[n]
    }

    BEGIN {
      k = "NA"; mcs = "NA"; fc = "NA"; singleton = "NA"; coverage = "NA"; have_ari = 0; ari = "NA"
    }

    /kNNG k:/ || /^[[:space:]]*k:[[:space:]]*[0-9]+[[:space:]]*$/ { k = lastfield($0); next }
    /Min cluster size/       { mcs = lastfield($0); next }
    /Evaluating Clustering Results, no noise points =/ { singleton = lastfield($0); next }
    /#of final clusters/     { fc = lastfield($0); next }
    /Cluster coverage \(%\)/ { coverage = lastfield($0); next }
    /ARI/                    { ari = lastfield($0); have_ari = 1; next }
    /AMI/ {
      if (have_ari) {
        ami = lastfield($0)
        printf "%s,%s,%s,%s,%s,%s,%s,%s\n", k, mcs, fc, singleton, coverage, ari, ami, fname
        have_ari = 0
      }
      next
    }
  ' "$file"
done
