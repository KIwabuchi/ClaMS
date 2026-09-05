#!/bin/bash
# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# Extracts clustering execution times from ClaMS benchmark log files
# (e.g., out.log). Each stage is delimited by a "<Stage>: <timestamp>" line
# and a matching "Finished <Stage>: <timestamp>" line; the elapsed time
# between the two is reported in seconds. Only the first occurrence of each
# stage pair is used (e.g., "Running CLAMS-HDBSCAN" and
# "Assign clusters to noise points" can repeat per min-cluster-size value).
#
# Usage: extract_clams_clustering_time.sh <log_file> [<log_file> ...]

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <log_file> [<log_file> ...]" >&2
  exit 1
fi

printf "kNNG k,nodes,tasks/node,kNNG (s),MFC (s),AMST (s),CLAMS-HDBSCAN (s),Noise clustering (s),file\n"

for file in "$@"; do
  awk -v fname="$file" '
    # Convert a "YYYY/MM/DD HH:MM:SS" timestamp found at the end of the line to epoch seconds
    function to_epoch(line,    ts, n, a) {
      if (!match(line, /[0-9]{4}\/[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$/)) return -1
      ts = substr(line, RSTART, RLENGTH)
      n = split(ts, a, /[\/ :]+/)
      return mktime(a[1] " " a[2] " " a[3] " " a[4] " " a[5] " " a[6])
    }

    # Return the last whitespace/colon separated token on the line, i.e. its value
    function lastfield(s,    n, arr) {
      n = split(s, arr, /[[:space:]:]+/)
      return arr[n]
    }

    function elapsed(start, finish) {
      if (start == -1 || finish == -1) return "NA"
      return finish - start
    }

    BEGIN {
      k = "NA"; nodes = "NA"; tasks = "NA"
      knng_start = -1; knng_end = -1
      mfc_start = -1; mfc_end = -1
      amst_start = -1; amst_end = -1
      hdbscan_start = -1; hdbscan_end = -1
      noise_start = -1; noise_end = -1
    }

    /^kNNG k:/ && k == "NA"           { k = lastfield($0) }
    /^Compute nodes:/ && nodes == "NA" { nodes = lastfield($0) }
    /^Tasks per node:/ && tasks == "NA" { tasks = lastfield($0) }

    /^Building KNNG:/ && knng_start == -1                          { knng_start = to_epoch($0) }
    /^Finished Building KNNG:/ && knng_start != -1 && knng_end == -1 { knng_end = to_epoch($0) }

    /^Connecting the CCs using MFC:/ && mfc_start == -1                          { mfc_start = to_epoch($0) }
    /^Finished Connecting the CCs using MFC:/ && mfc_start != -1 && mfc_end == -1 { mfc_end = to_epoch($0) }

    /^Running AMST, approx bound/ && amst_start == -1                          { amst_start = to_epoch($0) }
    /^Finished Running AMST, approx bound/ && amst_start != -1 && amst_end == -1 { amst_end = to_epoch($0) }

    /^Running CLAMS-HDBSCAN:/ && hdbscan_start == -1                          { hdbscan_start = to_epoch($0) }
    /^Finished Running CLAMS-HDBSCAN:/ && hdbscan_start != -1 && hdbscan_end == -1 { hdbscan_end = to_epoch($0) }

    /^Assign clusters to noise points:/ && noise_start == -1                          { noise_start = to_epoch($0) }
    /^Finished Assign clusters to noise points:/ && noise_start != -1 && noise_end == -1 { noise_end = to_epoch($0) }

    END {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", k, nodes, tasks, \
        elapsed(knng_start, knng_end), elapsed(mfc_start, mfc_end), \
        elapsed(amst_start, amst_end), elapsed(hdbscan_start, hdbscan_end), \
        elapsed(noise_start, noise_end), fname
    }
  ' "$file"
done
