#!/usr/bin/env python3
# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.
"""Extract clustering quality results from ClaMS benchmark logs.

Usage: extract_clams_clustering_qualities.py <log_file> [<log_file> ...]
"""

import csv
import re
import sys


PARAMETER_NAMES = ("kNNG-k", "Min-Cluster-Size")
PARAMETER = re.compile(
    r"^Parameter (" + "|".join(map(re.escape, PARAMETER_NAMES)) + r"):\s*(\S+)")
HEADER = (
    "kNNG k", "Min Cluster Size", "Found Clusters", "MST clustering",
    "Cluster coverage (%)", "ARI", "AMI", "file",
)


def extract(path):
    """Yield one row for each ARI/AMI pair in a log."""
    parameters = {name: "NA" for name in PARAMETER_NAMES}
    found_clusters = mst_clustering = coverage = "NA"
    ari = None

    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := PARAMETER.match(line):
                parameters[match.group(1)] = match.group(2)
            elif "Number of flat clusters:" in line:
                found_clusters = line.split()[-1]
            elif "MST-based cluster guess:" in line:
                mst_clustering = line.split()[-1]
            elif "Cluster coverage (%):" in line:
                coverage = line.split()[-1]
            elif line.startswith("ARI:"):
                ari = line.split()[-1]
            elif line.startswith("AMI:") and ari is not None:
                yield (
                    *(parameters[name] for name in PARAMETER_NAMES),
                    found_clusters, mst_clustering, coverage, ari,
                    line.split()[-1], path,
                )
                ari = None


def main(argv):
    if not argv:
        print(
            f"Usage: {sys.argv[0]} <log_file> [<log_file> ...]", file=sys.stderr)
        return 1

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(HEADER)
    for path in argv:
        try:
            writer.writerows(extract(path))
        except OSError as exc:
            print(f"{sys.argv[0]}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
