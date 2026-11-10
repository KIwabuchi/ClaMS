#!/usr/bin/env python3
# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.
"""Extract stage timings for each parameter combination in benchmark logs.

Usage: extract_clams_clustering_time.py <log_file> [<log_file> ...]
"""

import csv
from datetime import datetime
import re
import sys


# Metadata fields to extract from the log files
COMMON_METADATA = (
    ("Compute nodes:", "nodes"),
    ("Tasks per node:", "tasks"),
)

# Stage names and parameter names to extract from the logs
STAGES = (
    ("Build-kNNG", "kNNG-k"),
    ("MFC", None),
    ("AMST", "AMST-ApproxBound"),
    ("ClaMS-HDBSCAN", "Min-Cluster-Size"),
    ("Noise-Points-Clustering", None),
)

PARAMETER_NAMES = tuple(parameter for _, parameter in STAGES if parameter)
HEADER = (
    "kNNG-k", "AMST Approx Bound", "Min Cluster Size", "nodes", "tasks/node", "kNNG (s)", "MFC (s)",
    "AMST (s)", "ClaMS-HDBSCAN (s)", "Noise points clustering (s)", "file",
)

# Regular expression to match timestamps at the end of lines
TIMESTAMP = re.compile(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})$")
# Regular expression to match parameter lines in the log files
PARAMETER = re.compile(
    r"^Parameter (" + "|".join(map(re.escape, PARAMETER_NAMES)) + r"):\s*(\S+)")


# Extract the timestamp at the end of a line, if present
def timestamp_at_end(line):
    match = TIMESTAMP.search(line.rstrip("\n"))
    if match is None:
        return None
    try:
        # The benchmark uses local time, as did awk's mktime in the shell script.
        return int(datetime.strptime(match.group(1), "%Y/%m/%d %H:%M:%S").timestamp())
    except ValueError:
        return None


def stage_event(line):
    for index, (stage, _) in enumerate(STAGES):
        if line.startswith("Start stage " + stage):
            return index, "start"
        if line.startswith("Finished stage " + stage):
            return index, "finish"
    return None


def extract(path):
    """Return one row per leaf experiment, sharing earlier stage values."""
    common_metadata = {key: "NA" for _, key in COMMON_METADATA}
    parameters = {name: "NA" for name in PARAMETER_NAMES}
    starts = [None] * len(STAGES)
    finishes = [None] * len(STAGES)
    rows = []
    deepest_stage = -1

    def row():
        elapsed = [
            "NA" if start is None or finish is None else finish - start
            for start, finish in zip(starts, finishes)
        ]
        return [*(parameters[name] for name in PARAMETER_NAMES),
                common_metadata["nodes"], common_metadata["tasks"],
                *elapsed, path]

    with open(path, encoding="utf-8", errors="replace") as log:
        for line in log:
            for prefix, key in COMMON_METADATA:
                if line.startswith(prefix) and common_metadata[key] == "NA":
                    common_metadata[key] = re.split(
                        r"[\s:]+", line.strip())[-1]

            if match := PARAMETER.match(line):
                name, value = match.groups()
                parameters[name] = value

            event = stage_event(line)
            if event is not None:
                index, kind = event
                if kind == "start":
                    # A repeated stage begins the next survey combination.
                    if index <= deepest_stage:
                        rows.append(row())
                    for later in range(index, len(STAGES)):
                        starts[later] = finishes[later] = None
                        parameter = STAGES[later][1]
                        if parameter:
                            parameters[parameter] = "NA"
                    starts[index] = timestamp_at_end(line)
                    deepest_stage = index
                elif starts[index] is not None and finishes[index] is None:
                    finishes[index] = timestamp_at_end(line)

    rows.append(row())
    return rows


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
