#!/usr/bin/env python3
"""
Evaluate the correlation between KNNG neighbors and clusters.

Take a KNNG dump and a cluster label file, then count how many of the first k
neighbors of each point belong to the same cluster as the point itself.

Usage:
    python evaluate_correlation_knng_and_clusters.py -g knng_dir -c clusters.txt -k 10

Arguments:
    -g/--graph: KNNG file or directory containing KNNG dump files.
    -N/--no-distance: indicate that the KNNG files do not contain distance lines.
    -c/--cluster: cluster file or directory containing cluster-label files.
    -k/--neighbors: number of neighbors to consider for each point.
    -h/--help: show this help message and exit.

KNNG file format:
    Each record is encoded as two lines:
        <source_id> <neighbor_id_1> <neighbor_id_2> ...
        <dummy_distance> <distance_1> <distance_2> ...
    The leading distance is a dummy value corresponding to the source ID and will be ignored.
    If the KNNG files do not contain distance lines, use the -N/--no-distance flag.

Cluster file format:
    Each line contains a point and its cluster ID separated by whitespace.
    The first line can be a comment starting with '#'.
    A file containing only cluster IDs is also accepted; in that case point IDs are
    assumed to be 0, 1, 2, ... in order.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def resolve_files(paths: Sequence[str]) -> list[Path]:
    """Resolve file, directory, and glob inputs to a unique sorted file list."""
    resolved: set[Path] = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            matches = set(path.glob("*.txt")) | set(path.glob("knng-*.txt"))
        elif path.is_file():
            matches = {path}
        else:
            matches = {Path(match) for match in glob.glob(value)}

        resolved.update(match.resolve() for match in matches if match.is_file())

    if not resolved:
        raise FileNotFoundError(f"No files matched: {paths}")

    return sorted(resolved)


def iter_knng_records(path: Path, contains_distance: bool) -> Iterator[tuple[int, list[int]]]:
    """Yield (source_id, neighbor_ids) pairs from one KNNG dump file."""
    with path.open("r", encoding="utf-8") as knng_file:
        while True:
            ids_line = knng_file.readline()
            if not ids_line:
                return

            stripped = ids_line.strip()
            if not stripped:
                # This is a critical error.
                raise ValueError(f"{path}: failed to parse row: {ids_line!r}")

            if contains_distance:
                # Skip the distance line if it exists, as it will be processed later.
                knng_file.readline()

            src_id = int(stripped.split()[0])
            neighbor_ids = [int(value) for value in stripped.split()[1:]]

            yield src_id, neighbor_ids

def read_cluster_labels(cluster_path: str | Path) -> dict[int, int]:
    """Read cluster labels from a file or directory into a point->cluster map."""
    labels: dict[int, int] = {}
    files = resolve_files([str(cluster_path)])

    if not files:
        raise FileNotFoundError(f"No cluster files found in {cluster_path}")

    # Heuristic: determine whether the file format contains point IDs by scanning
    # the first non-comment, non-empty line.
    first_file = files[0]
    contains_ids = False
    with first_file.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if len(stripped.split()) >= 2:
                contains_ids = True
            break

    for file in files:
        with file.open("r") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                items = stripped.split()
                if contains_ids:
                    if len(items) < 2:
                        raise ValueError(f"{file}: invalid cluster row: {line.rstrip()}")
                    point_id, cluster_id = int(items[0]), int(items[1])
                else:
                    point_id = len(labels)
                    cluster_id = int(items[0])

                if point_id in labels:
                    raise ValueError(f"Duplicate point ID {point_id} in cluster file {file}")
                labels[point_id] = cluster_id

    if not labels:
        raise ValueError(f"No cluster labels were loaded from {cluster_path}")

    return labels


def evaluate_knng_clusters(graph_path: str | Path, cluster_path: str | Path, k: int = 10, no_distance: bool = False):
    """Return per-point and aggregate same-cluster neighbor statistics."""
    if k <= 0:
        raise ValueError("-k/--neighbors must be positive")

    labels = read_cluster_labels(cluster_path)
    print(f"Loaded {len(labels)} cluster labels from {cluster_path}")

    files = resolve_files([str(graph_path)])

    per_point: list[tuple[int, int, int, int, float]] = []
    total_same = 0
    total_neighbors = 0
    covered_points = 0
    noise_points = 0

    for knng_file in files:
        print(f"Evaluating KNNG file {knng_file} with k={k}")
        for source_id, neighbor_ids in iter_knng_records(knng_file, not no_distance):
            if source_id not in labels:
                continue

            source_label = labels[source_id]
            if source_label == -1:
                # Skip points that are not assigned to any cluster.
                noise_points += 1
                continue

            limited_neighbors = neighbor_ids[:k]
            if not limited_neighbors:
                continue

            same_count = 0
            for neighbor_id in limited_neighbors:
                if labels.get(neighbor_id) == source_label:
                    same_count += 1

            ratio = same_count / len(limited_neighbors)
            per_point.append((source_id, source_label, same_count, len(limited_neighbors), ratio))
            total_same += same_count
            total_neighbors += len(limited_neighbors)
            covered_points += 1

    avg_ratio = (total_same / total_neighbors) if total_neighbors > 0 else 0.0
    return per_point, {
        "num_points": covered_points,
        "num_noise_points": noise_points,
        "num_same_cluster_neighbors": total_same,
        "num_considered_neighbors": total_neighbors,
        "avg_same_cluster_fraction": avg_ratio,
        "num_clustered_points": len(labels),
    }


def _parse_k_values(raw_values: Sequence[Sequence[str]] | None) -> list[int]:
    """Normalize one or more k values from CLI input (supports repeated or comma-separated lists)."""
    values: list[int] = []
    raw_entries = raw_values or []

    for group in raw_entries:
        if isinstance(group, str):
            group_tokens = [group]
        else:
            group_tokens = list(group)

        for token in group_tokens:
            for item in str(token).split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    value = int(item)
                except ValueError as exc:
                    raise argparse.ArgumentTypeError(f"invalid neighbor count: {item!r}") from exc
                if value <= 0:
                    raise argparse.ArgumentTypeError(f"neighbor count must be positive: {value}")
                values.append(value)

    if not values:
        return [10]
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how often a point's nearest neighbors belong to the same "
            "cluster as the point itself."
        )
    )
    parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="KNNG file or directory containing KNNG dump files",
    )
    parser.add_argument(
        "-N",
        "--no-distance",
        action="store_true",
        help="Indicate if the KNNG files *do not* contain distance information",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        required=True,
        help="Cluster label file or directory containing cluster-label files",
    )
    parser.add_argument(
        "-k",
        "--neighbors",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Neighbor count(s) to evaluate; accepts repeated uses or comma-separated "
            "values, e.g. -k 5 10 or -k 3,5,10"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output file path for the summary table",
    )
    args = parser.parse_args()
    args.neighbors = _parse_k_values(args.neighbors or [[10]])
    return args


def main() -> None:
    args = _parse_args()

    summary_lines: list[str] = []
    per_k_output: list[tuple[int, list[str]]] = []

    for k in args.neighbors:
        try:
            per_point, summary = evaluate_knng_clusters(args.graph, args.cluster, k, no_distance=args.no_distance)
            print(f"\nResults for k={k}:")
            print(summary["num_points"], "points evaluated,", summary["num_noise_points"], "noise points skipped.")
            print(summary["num_same_cluster_neighbors"], "same-cluster neighbors out of", summary["num_considered_neighbors"], "considered neighbors,", "fraction =", summary["avg_same_cluster_fraction"])
        except Exception as exc:  # pragma: no cover - CLI error path
            raise SystemExit(f"Error: {exc}") from exc

        summary_lines.append(f"=== k={k} ===")
        summary_lines.append(
            f"Evaluated {summary['num_points']} source points; "
            f"Skipped {summary['num_noise_points']} noise points"
        )
        summary_lines.append(
            f"Compared the first {k} neighbors per point; "
            f"{summary['num_same_cluster_neighbors']}/{summary['num_considered_neighbors']} "
            f"were in the same cluster (fraction={summary['avg_same_cluster_fraction']:.4f})"
        )

        per_k_lines = ["point_id  cluster_id  same_cluster  total_checked  fraction"]
        for source_id, cluster_id, same_count, checked_count, ratio in per_point:
            line = f"{source_id:>8}  {cluster_id:>10}  {same_count:>12}  {checked_count:>12}  {ratio:>8.4f}"
            per_k_lines.append(line)
        per_k_output.append((k, per_k_lines))

    for line in summary_lines:
        print(line)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(line for _, lines in per_k_output for line in lines) + "\n")
        print(f"\nPer-k statistics written to {output_path}")


if __name__ == "__main__":
    main()

