#!/usr/bin/env python3
"""Evaluate clustering separation without ground-truth labels.

For every clustered point, traverse the directed kNNG to find reachable
same-cluster and different-cluster points with the smallest direct feature
distance from the source. The graph defines reachability only; stored edge
distances are ignored.

kNNG records use two lines per point:
	point_id neighbor_id_1 neighbor_id_2 ...
	dummy_distance distance_1 distance_2 ...

Cluster assignments contain two columns: point_id cluster_id. Cluster ID -1
denotes noise and is excluded as a source or candidate.
"""

from __future__ import annotations
from script.utilities import cosine_distance, l1_distance, l2_distance
from script.data_readers import read_cluster_labels, read_knng, read_points

import argparse
import csv
import heapq
import math
from collections import deque
from pathlib import Path
import sys
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


Graph: TypeAlias = dict[int, list[tuple[int, float | None]]]
DistanceFunction: TypeAlias = Callable[[ArrayLike, ArrayLike], float]
ResultRow: TypeAlias = dict[str, int | float | None]


def find_closest_cluster_point_dijkstra(
    source_id: int,
    labels: dict[int, int],
    graph: Graph,
    points: np.ndarray,
    dist_func: DistanceFunction,
) -> dict[str, tuple[int, int, float] | None]:
    """Find reachable cluster points nearest by direct feature distance.

    The graph determines reachability only. The frontier is prioritized by
    direct source-to-node distance. Returned values are
    ``(point_id, hops, direct_distance)``; equal candidate distances prefer
    fewer distances.
    """

    if source_id < 0 or source_id >= len(points):
        raise ValueError(
            "Source point ID is outside the loaded feature vectors")
    if source_id not in labels:
        raise ValueError(f"Source point ID {source_id} has no cluster label")

    source_cluster = labels[source_id]
    if source_cluster < 0:
        # Skip noise points as source
        return {"in": None, "out": None}

    # found contains:
    # "in" or "out": (closest_point_id, hops, distance) or None
    found: dict[str, tuple[int, int, float] | None] = {"in": None, "out": None}

    # Frontier priority queue contains tuples of (direct_distance, hops, point_id)
    frontier = [(0.0, 0, source_id)]

    visited: set[int] = set()
    visited.add(source_id)

    while frontier:
        source_distance, hops, point_id = heapq.heappop(frontier)

        neighbor_hops = hops + 1
        for neighbor_id, _stored_distance in graph.get(point_id, ()):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)

            neighbor_distance = float(
                dist_func(points[source_id], points[neighbor_id]))
            if not math.isfinite(neighbor_distance) or neighbor_distance < 0:
                raise ValueError(
                    "Distance function must return a finite non-negative value"
                )
            heapq.heappush(
                frontier,
                (neighbor_distance, neighbor_hops, neighbor_id),
            )

            category = "in" if labels[neighbor_id] == source_cluster else "out"
            current = found[category]
            if current is None or neighbor_distance < current[2]:
                found[category] = (
                    neighbor_id, neighbor_hops, neighbor_distance)

    return found


def find_closest_cluster_point_bfs(
    source_id: int,
    labels: dict[int, int],
    graph: Graph,
    points: np.ndarray,
    dist_func: DistanceFunction,
) -> dict[str, tuple[int, int, float] | None]:
    """Find closest cluster points by BFS depth, ranking same-depth ties by distance.

    The first reachable graph layer containing a point in each category is
    considered. Within that layer, direct feature distance selects the point.
    """
    if source_id < 0 or source_id >= len(points):
        raise ValueError(
            "Source point ID is outside the loaded feature vectors")
    if source_id not in labels:
        raise ValueError(f"Source point ID {source_id} has no cluster label")

    source_cluster = labels[source_id]
    if source_cluster < 0:
        return {"in": None, "out": None}

    found: dict[str, tuple[int, int, float] | None] = {"in": None, "out": None}
    visited = {source_id}
    frontier = deque([source_id])
    hops = 0

    while frontier and any(result is None for result in found.values()):
        next_frontier: list[int] = []
        layer_matches: dict[str, tuple[int, float]
                            | None] = {"in": None, "out": None}

        while frontier:
            point_id = frontier.popleft()
            if point_id != source_id and labels.get(point_id, -1) >= 0:
                category = "in" if labels[point_id] == source_cluster else "out"
                if found[category] is None:
                    distance = float(
                        dist_func(points[source_id], points[point_id]))
                    if not math.isfinite(distance) or distance < 0:
                        raise ValueError(
                            "Distance function must return a finite non-negative value"
                        )
                    current = layer_matches[category]
                    if current is None or (distance, point_id) < current:
                        layer_matches[category] = (distance, point_id)

            for neighbor_id, _stored_distance in graph.get(point_id, ()):
                if neighbor_id in visited:
                    continue
                if neighbor_id < 0 or neighbor_id >= len(points):
                    raise ValueError(
                        "Neighbor point ID is out of the expected range")
                visited.add(neighbor_id)
                next_frontier.append(neighbor_id)

        for category, match in layer_matches.items():
            if match is not None:
                distance, point_id = match
                found[category] = (point_id, hops, distance)

        frontier = deque(next_frontier)
        hops += 1

    return found


def compute_score(distance: float, epsilon: float) -> float:
    """Return the base-10 log score for a non-negative distance."""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    return math.log10(distance + epsilon)


def find_closest_reachable_distances(
    points: np.ndarray,
    graph: Graph,
    dist_func: DistanceFunction,
    labels: dict[int, int],
    algorithm: str = "bfs",
) -> list[ResultRow]:
    """Find closest reachable in- and out-of-cluster distances per point."""
    results: list[ResultRow] = []
    for source_id in sorted(labels):
        cluster_id = labels[source_id]
        if cluster_id < 0:
            continue
        if algorithm == "bfs":
            closest = find_closest_cluster_point_bfs(
                source_id, labels, graph, points, dist_func
            )
        elif algorithm == "dijkstra":
            closest = find_closest_cluster_point_dijkstra(
                source_id, labels, graph, points, dist_func
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        row: ResultRow = {
            "point_id": source_id,
            "cluster_id": cluster_id,
        }
        for category in ("in", "out"):
            match = closest[category]
            row[f"closest_{category}_point_id"] = match[0] if match else None
            row[f"closest_{category}_hops"] = match[1] if match else None
            row[f"closest_{category}_distance"] = match[2] if match else None
        results.append(row)
    return results


# def write_csv(path: str | Path, results: list[ResultRow]) -> None:
#     """Write per-point results as CSV."""
#     fields = [
#         "point_id",
#         "cluster_id",
#         "closest_in_point_id",
#         "closest_in_hops",
#         "closest_in_distance",
#         "closest_out_point_id",
#         "closest_out_hops",
#         "closest_out_distance",
#     ]
#     output_path = Path(path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with output_path.open("w", encoding="utf-8", newline="") as output_file:
#         writer = csv.DictWriter(output_file, fieldnames=fields)
#         writer.writeheader()
#         writer.writerows(results)


def write_txt(path: str | Path, results: list[ResultRow]) -> None:
    """Write per-point results as plain text."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Dump keys in the first line
    keys = list(results[0].keys()) if results else []
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("\t".join(keys) + "\n")
        for row in results:
            output_file.write("\t".join(str(row[key]) for key in keys) + "\n")


def compute_score_hist(results: list[ResultRow], epsilon: float, num_bins: int) -> dict[str, list[float]]:
    """Compute histograms of log scores for in-cluster and out-of-cluster distances."""
    hist = {"in": [0.0] * num_bins, "out": [0.0] * num_bins}

    # First, compute the min and max score
    min_scores = {"in": float("inf"), "out": float("inf")}
    max_scores = {"in": float("-inf"), "out": float("-inf")}
    for row in results:
        if row["closest_in_distance"] is not None:
            score = compute_score(float(row["closest_in_distance"]), epsilon)
            min_scores["in"] = min(min_scores["in"], score)
            max_scores["in"] = max(max_scores["in"], score)
        if row["closest_out_distance"] is not None:
            score = compute_score(float(row["closest_out_distance"]), epsilon)
            min_scores["out"] = min(min_scores["out"], score)
            max_scores["out"] = max(max_scores["out"], score)
    min_score = min(min_scores["in"], min_scores["out"])
    max_score = max(max_scores["in"], max_scores["out"])

    bin_size = max((max_score - min_score) / num_bins, 1e-12)

    for row in results:
        if row["closest_in_distance"] is not None:
            score = compute_score(float(row["closest_in_distance"]), epsilon)
            bin_index = min(int((score - min_score) / bin_size), num_bins - 1)
            hist["in"][bin_index] += 1
        if row["closest_out_distance"] is not None:
            score = compute_score(float(row["closest_out_distance"]), epsilon)
            bin_index = min(int((score - min_score) / bin_size), num_bins - 1)
            hist["out"][bin_index] += 1
    return hist


def print_summary(results: list[ResultRow]) -> None:
    """Print counts and means of distances by category, and the global score."""
    print(f"Evaluated {len(results)} clustered points")
    for category, description in (("in", "in-cluster"), ("out", "out-of-cluster")):
        values = [
            float(row[f"closest_{category}_distance"])
            for row in results
            if row[f"closest_{category}_distance"] is not None
        ]
        if values:
            print(
                f"Closest {description}: {len(values)} points; "
                f"mean distance = {sum(values) / len(values):.6f}"
            )
        else:
            print(f"Closest {description}: no reachable candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--points", required=True,
                        help="Feature vectors file or directory")
    parser.add_argument(
        "-I", "--ids", action="store_true",
        help="Feature-vector rows include point IDs",
    )
    parser.add_argument("-g", "--graph", required=True,
                        help="kNNG file or directory of files")
    parser.add_argument(
        "--knng-format",
        choices=("M", "MI", "B", "BI"),
        help="Distance-bearing kNNG format (default: MI)",
        required=True,
    )
    parser.add_argument(
        "-f", "--dist_func", choices=("l2", "l1", "cosine"),
        help="Distance function to use",
        required=True,
    )
    parser.add_argument("-c", "--cluster", required=True,
                        help="Point ID and cluster ID file")
    parser.add_argument(
        "-e", "--epsilon", type=float, default=0.001,
        help="Positive value added before taking log10 (default: 0.001)",
    )
    parser.add_argument(
        "-a", "--algorithm", action="store",
        help="Traversal algorithm used (BFS or Dijkstra)", choices=("bfs", "dijkstra"),
        default="bfs",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-o", "--output", help="Optional per-point CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        print("Reading points...")
        points = read_points(args.points, pattern=None, has_ids=args.ids)
        print(f"Loaded {len(points)} points")

        print("Reading graph...")
        graph = read_knng(args.graph, args.knng_format)
        print(f"Loaded {len(graph)} graph points")

        print("Reading cluster labels...")
        labels_array = read_cluster_labels(args.cluster)
        labels = {
            point_id: int(label)
            for point_id, label in enumerate(labels_array)
            if label >= 0
        }
        print(f"Loaded {len(labels)} assigned cluster labels")

        distance_functions: dict[str, DistanceFunction] = {
            "l2": l2_distance,
            "l1": l1_distance,
            "cosine": cosine_distance,
        }
        results = find_closest_reachable_distances(
            points, graph, distance_functions[args.dist_func], labels, algorithm=args.algorithm
        )

        print_summary(results)

        hist = compute_score_hist(results, epsilon=args.epsilon, num_bins=100)
        print("Histogram of closest reachable distances:")
        for category, counts in hist.items():
            print(f"{category}: {counts}")

        if args.output:
            file_name = args.output + "_scores.txt"
            print(f"Writing per-point results to {file_name}...")
            write_txt(file_name, results)
            print(f"Per-point results written to {file_name}")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
