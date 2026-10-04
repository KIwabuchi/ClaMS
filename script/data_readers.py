# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from .utilities import find_files_in_dir

def read_points(
    data_path: str | Path,
    pattern: str | None = None,
    has_ids: bool = True,
) -> np.ndarray:
    """Load whitespace-separated feature vectors into a dense NumPy array.

    Each non-empty row contains feature values, optionally preceded by a point
    ID. When IDs are present, missing IDs are represented by 0-based row numbers.
    Multiple files require explicit IDs so records can be combined unambiguously.
    """
    print(f"Loading data from {data_path}", flush=True)
    files = find_files_in_dir([data_path], pattern)
    if len(files) > 1 and not has_ids:
        raise ValueError("Multiple point files require point IDs")

    points_table: dict[int, list[float]] = {}
    dimensions: int | None = None
    for file in files:
        with file.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                items = line.split()
                if not items:
                    continue

                if has_ids:
                    try:
                        point_id = int(items.pop(0))
                    except ValueError as exc:
                        raise ValueError(
                            f"{file}:{line_number}: invalid point ID"
                        ) from exc
                    if point_id < 0:
                        raise ValueError(f"{file}:{line_number}: point IDs must be non-negative")
                else:
                    point_id = len(points_table)

                if not items:
                    raise ValueError(f"{file}:{line_number}: feature vector is empty")
                if point_id in points_table:
                    raise ValueError(f"{file}:{line_number}: duplicate point ID {point_id}")

                try:
                    feature = [float(value) for value in items]
                except ValueError as exc:
                    raise ValueError(f"{file}:{line_number}: invalid feature value") from exc

                if dimensions is None:
                    dimensions = len(feature)
                elif len(feature) != dimensions:
                    raise ValueError(
                        f"{file}:{line_number}: dimension mismatch: "
                        f"{len(feature)} != {dimensions}"
                    )
                points_table[point_id] = feature

    if not points_table or dimensions is None:
        raise ValueError(f"No feature vectors were loaded from {data_path}")

    print(f"Loaded {len(points_table)} items from {len(files)} files", flush=True)

    points = np.zeros((max(points_table) + 1, dimensions), dtype=float)
    for point_id, feature in points_table.items():
        points[point_id] = feature

    return points


def read_cluster_labels(
    data_path: str | Path,
    pattern: str | None = None,
) -> np.ndarray:
    """Load cluster labels from one or more files into a dense integer array.

    Rows may contain only a cluster ID (point ID is the 0-based row number) or a
    point ID followed by a cluster ID. Blank lines and comments are ignored.
    Multiple files require explicit IDs.
    """
    files = find_files_in_dir([data_path], pattern)

    print(f"Found {len(files)} files in {data_path}", flush=True)
    first_data_row: list[str] | None = None
    for file in files:
        with file.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    first_data_row = stripped.split()
                    break
        if first_data_row is not None:
            break

    if first_data_row is None:
        raise ValueError(f"No cluster labels were found in {data_path}")

    contains_ids = len(first_data_row) >= 2

    if contains_ids:
        print("Loading point IDs and labels", flush=True)
    else:
        print("Loading only labels", flush=True)

    if len(files) > 1 and not contains_ids:
        raise ValueError("Multiple cluster files require point IDs")

    labels_dict: dict[int, int] = {}
    for file in files:
        with file.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                items = stripped.split()

                try:
                    if contains_ids:
                        if len(items) < 2:
                            raise ValueError("expected point ID and cluster ID")
                        point_id, label = int(items[0]), int(items[1])
                    else:
                        point_id, label = len(labels_dict), int(items[0])
                except ValueError as exc:
                    raise ValueError(f"{file}:{line_number}: invalid cluster row") from exc

                if point_id < 0:
                    raise ValueError(f"{file}:{line_number}: point IDs must be non-negative")
                if point_id in labels_dict:
                    raise ValueError(f"{file}:{line_number}: duplicate point ID {point_id}")
                labels_dict[point_id] = label

    if not labels_dict:
        raise ValueError(f"No cluster labels were found in {data_path}")

    print(f"Loaded {len(labels_dict)} items from {len(files)} files", flush=True)
    max_id = max(labels_dict)
    print(f"Max ID: {max_id}", flush=True)

    labels = np.full(max_id + 1, -1)
    for point_id, label in labels_dict.items():
        labels[point_id] = label

    return labels

Graph: TypeAlias = dict[int, list[tuple[int, float]]]

def read_knng(
    graph_path: str | Path,
    knng_format: str,
    path_pattern: str | None = None,
) -> Graph:
    """Load a kNN graph as source IDs mapped to neighbor/distance pairs.

    ``knng_format`` selects ``N`` (neighbor IDs only), ``B`` (blocked ID rows
    followed by their distance rows), or ``M`` (alternating ID and distance
    rows). Append ``I`` to any format when each ID row starts with its source
    point ID; otherwise source IDs are assigned sequentially across the files.
    Distances are ``None`` for the ID-only format. With explicit source IDs,
    the corresponding first distance is treated as the dummy source distance.
    """
    format_code = knng_format.upper()
    base_formats = [code for code in format_code if code in "NBM"]
    if len(base_formats) != 1 or any(code not in "NBMI" for code in format_code):
        raise ValueError(f"Unsupported kNNG format: {knng_format!r}")
    base_format = base_formats[0]
    contains_ids = "I" in format_code

    files = find_files_in_dir([graph_path], path_pattern)
    graph: dict[int, list[tuple[int, float | None]]] = {}
    next_source_id = 0

    def parse_id_row(file: Path, line_number: int, line: str) -> tuple[int, list[int]]:
        nonlocal next_source_id
        try:
            values = [int(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(f"{file}:{line_number}: invalid neighbor ID row") from exc
        if contains_ids:
            if not values:
                raise ValueError(f"{file}:{line_number}: missing source point ID")
            source_id, neighbor_ids = values[0], values[1:]
        else:
            source_id, neighbor_ids = next_source_id, values
            next_source_id += 1
        if source_id < 0 or any(neighbor_id < 0 for neighbor_id in neighbor_ids):
            raise ValueError(f"{file}:{line_number}: point IDs must be non-negative")
        if source_id in graph:
            raise ValueError(f"{file}:{line_number}: duplicate source point ID {source_id}")
        return source_id, neighbor_ids

    def parse_distances(file: Path, line_number: int, line: str) -> list[float]:
        try:
            distances = [float(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(f"{file}:{line_number}: invalid distance row") from exc
        if any(not math.isfinite(distance) or distance < 0 for distance in distances):
            raise ValueError(f"{file}:{line_number}: distances must be finite and non-negative")
        return distances

    for file in files:
        with file.open("r", encoding="utf-8") as input_file:
            lines = [
                (line_number, line.strip())
                for line_number, line in enumerate(input_file, start=1)
                if line.strip() and not line.lstrip().startswith("#")
            ]

        if base_format == "N":
            for line_number, line in lines:
                source_id, neighbor_ids = parse_id_row(file, line_number, line)
                graph[source_id] = [(neighbor_id, None) for neighbor_id in neighbor_ids]
            continue

        if len(lines) % 2 != 0:
            raise ValueError(f"{file}: expected an even number of kNNG lines")
        half = len(lines) // 2 if base_format == "B" else None
        if base_format == "B":
            id_rows = lines[:half]
            distance_rows = lines[half:]
        else:
            id_rows = lines[::2]
            distance_rows = lines[1::2]

        for (id_line_number, id_line), (distance_line_number, distance_line) in zip(
            id_rows, distance_rows
        ):
            source_id, neighbor_ids = parse_id_row(file, id_line_number, id_line)
            distances = parse_distances(file, distance_line_number, distance_line)
            if contains_ids:
                if len(distances) != len(neighbor_ids) + 1:
                    raise ValueError(
                        f"{file}:{distance_line_number}: expected a dummy distance and "
                        f"{len(neighbor_ids)} neighbor distances"
                    )
                distances = distances[1:]
            elif len(distances) != len(neighbor_ids):
                raise ValueError(
                    f"{file}:{distance_line_number}: {len(neighbor_ids)} neighbors but "
                    f"{len(distances)} distances"
                )
            graph[source_id] = list(zip(neighbor_ids, distances))

    if not graph:
        raise ValueError(f"No kNNG rows were loaded from {graph_path}")
    return graph
