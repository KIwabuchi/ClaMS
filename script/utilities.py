# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

from __future__ import annotations
import datetime
from collections.abc import Sequence
import glob
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_vector_pair(
    vec1: ArrayLike, vec2: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert and validate two finite, non-empty one-dimensional vectors."""
    first = np.asarray(vec1, dtype=np.float64)
    second = np.asarray(vec2, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Vectors must be one-dimensional")
    if first.size != second.size:
        raise ValueError("Vectors must have the same length")
    if first.size == 0:
        raise ValueError("Vectors must not be empty")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Vectors must contain only finite values")
    return first, second


def show_time_now():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


def find_files_in_dir(
    paths: Sequence[str | Path], pattern: str | None = None
) -> list[Path]:
    """Resolve files, directories, and glob patterns to unique sorted paths.

    ``pattern`` filters entries inside directory inputs. Nonexistent path inputs
    are treated as glob patterns. Directory searches are not recursive.
    """
    resolved: set[Path] = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            matches = path.glob(pattern or "*")
        elif path.is_file():
            matches = (path,)
        else:
            # Nonexistent inputs may still be glob expressions, such as *.txt.
            matches = (Path(match) for match in glob.glob(str(value)))

        resolved.update(match.resolve() for match in matches if match.is_file())

    if not resolved:
        formatted_paths = ", ".join(map(str, paths))
        raise FileNotFoundError(f"No files matched: {formatted_paths}")

    return sorted(resolved)


def create_dir(dir_path: str | Path) -> None:
    """Create a directory and any missing parent directories if needed."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def grep_file(file: str | Path, text_pattern: str) -> list[str]:
    """Return the lines in ``file`` that contain the given substring."""
    with Path(file).open("r", encoding="utf-8") as input_file:
        return [line for line in input_file if text_pattern in line]


def parse_range(range_str: str | None) -> list[int | None]:
    """Parse an inclusive integer range or comma-separated integer list.

    Examples: ``"1-3"`` becomes ``[1, 2, 3]`` and ``"1,3"`` becomes
    ``[1, 3]``. ``None`` is preserved as ``[None]`` for optional CLI values.
    """
    if range_str is None:
        return [None]

    value = range_str.strip()
    if not value:
        raise ValueError("range must not be empty")

    if "-" in value:
        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid integer range: {range_str!r}")
        start, end = (int(part.strip()) for part in parts)
        if start > end:
            raise ValueError(f"range start must not exceed its end: {range_str!r}")
        return list(range(start, end + 1))

    return [int(part.strip()) for part in value.split(",")]


def parse_float_list(float_list_str: str | None) -> list[float | None]:
    """Parse comma-separated floating-point values into a list.

    ``None`` is preserved as ``[None]`` for optional CLI values.
    """
    if float_list_str is None:
        return [None]

    value = float_list_str.strip()
    if not value:
        raise ValueError("float list must not be empty")
    return [float(part.strip()) for part in value.split(",")]


def execute_cmd(
    log_file: str | Path,
    err_log_file: str | Path,
    command: str,
    cwd: str | Path = "./",
) -> None:
    """Run a shell command, append its output to logs, and exit on failure.

    The command is interpreted by the system shell. Standard output and
    standard error are appended to their respective log files.
    """
    print(f"In {Path(cwd).resolve()}")
    print(f"Command: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    with Path(log_file).open("a", encoding="utf-8") as output_file:
        output_file.write(f"Command: {command}\n")
        output_file.write(result.stdout)
    with Path(err_log_file).open("a", encoding="utf-8") as error_file:
        error_file.write(result.stderr)

    if result.returncode == 0:
        print("Command executed successfully.\n")
    else:
        print(f"Error executing command. See {err_log_file}.\n")
        raise SystemExit(result.returncode)


# Distance (similarity) functions
# The smaller the distance value, the more similar the vectors are.
# Inputs are numpy arrays or lists of floats.
def l2_distance(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """Compute the Euclidean (L2) distance between two vectors."""
    first, second = _as_vector_pair(vec1, vec2)
    return float(np.linalg.norm(first - second))


def l1_distance(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """Compute the Manhattan (L1) distance between two vectors."""
    first, second = _as_vector_pair(vec1, vec2)
    return float(np.sum(np.abs(first - second)))


def cosine_distance(vec1: ArrayLike, vec2: ArrayLike) -> float:
    """Compute the Cosine distance between two vectors (1 - cosine similarity)."""
    first, second = _as_vector_pair(vec1, vec2)
    norm1 = np.linalg.norm(first)
    norm2 = np.linalg.norm(second)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Vectors must not be zero vectors")

    cosine_similarity = np.dot(first / norm1, second / norm2)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)
