# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import datetime
import os


# Modified from:
# https://gist.github.com/lmcinnes/24ed5c22c80125be5133811d677eae7b
def eval_clusters(cluster_labels, true_labels, singleton_cluster_to_noise_points=False, ignore_true_noise_points=False):
    max_cluster_id = max(np.max(cluster_labels), np.max(true_labels))

    if np.any(true_labels < 0):
        print("Ground truth labels contain noise points")
        pct_clustered_gt = (np.sum(true_labels >= 0) / cluster_labels.shape[0])
        print(f"GT clustered Points: {pct_clustered_gt * 100:.2f}%")

        if ignore_true_noise_points:
            print("Remove true noise points from the evaluation")
            print(f"Before filtering: {len(true_labels)} points")
            mask = true_labels >= 0
            true_labels = true_labels[mask]
            cluster_labels = cluster_labels[mask]
            max_cluster_id = max(np.max(cluster_labels), np.max(true_labels))
            print(f"After filtering: {len(true_labels)} points")


    if np.any(cluster_labels < 0):  # Has noise points
        clustered_points = (cluster_labels >= 0)
        pct_clustered = (np.sum(clustered_points) / cluster_labels.shape[0])
        print(f"Cluster coverage (%): {pct_clustered * 100:.2f}")

        if singleton_cluster_to_noise_points:
            print(
                "Assigning a singleton cluster to each noise point in the clustering result")
            cluster_labels = assign_singleton_cluster_to_noise_points(cluster_labels, max_cluster_id)
            ari = adjusted_rand_score(true_labels, cluster_labels)
            ami = adjusted_mutual_info_score(true_labels, cluster_labels)
        else:
            ari = adjusted_rand_score(true_labels[clustered_points],
                                    cluster_labels[clustered_points])
            ami = adjusted_mutual_info_score(true_labels[clustered_points],
                                            cluster_labels[clustered_points])
            # sil = silhouette_score(raw_data[clustered_points], cluster_labels[clustered_points])
        pct_clustered = (np.sum(clustered_points) / cluster_labels.shape[0])
        print(f"Cluster coverage (%): {pct_clustered * 100:.2f}")

        if singleton_cluster_to_noise_points:
            print(
                "Assigning a singleton cluster to each noise point in the clustering result")
            cluster_labels = assign_singleton_cluster_to_noise_points(cluster_labels, max_cluster_id)
            ari = adjusted_rand_score(true_labels, cluster_labels)
            ami = adjusted_mutual_info_score(true_labels, cluster_labels)
        else:
            ari = adjusted_rand_score(true_labels[clustered_points],
                                    cluster_labels[clustered_points])
            ami = adjusted_mutual_info_score(true_labels[clustered_points],
                                            cluster_labels[clustered_points])
            # sil = silhouette_score(raw_data[clustered_points], cluster_labels[clustered_points])
    else:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        ami = adjusted_mutual_info_score(true_labels, cluster_labels)
        # sil = silhouette_score(raw_data, cluster_labels)
        print(f"No noise points in the clustering result")

    print(f"ARI: {ari:.4f}")
    print(f"AMI: {ami:.4f}")


# Assign a cluster ID to each noise point
def assign_singleton_cluster_to_noise_points(cluster_labels, noise_id_offset):
def assign_singleton_cluster_to_noise_points(cluster_labels, noise_id_offset):
    new_labels = cluster_labels.copy()

    cnt_noise = 0
    for i, label in enumerate(cluster_labels):
        if label == -1:
            cnt_noise += 1
            new_labels[i] = cnt_noise + noise_id_offset
            new_labels[i] = cnt_noise + noise_id_offset

    return new_labels


def show_time_now():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


def find_files_in_dir(dir_path, ext=''):
    if not os.path.isdir(dir_path):
        return [dir_path]

    files = []
    for file in os.listdir(dir_path):
        if len(ext) == 0 or file.endswith(ext):
            files.append(os.path.join(dir_path, file))
    return files


# Read point(feature) data from file(s)
# If data_path is a directory, load all files in the directory.
# Return a dense numpy array of feature vectors.
# Each row is a feature vector.
# Input files can have point IDs in the first column.
# Point IDs are expected to be 0, 1, 2, ..., N-1,
# where N is the number of points.
# If the first column is not a point ID, set has_ids to False.
# If there are multiple files in a directory, the point IDs must be present.
def read_point_data(data_path, has_ids=True):
    print(f"Loading data from {data_path}")
    files = find_files_in_dir(data_path)

    if len(files) == 0:
        print(f"No files found in {data_path}")
        exit(1)

    if len(files) > 1 and not has_ids:
        print("Multiple files are provided,"
              "but has_ids is false.")

    points_table = {}
    dimensions = 0
    for file in files:
        for line in open(file, 'r'):
            if len(line.strip()) == 0:
                print(f"Empty line found in {file}")
                continue
            items = line.split()
            if has_ids:
                pid = int(items[0])
                items = items[1:]
            else:
                pid = len(points_table)

            if pid in points_table:
                print(f"Duplicate ID in the feature file: {pid}")
                exit(1)

            if dimensions == 0:
                dimensions = len(items)
            elif dimensions != len(items):
                print(
                    f"Dimension mismatch in {file}: {len(items)} != {dimensions}")
                exit(1)

            points_table[pid] = list(map(float, items))

    print(f"Loaded {len(points_table)} items from {len(files)} files")

    # numpy array of feature vectors
    # if the IDs are not continuous, fill the missing IDs with -1
    max_id = max(points_table.keys())
    points = np.full((max_id + 1, len(points_table[0])), 0.0)
    for id, feature in points_table.items():
        points[id] = feature

    return points


# Load cluster label or ground truth label data
# If data_path is a directory, load all files in the directory
# Return a dense numpy array of labels
# Unclustered points are filled with -1
#
# There are two accepted file types:
#
# 1. Only cluster IDs:
# Each line contains a cluster ID.
# Point IDs are assumed to be 0, 1, 2, ..., N-1, where N is the number of points.
# This mode is not acceptable if there are multiple files in a directory.
#
# 2. Point IDs and cluster IDs:
# Contains two columns:
# the first column is for point IDs and the second column is for cluster IDs.
#
# Both File types can also contain comment lines, which must start from #.
def read_label_data(data_path):
    print(f"Loading data from {data_path}")
    labels_dict = {}
    files = []
    if os.path.isdir(data_path):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    else:
        files.append(data_path)

    if len(files) == 0:
        print(f"No files found in {data_path}")
        exit(1)

    contains_ids = False
    for line in open(files[0], 'r'):
        if len(line.split()) >= 2:
            contains_ids = True
            break

    if contains_ids:
        print("Loading point IDs and labels")
    else:
        print("Loading only labels")

    if len(files) > 1 and not contains_ids:
        print("Multiple files are provided,"
              "but no IDs are found in the first file.")

    max_id = 0
    for file in files:
        for line in open(file, 'r'):

            # Skip comments
            if line.startswith("#"):
                continue

            if contains_ids:
                items = line.split()
                if len(items) < 2:
                    print(f"Invalid line: {line}")
                    exit(1)
                pid = int(items[0])
                label = int(items[1])
                max_id = max(max_id, pid)
            else:
                pid = len(labels_dict)
                max_id = pid
                label = int(line)

            if pid in labels_dict:
                print(f"Duplicate ID in the ground truth file: {pid}")
                exit(1)
            labels_dict[pid] = label

    print(f"Loaded {len(labels_dict)} items from {len(files)} files")
    print(f"Max ID: {max_id}")

    # numpy array of labels
    # if the IDs are not continuous, fill the missing IDs with -1
    labels = np.full(max_id + 1, -1)
    for id, label in labels_dict.items():
        labels[id] = label

    return labels
