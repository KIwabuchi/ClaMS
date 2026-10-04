# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import datetime
import os
import sys
from script.utilities import *

# Modified from:
# https://gist.github.com/lmcinnes/24ed5c22c80125be5133811d677eae7b
def eval_clusters(cluster_labels, true_labels, singleton_cluster_to_noise_points=False, ignore_true_noise_points=False):
    print(f"Assigning singleton clusters to noise points: {singleton_cluster_to_noise_points}")
    print(f"Ignore true noise points in the ground truth: {ignore_true_noise_points}")
    max_cluster_id = max(np.max(cluster_labels), np.max(true_labels))

    if np.any(true_labels < 0):
        print("Ground truth labels contain noise points")
        pct_clustered_gt = (np.sum(true_labels >= 0) / cluster_labels.shape[0])
        print(f"Ground truth cluster coverage (%): {pct_clustered_gt * 100:.2f}%")

        if ignore_true_noise_points:
            print("Remove noise points in the ground truth from the evaluation")
            print(f"Before filtering: {len(true_labels)} ground truth points")
            mask = true_labels >= 0
            true_labels = true_labels[mask]
            cluster_labels = cluster_labels[mask]
            max_cluster_id = max(np.max(cluster_labels), np.max(true_labels))
            print(f"After filtering: {len(true_labels)} ground truth points")

    non_noise_points_mask = (cluster_labels >= 0)
    num_non_noise_points = np.sum(non_noise_points_mask)
    pct_clustered = (num_non_noise_points / cluster_labels.shape[0])
    print(f"Cluster coverage (%): {pct_clustered * 100:.2f}%")

    # Make sure always num_non_noise_points <= cluster_labels.shape[0]
    assert num_non_noise_points <= cluster_labels.shape[0]

    if num_non_noise_points != cluster_labels.shape[0]:  # Has noise points
        if singleton_cluster_to_noise_points:
            print(
                "Assigning a singleton cluster to each noise point in the clustering result")
            cluster_labels = assign_singleton_cluster_to_noise_points(cluster_labels, max_cluster_id)
            ari = adjusted_rand_score(true_labels, cluster_labels)
            ami = adjusted_mutual_info_score(true_labels, cluster_labels)
        else:
            print("Noise points are ignored in the evaluation")
            ari = adjusted_rand_score(true_labels[non_noise_points_mask],
                                    cluster_labels[non_noise_points_mask])
            ami = adjusted_mutual_info_score(true_labels[non_noise_points_mask],
                                            cluster_labels[non_noise_points_mask])
            # sil = silhouette_score(raw_data[non_noise_points_mask], cluster_labels[non_noise_points_mask])
    else:
        print(f"No noise points in the clustering result")
        ari = adjusted_rand_score(true_labels, cluster_labels)
        ami = adjusted_mutual_info_score(true_labels, cluster_labels)
        # sil = silhouette_score(raw_data, cluster_labels)

    print(f"ARI: {ari:.4f}")
    print(f"AMI: {ami:.4f}")


# Assign a cluster ID to each noise point
def assign_singleton_cluster_to_noise_points(cluster_labels, noise_id_offset):
    new_labels = cluster_labels.copy()

    cnt_noise = 0
    for i, label in enumerate(cluster_labels):
        if label == -1:
            cnt_noise += 1
            new_labels[i] = cnt_noise + noise_id_offset
            new_labels[i] = cnt_noise + noise_id_offset

    print(f"Assigned singleton clusters to {cnt_noise} noise points")

    return new_labels
