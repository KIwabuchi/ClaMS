# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


"""

Compute the clustering quality metrics (ARI, AMI).
Noise points (cluster id is -1) are ignored in the evaluation.
Takes two data: cluster labels and ground truth labels.

Usage:
    python eval_clustering_quality.py -c cluster.txt -g gt.txt
    or
    python eval_clustering_quality.py -c cluster_dir -g gt_dir

Arguments:
    -c: path to the cluster file or directory containing cluster files (required)
    -g: path to the ground truth file or directory containing ground truth files (required)

There are two accepted file types:

1. Only cluster IDs:
Each line contains a cluster ID.
Point IDs are assumed to be 0, 1, 2, ..., N-1, where N is the number of points.
This mode is not acceptable if there are multiple files in a directory.

2. Point IDs and cluster IDs:
Contains two columns:
the first column is for point IDs and the second column is for cluster IDs.

Both File types can also contain comment lines, which must start from #.

"""


import os
import argparse
import numpy as np
from clustering_utilities import *


def parse_options():
    parser = argparse.ArgumentParser(
        description='Evaluate clustering quality')

    parser.add_argument('-c', '--cluster',
                        dest='cluster_path',
                        required=True, action='store', type=str,
                        help='Path to file or directory containing cluster labels')
    parser.add_argument('-g', '--ground_truth',
                        dest='gt_path',
                        required=True, action='store', type=str,
                        help='Path to file or directory containing ground truth labels')
    parser.add_argument('-s', '--singleton_noise_points', action='store_true',
                        help='Assign a singleton cluster to each noise point in the clustering labels to evaluate.')
    parser.add_argument('-T', '--ignore_true_noise_points', action='store_true',
                        help='Remove true noise points from the evaluation.')

    args = parser.parse_args()
    return args


def main():
    opt = parse_options()

    cluster_labels = read_label_data(opt.cluster_path)
    true_labels = read_label_data(opt.gt_path)

    try:
        eval_clusters(cluster_labels, true_labels, opt.singleton_noise_points, opt.ignore_true_noise_points)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
