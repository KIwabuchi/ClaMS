# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


"""
Benchmark  HDBSCAN
This script is a simple wrapper over the run_hdbscan.py script to benchmark HDBSCAN clustering.

Usage example:
python3 ../script/run_hdbscan_bench.py -p ../dataset/usps/points.txt -m 10,15,20 -s 8-10 -g ../dataset/usps/labels.txt
"""

import argparse
import hdbscan
import os
import numpy as np
import time

from clustering_utilities import *
from bench_utilities import *


def parse_options():
    parser = argparse.ArgumentParser(
        description='HDBSCAN clustering')

    # For input point (feature) data
    parser.add_argument('-p', '--point_data_path',
                        dest='point_data_path',
                        required=False, action='store', type=str,
                        help='Input point file path')

    parser.add_argument('-I', '--has_ids',
                        dest='has_ids',
                        required=False, action='store_true',
                        help='If specified, the input point file has point IDs')

    # HDBSCAN parameters
    # Min cluster size, conmma separated list of min cluster sizes, or range of min cluster sizes
    parser.add_argument('-m', '--min_cluster_size',
                        dest='min_cluster_size_range',
                        required=True, action='store', type=str,
                        help='Minimum cluster size. Comma separated list of values or range of values (e.g., 5,10,15 or 5-15)')

    # Min samples, comma separated list of min samples, or range of min samples
    parser.add_argument('-s', '--min_samples',
                        dest='min_samples_range',
                        required=False, action='store', type=str,
                        help='#of samples in a neighborhood for a point to be considered as a core point. Comma separated list of values or range of values (e.g., 5,10,15 or 5-15)'
                             'When None, defaults to min_cluster_size')

    # Output file paths
    parser.add_argument('-o', '--cluster_labels_out_path',
                        dest='cluster_labels_out_path',
                        required=False, action='store', type=str,
                        default='out_clusters.txt',
                        help='File path to store computed cluster IDs')

    parser.add_argument('-M', '--mst_out_path',
                        dest='mst_out_path',
                        required=False, action='store', type=str,
                        help='Output file path for intermediate MST data')

    # Ground truth file path
    parser.add_argument('-g', '--ground_truth',
                        dest='gt_file',
                        required=False, action='store', type=str,
                        help='Ground truth file path')

    args = parser.parse_args()
    return args


def run_hdbscan_kernel(points, min_cluster_size, min_samples,
                       gen_min_span_tree=False):
    clusters = hdbscan.HDBSCAN(gen_min_span_tree=gen_min_span_tree,
                               min_cluster_size=min_cluster_size,
                               min_samples=min_samples,
                               core_dist_n_jobs=-1)
    print(f'\nStart clustering', flush=True)
    show_time_now()
    # start timer
    start_time = time.time()
    clusters.fit(points)
    print(f'Finish clustering in {time.time() - start_time:.2f} seconds', flush=True)
    show_time_now()

    return clusters


def run_hdbscan(points, min_cluster_size, min_samples,
                cluster_labels_out_path='out_clusters.txt',
                gt_file=None,
                mst_out_path=None,
                condensed_tree_out_path=None,
                cluster_persistence_out_path=None,
                assign_cluster_to_noise=False):
    clusters = run_hdbscan_kernel(points, min_cluster_size, min_samples,
                                  gen_min_span_tree=(mst_out_path is not None))

    # Evaluate the clustering quality
    if gt_file:
        print('\nLoading ground truth data')
        gt_labels = read_label_data(gt_file)

        print('\nEvaluating clustering quality')
        eval_clusters(clusters.labels_, gt_labels)

        if assign_cluster_to_noise:
            print('\nAssigning a cluster ID to every noise point')
            no_noise_labels = assign_singleton_cluster_to_noise_point(
                clusters.labels_)
            eval_clusters(no_noise_labels, gt_labels)

    # Save the condensed tree data
    if condensed_tree_out_path:
        print(f'\nSaving condensed tree data in {condensed_tree_out_path}')
        clusters.condensed_tree_.to_pandas().to_csv(condensed_tree_out_path)

    # Save the MST data
    if mst_out_path:
        print(f'\nSaving MST data in {mst_out_path}')
        with open(mst_out_path, 'w') as fout_mst:
            mst = clusters.minimum_spanning_tree_.to_numpy()
            # Format: Point0, Point1, Distance
            for edge in mst:
                fout_mst.write(f'{int(edge[0])}\t{int(edge[1])}\t{edge[2]}\n')
            fout_mst.close()
            show_time_now()

    # Save the cluster IDs.
    if cluster_labels_out_path:
        print(f'\nSaving cluster IDs in {cluster_labels_out_path}')
        with open(cluster_labels_out_path, 'w') as fout:
            fout.write(f'# Node ID\tCluster ID\n')
            for i, label in enumerate(clusters.labels_):
                fout.write(f'{i}\t{label}\n')
            show_time_now()
            print(f'Cluster IDs are saved in {cluster_labels_out_path}')

    # Save the cluster persistence data
    if cluster_persistence_out_path:
        print(
            f'\nSaving cluster persistence data in {cluster_persistence_out_path}')
        with open(cluster_persistence_out_path, 'w') as fout:
            fout.write('Cluster ID\tPersistence\n')
            for i, persistence in enumerate(clusters.cluster_persistence_):
                fout.write(f'{i}\t{persistence}\n')


def main():
    opts = parse_options()


    points = read_point_data(opts.point_data_path, opts.has_ids)

    min_cluster_size_list = parse_range(opts.min_cluster_size_range)
    min_samples_list = parse_range(opts.min_samples_range)

    for min_cluster_size in min_cluster_size_list:
        for min_samples in min_samples_list:
            print(f'\n--------------------------------', flush=True)
            print((f"min_cluster_size: {min_cluster_size}"), flush=True)
            print((f"min_samples: {min_samples}"), flush=True)
            run_hdbscan(points, min_cluster_size, min_samples,
                        gt_file=opts.gt_file,
                        cluster_labels_out_path=opts.cluster_labels_out_path,
                        mst_out_path=opts.mst_out_path)


if __name__ == '__main__':
    main()
