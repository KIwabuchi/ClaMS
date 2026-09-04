# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


"""
Benchmark  HDBSCAN
This script is a simple wrapper over the run_hdbscan.py script to benchmark HDBSCAN clustering.

Usage example:
python3 ../script/run_hdbscan_bench.py -p ../dataset/usps/points.txt -m 10,15,20 -s 8-10 -g ../dataset/usps/labels.txt
"""

import argparse
from clustering_utilities import *
from script.benchmark.bench_utilities import *
from script.benchmark.hdbscan.run_hdbscan import run_hdbscan


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


def main():
    opts = parse_options()

    points = read_point_data(opts.point_data_path, opts.has_ids)

    min_cluster_size_list = parse_range(opts.min_cluster_size_range)
    min_samples_list = parse_range(opts.min_samples_range)

    for min_cluster_size in min_cluster_size_list:
        for min_samples in min_samples_list:
            print(f'\n--------------------------------')
            print((f"min_cluster_size: {min_cluster_size}"))
            print((f"min_samples: {min_samples}"))
            run_hdbscan(points, min_cluster_size, min_samples,
                        gt_file=opts.gt_file,
                        cluster_labels_out_path=opts.cluster_labels_out_path,
                        mst_out_path=opts.mst_out_path)


if __name__ == '__main__':
    main()
