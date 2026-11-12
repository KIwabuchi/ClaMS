# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


"""

Calculate the cluster size histgram

Input: cluster label file

"""

import argparse

import numpy as np

from script.data_readers import read_cluster_labels


def parse_options():
    parser = argparse.ArgumentParser(
        description='Calculate the cluster size histgram')

    parser.add_argument('-c', '--cluster',
                        dest='cluster_path',
                        required=True, action='store', type=str,
                        help='Path to file or directory containing cluster labels')

    args = parser.parse_args()
    return args


def main():
    opt = parse_options()

    cluster_labels = read_cluster_labels(opt.cluster_path, "*")

    cluster_sizes = np.bincount(cluster_labels)
    print(cluster_sizes)


if __name__ == '__main__':
    main()
