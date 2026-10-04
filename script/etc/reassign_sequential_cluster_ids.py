# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


"""""
Input file example
--
# Node ID       Cluster ID
0 8
1 8
2 4
--

Convert the cluster IDs to packed and 0-based cluster IDs.
The lower cluster ID will be 0, and the higher cluster ID will be N-1, where N is the number of clusters.
Cluster IDs are assigned from lower node IDs.

Output file example
--
# Node ID       Cluster ID
0 0
1 0
2 1
--
"""""

import os
import argparse
import numpy as np
from script.benchmark.clustering_utilities import *
from script.utilities import *
from script.data_readers import *

def main():
    parser = argparse.ArgumentParser(
        description='Reassign cluster IDs sequentially')
    parser.add_argument('-i', '--input',
                        dest='input_cluster_ids_path',
                        required=True, action='store', type=str,
                        help='Path to file containing cluster labels')
    parser.add_argument('-o', '--output',
                        dest='output_path',
                        required=True, action='store', type=str,
                        help='Path to the output file')
    args = parser.parse_args()

    cluster_ids = read_cluster_labels(args.input_cluster_ids_path, "*")

    # Make new cluster IDs
    new_ids = dict()
    for label in cluster_ids:
        if label not in new_ids:
            new_ids[label] = len(new_ids)

    # Save the new cluster IDs
    with open(args.output_path, 'w') as file:
        file.write('# Node ID\tCluster ID\n')
        for i, label in enumerate(cluster_ids):
            file.write(f'{i}\t{new_ids[label]}\n')

    print(f'New cluster IDs are saved in {args.output_path}')


if __name__ == '__main__':
    main()
