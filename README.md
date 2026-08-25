
# Clustering at Massive Scale (ClaMS)

ClaMS is a distributed memory HPC clustering tool inspired by
[HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). The goal of this work is to provide a tool
for clustering datasets with billions of points in general metric spaces for users with access to HPC systems. The
general approach taken is to take the standard HDBSCAN algorithm and swap components that do not scale or apply to
non-Euclidean data for scalable primitives. This often requires resorting to algorithms that are approximations of what
is done in HDBSCAN, sometimes without approximation guarantees.

Note: This code works best if the input data is de-duplicated, with exact duplicate points removed. 
If not, HDBSCAN will identify clusters with duplicate points as highly stable and always select them.


## Build
```shell


git clone https://github.com/llnl/clams.git
cd clams
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# If one runs into latomic link issue with Boost >= v1.87,
# try the following cmake option:
# -DBOOST_UUID_LINK_LIBATOMIC=OFF

# Optional
# Install the Python dependencies
# Python dependencies are used for clustering metric evaluation (non YGM-based),
# generating demo datasets, or benchmarking HDBSCAN.
# If one does not need these functionalities, the Python dependencies are not required.
python3 -m venv ./venv
source ./venv/bin/activate
pip install scikit-learn hdbscan numpy pandas
```

## Demo Datasets

The `dataset` directory contains scripts to achieve some famous datasets.

## Run

Currently, the benchmark scripts are designed for running jobs on slurm-based (srun and sbatch) HPC systems.

## Run specifying dataset

```shell
cd clams/build
# Activate the Python virtual environment to use the Python dependencies, if needed
source ./venv/bin/activate

# Generate the shell script
# Use '-h' option to see the help
# '-S' option to submit a sbatch job using a generated batch script
# Shows the path to the generated script at the end

## Without '-y' option, uses Python script for clustering metric evaluation
python3 ../script/benchmark/run_clams_bench.py -p point_features.txt -g ground_truth_labels.txt -S

## To instead use YGM script for clustering metric evaluation, use '-y' option
## In this case, the labels need to be in a file with each row in the form "point_id label"
python3 ../script/benchmark/run_clams_bench.py -p point_features.txt -g ground_truth_labels_with_point_ids.txt -y -S

# To submit a job manually using the generated script
sbatch /path/to/generated_batch_script.sh

# Use a demo dataset
pushd ../dataset/mnist
python3 ./download_mnist.py
popd
python3 ../script/benchmark/run_clams_bench.py -S -p ../dataset/mnist/points.txt -g ../dataset/mnist/labels.txt
```

## Running HDBSCAN

```shell
cd clams/build
source ./venv/bin/activate

# In clams/build
# -m: min cluster size
# -s: min samples
python3 ../script/benchmark/hdbscan/run_hdbscan.py -m 10 -s 5 -p ../dataset/fashion-mnist/points.txt -g ../dataset/fashion-mnist/labels.txt
```

# License

This project is licensed under the BSD-Commercial license – see the [LICENSE](LICENSE) file for details.

# Release
LLNL-CODE-2015147
