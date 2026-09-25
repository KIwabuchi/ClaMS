"""
This script installs third-party libraries (TPLs) for running HPC-scale clustering.
The script clones and builds a YGM-based partition comparison

Usage:
    # Assuming we are in 'build' directory
    python ../script/install_tpls.py [-j number_of_jobs] [-d destination]
"""


import os
import subprocess
import argparse
from datetime import datetime

# Execute a shell command
# out_file: output file to write the stdout
# err_file: output file to write the stderr
# command: command to execute
# cwd: working directory
def execute_cmd(log_file, err_log_file, command, cwd='./'):
    print(f'In {os.path.abspath(cwd)}')
    print(f'Command: {command}')
    result = subprocess.run(f'{command}', shell=True,
                            cwd=cwd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    with open(log_file, 'a') as f:
        f.write(f'Command: {command}')
        f.write(result.stdout.decode("utf8"))
    with open(err_log_file, 'a') as f:
        f.write(result.stderr.decode("utf8"))

    if result.returncode == 0:
        print("Command executed successfully.\n")
    else:
        print(f"Error executing command. See {err_log_file}.\n")
        exit(1)


def parse_options():
    parser = argparse.ArgumentParser(
        description='Install third-party libraries (TPLs) for'
                    'running HPC-scale clustering')

    # Install destination directory
    parser.add_argument('-d', '--destination',
                        dest='destination',
                        required=False, action='store', type=str,
                        default='./tpls',
                        help='Destination directory to install TPLs')

    # #of parallel jobs for make
    parser.add_argument('-j', '--jobs',
                        dest='jobs',
                        required=False, action='store', type=int,
                        default=1,
                        help='#of parallel jobs for make')

    options = parser.parse_args()
    return options


def main():

    # Parse the command line arguments
    opts = parse_options()
    install_dir = os.path.abspath(opts.destination)
    print(f"Install destination directory: {install_dir}\n")

    # get timestamp to use it log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    out_log = f'out_install_tpls_{timestamp}.log'
    with open(out_log, 'w') as f:
        pass
    print(f"Log file: {out_log}")

    err_log = f'err_install_tpls_{timestamp}.log'
    with open(err_log, 'w') as f:
        pass
    print(f"Error file: {err_log}\n")

    # Check if the destination directory exists
    if os.path.exists(install_dir):
        # Ask the user if it is okay to delete the directory
        response = input(
            f"Directory {install_dir} exists. Do you want to delete it? (y/n): ")
        if response.lower() == 'y':
            print(f"Deleting directory {install_dir}")
            execute_cmd(out_log, err_log, f'rm -rf {install_dir}')
            execute_cmd(out_log, err_log, f'mkdir -p {install_dir}')
        # else:
        #     print("Exiting...")
        #     exit(1)
    else:
        # Create the destination directory
        execute_cmd(out_log, err_log, f'mkdir -p {install_dir}')

    # Partition comparison with YGM
    print('Cloning and building YGM-based partition-comparison...')
    execute_cmd(out_log, err_log,
                'git clone https://github.com/llnl/clams-cc.git', cwd=f'{install_dir}')
    execute_cmd(out_log, err_log,
                f'mkdir -p {install_dir}/clams-cc/build')
    execute_cmd(out_log, err_log, 'cmake -DCMAKE_BUILD_TYPE=Release ..',
                cwd=f'{install_dir}/clams-cc/build')
    execute_cmd(out_log, err_log, f'make -j {opts.jobs}',
                cwd=f'{install_dir}/clams-cc/build')

    print("Installation complete.")


if __name__ == "__main__":
    main()
