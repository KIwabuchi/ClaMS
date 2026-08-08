# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


import subprocess
import os


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


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    # Check if the directory was created
    if not os.path.exists(dir_path):
        print(f"Error: Could not create directory {dir_path}")
        exit(1)


def set_up_batch_header(script_file, job_name, job_dir, num_nodes):
    # Make dir
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    script_file.write("#!/bin/bash\n")
    script_file.write(f"#SBATCH --job-name={job_name}\n")
    script_file.write(f"#SBATCH --nodes={num_nodes}\n")

    out_file = f"{job_dir}/out.log"
    script_file.write(f"#SBATCH --output={out_file}\n")
    err_file = f"{job_dir}/err.log"
    script_file.write(f"#SBATCH --error={err_file}\n\n")

    return out_file, err_file


# Function to write commands to the shell script file
def add_cmd(command, script_file, echo=True,
            check_return_code=True):
    if echo:
        script_file.write(f"echo \"Command: {command}\"\n")

    script_file.write(f"{command}\n")

    if check_return_code:
        script_file.write("\nif [ $? -ne 0 ]; then\n")
        script_file.write("  echo \"Error executing command.\"\n")
        script_file.write("  exit 1\n")
        script_file.write("fi\n\n")


def add_srun_cmd(num_tasks_per_node, command, script_file, echo=True,
                 check_return_code=True):
    add_cmd(f"srun --ntasks-per-node={num_tasks_per_node} {command}",
            script_file, echo, check_return_code)


def grep_file(file, text_pattern):
    # Return the lines that contain the pattern
    with open(file, 'r') as f:
        lines = f.readlines()
        return [line for line in lines if text_pattern in line]

# Parse a range string and return a list of integers
# e.g., "1-5" -> [1, 2, 3, 4, 5]
# e.g., "1,3,5" -> [1, 3, 5]
def parse_range(range_str):
    if range_str is None:
        return [None]

    if '-' in range_str:
        start, end = range_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(x) for x in range_str.split(',')]

# Parse a list of floats from a string and return a list of floats
# e.g., "1.0,2.5,3.14" -> [1.0, 2.5, 3.14]
def parse_float_list(float_list_str):
    if float_list_str is None:
        return [None]

    return [float(x) for x in float_list_str.split(',')]