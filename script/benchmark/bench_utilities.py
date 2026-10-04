# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

import subprocess
import os

from script.utilities import *

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
