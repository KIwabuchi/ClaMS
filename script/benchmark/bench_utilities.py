# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

import os


def set_up_batch_header(script_file, job_name, job_dir, num_nodes,
                        num_tasks_per_node):
    # Make dir
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    script_file.write("#!/bin/bash\n")
    script_file.write(f"#SBATCH --job-name={job_name}\n")
    script_file.write(f"#SBATCH --nodes={num_nodes}\n")
    script_file.write(f"#SBATCH --ntasks-per-node={num_tasks_per_node}\n")

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


def add_msg(msg, script_file):
    script_file.write(f"echo \"{msg}\"\n")


def add_parameter_msg(parameter, value, script_file):
    msg = f"Parameter {parameter}: {value}"
    add_msg(msg, script_file)


def add_srun_cmd(num_nodes, num_tasks_per_node, command, script_file, echo=True,
                 check_return_code=True):
    num_tasks = num_nodes * num_tasks_per_node
    add_cmd(f"srun --nodes={num_nodes} --ntasks={num_tasks} "
            f"--ntasks-per-node={num_tasks_per_node} {command}",
            script_file, echo, check_return_code)
