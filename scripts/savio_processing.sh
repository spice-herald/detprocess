#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=account_name
#
# Partition:
#SBATCH --partition=partition_name
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=24
#
# Wall clock limit:
#SBATCH --time=00:30:00
#
## Command(s) to run:
module load python
ipcluster start -n $SLURM_NTASKS &    # Start worker processes
sleep 120                             # Wait until all worker processes have successfully started
ipython process_parallel.py > process.pyout
ipcluster stop
