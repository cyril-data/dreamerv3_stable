#!/bin/bash
# wrapper.sh

CONFIG_NAME=$(basename $2 .yml)
sbatch --job-name=$CONFIG_NAME --output=logs/$CONFIG_NAME.err --error=logs/$CONFIG_NAME.err job.slurm $1 $2
