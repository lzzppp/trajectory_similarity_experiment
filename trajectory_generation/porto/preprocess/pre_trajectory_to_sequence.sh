#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=50
#SBATCH --time=10:00:00
#SBATCH --job-name=Pre_Trajectory2Sequence
python map_trajectory_to_sequence.py