#!/bin/bash
#SBATCH --export=ALL
#SBATCH --nodes=1 
#SBATCH --time=00:30:00 
#SBATCH --cpus-per-task=8 
#SBATCH --reservation=comp422
#SBATCH --gres=gpu:volta:1

time ./fill-opt -n 400000000

# time ./test-opt

# time ./testscan.cu