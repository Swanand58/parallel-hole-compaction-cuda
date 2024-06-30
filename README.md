# Parallel Hole Compaction CUDA

This repository contains an implementation of parallel hole compaction using CUDA, a parallel computing platform and application programming interface model created by NVIDIA.

## Overview

In GPU-accelerated simulations of plasma circulating in a tokamak, each GPU is responsible for simulating a specific region. As the plasma circulates, hydrogen nuclei may move between regions, leaving empty slots or "holes" in the data array. This project implements a data-parallel algorithm in CUDA to perform hole compaction, reorganizing the data by filling these holes with non-negative elements from the end of the array.

## Assignment Requirements

The project simulates a scenario similar to plasma turbulence in tokamaks, using random numbers where negative values represent holes (nuclei that have moved). The solution follows these specific steps:

1. **Identify Holes**: Identify negative numbers in the sequence.
2. **Accumulate Hole Indices**: Use prefix sum operations to gather indices of the holes.
3. **Backfill Holes**: Fill the holes with non-negative numbers from the end of the sequence without moving positive numbers within the first \(N-H\) elements, where \(N\) is the total number of elements and \(H\) is the number of holes.

## Features

- Efficient parallel processing using CUDA
- Implementation of data-parallel operations to handle hole compaction
- Use of exclusive sum scan (prefix sum) operations

## Requirements

- NVIDIA CUDA Toolkit
- Compatible GPU with CUDA support
- C++ compiler (e.g., GCC)

## Installation

Clone the repository:

```bash
git clone https://github.com/Swanand58/parallel-hole-compaction-cuda.git
cd parallel-hole-compaction-cuda
```

## Compile the code using the following command:

```bash
nvcc -o hole_compaction hole_compaction.cu
```

# Usage

## Run the compiled program with:

```bash
./hole_compaction
```

## Parameters

- Input data: The program generates a sequence of random numbers where negative numbers represent holes.
- Output data: The output will be the original sequence with holes compacted.

# Algorithm

1. Identification of Holes:

- Negative numbers in the array are identified as holes.

2. Prefix Sum (Exclusive Sum Scan):

- An exclusive prefix sum operation is used to accumulate the indices of the holes.

3. Backfilling:

- The holes are backfilled using non-negative numbers from the end of the array, skipping over holes encountered.

4. Validation:

- The output is validated to ensure it matches the input sequence with the holes removed.

## Acknowledgments

- NVIDIA for providing the CUDA toolkit.
- Course materials on high-performance computing and plasma simulation.
- References on prefix sum operations and their implementation on GPUs.
