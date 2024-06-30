#include <stdio.h>
#include <iostream>
// option processing
#include <unistd.h>
#include <stdlib.h>
#include <chrono>
#include <cub/cub.cuh>


//******************************************************************************
// local includes
//******************************************************************************

#include "scan.hpp"
#include "alloc.hpp"

__global__ void exclusive_scan_in(int *output, int *input, int n) {

    extern __shared__ int temp[];  // Temporary array for scanning
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // Load input array into shared memory.
    // Exclusive scan will need to shift right by one index, hence setting temp[0] to 0
    temp[pout * n + thid] = (thid > 0) ? input[thid - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; 
        pin = 1 - pout;
        if (thid >= offset)
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
        else
            temp[pout * n + thid] = temp[pin * n + thid];
        __syncthreads();
    }

    output[thid] = temp[pout * n + thid];  // Write output
}


int main() {
    const int N = 1 << 20;  // Example size of 1M elements
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Initialize input with some values
    int *h_input = new int[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = i % 100;  // Arbitrary example data
    }
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Prepare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time CUB's exclusive scan
    float milliseconds = 0;
    cudaEventRecord(start);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes); 
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUB Exclusive Sum Time: " << milliseconds << " ms" << std::endl;
    cudaFree(d_temp_storage);

    // Time your custom exclusive scan
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int sharedSize = blockSize * sizeof(int);
    cudaEventRecord(start);
    exclusive_scan_in<<<numBlocks, blockSize, sharedSize>>>(d_output, d_input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Custom Exclusive Scan Time: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}