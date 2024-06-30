//******************************************************************************
// system includes
//******************************************************************************

// output
#include <stdio.h>
#include <iostream>
// option processing
#include <unistd.h>
#include <stdlib.h>
#include <chrono>


//******************************************************************************
// local includes
//******************************************************************************

#include "scan.hpp"
#include "alloc.hpp"

using namespace std::chrono;


//******************************************************************************
// macros 
//******************************************************************************

#define DEFAULT_N 100
#define H 2

#define NBLOCKS(n, block_size) ((n + block_size - 1) / block_size)

//******************************************************************************
// global variables 
//******************************************************************************

int debug_dump = 0;
int verification = 0;


//******************************************************************************
// compute a random vector of integers on the host, with about 1/H of the values
// negative.  negative values represent holes.
//******************************************************************************

void init_input(long *array, long n)
{
  for (long i = 0; i < n; i++) {
    int value = rand() % 10000;  
    if (value % H == 0) value = -value;
    array[i] = value;
  }
}


//******************************************************************************
// functions to dump a host array
//******************************************************************************

void dump(const char *what, long *array, long n)
{
  printf("*** %s ***\n", what);
  if (debug_dump == 0) {
    printf("<<output omitted: -d not specified>>\n\n");
    return;
  }
  for (long i = 0; i < n; i++) {
    printf("%5d ", array[i]);
  }
  printf("\n\n");
}


//******************************************************************************
// functions to dump a device array
//******************************************************************************

void dump_device(const char *what, long *d_array, long n)
{
  HOST_ALLOCATE(h_temp, n, long);
  cudaMemcpy(h_temp, d_array, n * sizeof(long), cudaMemcpyDeviceToHost);
  dump(what, h_temp, n);
  HOST_FREE(h_temp);
}
 

//******************************************************************************
// verification code
//******************************************************************************

// a comparison function for qsort to sort values into descending order 
int less(const void *left, const void *right)
{
  int *ileft = (int *) left; 
  int *iright = (int *) right; 
  return *iright - *ileft;
}


// sort a sequence of values in descending order, which puts all of the 
// negative values at the end
void sort(long *h_in, long n)
{
  qsort(h_in,  n, sizeof(long), less);
}


// verify that h_out contains the positive values of h_in
void verify(long *h_in, long *h_out, long n, long n_holes)
{
  if (verification == 0) {
    printf("<<verification omitted: -v not specified>>\n\n");
    return;
  }
  long non_holes = n - n_holes;
  long verified = 0;

  // sort h_in in descending order so that the positive values are at the front
  sort(h_in, n);

  // sort h_out in descending order
  sort(h_out, non_holes);

  // if any value of h_out is not equal to the corresponding value in h_in, the
  // hole compaction is wrong
  for (long i = 0; i < non_holes; i++) {
    if (h_in[i] != h_out[i]) {
      printf("verification failed: h_out[%d] (%d) != h_in[%d] (%d)\n", i, h_out[i], i, h_in[i]);  
    } else {
      verified++;
    }
  }
  if (verified == non_holes) printf("verification succeeded; %d non holes!\n", non_holes);
  printf("\n");
}
     

//******************************************************************************
// data-parallel code
//******************************************************************************

// example cuda function to add 1 to a vector
__global__ void
add_one(int *d_out, int *d_in, long n)
{
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_out[i] = d_in[i] + 1;
}


// example cuda function to set elements of a vector to 1
__global__ void
set_one(int *d_inout, long n)
{
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_inout[i] = 1;
}


__global__ void identify_holes(long *input, long *hole_flags, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hole_flags[idx] = input[idx] >= 0 ? 1 : 0; //(input[idx] < 0) ? 1 : 0;
    }
}

// __global__ void backfill_holes(int *input, int *output, int *hole_indices, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n && input[idx] >= 0) {
//         int new_idx = indices[idx];
//         output[new_idx] = input[idx];
//     }
// }
__global__ void backfill_holes(long *input, long *output, long *hole_indices, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && input[idx] >= 0) {
        // Moving non-negative number to new position according to prefix sum scan (hole indices) results
        long new_position = hole_indices[idx];
        output[new_position] = input[idx];
    }

    if (idx < n && input[idx] < 0) {
        long last_pos = n - 1;
        while (last_pos > idx && input[last_pos] < 0) {
            last_pos--;  // decrementing to find the last non-negative number
        }
        if (last_pos > idx && input[last_pos] >= 0) {
            output[idx] = input[last_pos];  // Placing the last non-negative in the hole position
            output[last_pos] = input[idx];   // placing the hole at the last non-negative's original position
        }
    }
}

//scan function
__global__ void exclusive_scan_in(long *output, long *input, long n) {

    extern __shared__ long temp[];  // Temporary array for scanning
    long thid = threadIdx.x;
    long pout = 0, pin = 1;

    // Load input array into shared memory.
    // Exclusive scan will need to shift right by one index, hence setting temp[0] to 0
    temp[pout * n + thid] = (thid > 0) ? input[thid - 1] : 0;
    __syncthreads();

    for (long offset = 1; offset < n; offset *= 2) {
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


//******************************************************************************
// a serial hole filling algorithm
//******************************************************************************

long
serial_fill_holes(long *output, long *input, long n)
{
  long n_holes = 0;
  
  long right = n - 1;   // right cursor in the input vector
  long left = 0;        // left cursor in the input and output vectors
  for (; left <= right; left++) {
    if (input[left] >= 0) {
      output[left] = input[left];
    } else {
      n_holes++; // count a hole in the prefix that needs filling
      while (right > left && input[right] < 0) {
        right--; n_holes++; // count a hole in the suffix backfill
      }
      if (right <= left) break; 
      output[left] = input[right--]; // fill a hole at the left cursor
    }
  }

  return n_holes;
}


long count_holes(long *in, long n)
{
  long n_holes = 0;
  for (long i = 0; i < n; i++) {
    if (in[i] < 0) n_holes++;
  }
  return n_holes;
} 

//******************************************************************************
// argument processing
//******************************************************************************

void getargs(int argc, char **argv, long *N)
{
  int opt;

  while ((opt = getopt(argc, argv, "dvn:")) != -1) {
    switch (opt) {
    case 'd':
      debug_dump = 1;
      break;

    case 'v':
      verification = 1;
      break;

    case 'n':
      *N = atol(optarg);
      break;

    default: /* '?' */
      fprintf(stderr, "Usage: %s [-n size] [-d] [-v]\n", argv[0]);
      exit(-1);
    }
  }
}

//******************************************************************************
//main function
//******************************************************************************

int main(int argc, char **argv)
{
  long N = DEFAULT_N;
  long n_holes;

  getargs(argc, argv, &N);

  printf("fill: N = %ld, debug_dump = %d\n", N, debug_dump);

  // long Nby10 = N / 10;

  HOST_ALLOCATE(h_input, N, long);
  HOST_ALLOCATE(h_output_serial, N, long);
  HOST_ALLOCATE(h_output_parallel, N, long);


  DEVICE_ALLOCATE(d_input, N, long);
  DEVICE_ALLOCATE(d_output, N, long);
  DEVICE_ALLOCATE(d_flags, N, long);
  DEVICE_ALLOCATE(hole_indices, N, long);

  //serial code
  init_input(h_input, N);


  HOST_ALLOCATE(h_input_copy, N, long);
  memcpy(h_input_copy, h_input, N * sizeof(long));

  dump("input", h_input, N);

  printf("count_holes returns n_holes = %ld\n\n", count_holes(h_input, N));

  auto start_cpu = high_resolution_clock::now();

  n_holes = serial_fill_holes(h_output_serial, h_input, N);

  auto stop_cpu = high_resolution_clock::now();
  auto duration_cpu = duration_cast<milliseconds>(stop_cpu - start_cpu);

  std::cout << "Time for serial compaction on 10 percent of N: " << duration_cpu.count() << " ms\n";

  printf("serial_fill_holes returns n_holes = %ld\n\n", n_holes);

  dump("h_output after serial_fill_holes", h_output_serial, N - n_holes);

  verify(h_input, h_output_serial, N, n_holes);

  //parallel code.

  //*********************************************************************
  // copy the input data to the GPU
  //*********************************************************************
  cudaMemcpy(d_input, h_input_copy, N * sizeof(long), cudaMemcpyHostToDevice);

  long threadsPerBlock = 256;
  long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  auto start_gpu = high_resolution_clock::now();

  identify_holes<<<blocks, threadsPerBlock>>>(d_input, d_flags, N);
  cudaDeviceSynchronize();

  dump_device("d_flags after marking non neg of d_input" , d_flags, N);

  auto scan_time_start = high_resolution_clock::now(); //time for scan
  long shared_size = N * sizeof(long);
  exclusive_scan_in<<<1, N, shared_size>>>(hole_indices, d_flags, N);
  cudaDeviceSynchronize();

  auto scan_time_stop = high_resolution_clock::now(); //stop time scan
  auto duration_scan = duration_cast<milliseconds>(scan_time_stop - scan_time_start);
  // ex_sum_scan(hole_indices, d_flags, N);
  std::cout << "Time for scan: " << duration_scan.count() << " ms\n";

  dump_device("hole_indices after ex sum scan d_flags" , hole_indices, N);

  backfill_holes<<<blocks, threadsPerBlock>>>(d_input, d_output, hole_indices, N);
  cudaDeviceSynchronize();

  dump_device("d_output after backfill_holes hole_indices" , d_output, N);
  auto stop_gpu = high_resolution_clock::now();
  auto duration_gpu = duration_cast<milliseconds>(stop_gpu - start_gpu);

  //*********************************************************************
  // copy the output data from the GPU
  //*********************************************************************
  cudaMemcpy(h_output_parallel, d_output, N * sizeof(long), cudaMemcpyDeviceToHost);

  long count;
  for (int i = 0; i < N; ++i) {
     if (h_output_parallel[i] >= 0) count++;
  }
  verify(h_input_copy, h_output_parallel, N, N - count);

  std::cout << "Time for parallel compaction on full N: " << duration_gpu.count() << " ms\n";

  // dump("h_output on host", h_output_parallel, N);

  HOST_FREE(h_input);
  HOST_FREE(h_output_serial);
  HOST_FREE(h_input_copy);
  HOST_FREE(h_output_parallel);
  
  DEVICE_FREE(d_input);
  DEVICE_FREE(d_output);
  DEVICE_FREE(d_flags);
  DEVICE_FREE(hole_indices);
  return 0;
}
