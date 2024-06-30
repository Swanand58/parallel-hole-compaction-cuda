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

void init_input(int *array, long n)
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

void dump(const char *what, int *array, long n)
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

void dump_device(const char *what, int *d_array, long n)
{
  HOST_ALLOCATE(h_temp, n, int);
  cudaMemcpy(h_temp, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
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
void sort(int *h_in, long n)
{
  qsort(h_in,  n, sizeof(int), less);
}


// verify that h_out contains the positive values of h_in
void verify(int *h_in, int *h_out, long n, long n_holes)
{
  if (verification == 0) {
    printf("<<verification omitted: -v not specified>>\n\n");
    return;
  }
  long non_holes = n - n_holes;
  int verified = 0;

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


__global__ void mark_non_negatives(int *input, int *marks, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        marks[idx] = input[idx] >= 0 ? 1 : 0;
    }
}

__global__ void compact_array(int *input, int *output, int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && input[idx] >= 0) {
        int new_idx = indices[idx];
        output[new_idx] = input[idx];
    }
}

//scan
__global__ void exclusive_scan_in(int *output, int *input, int n) {
    extern __shared__ int temp[];  // Temporary storage for inclusive scan
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    // Load input into shared memory.
    // Exclusive scan needs to shift right by one, hence setting temp[0] to 0
    temp[pout * n + thid] = (thid > 0) ? input[thid - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
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
serial_fill_holes(int *output, int *input, long n)
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


long count_holes(int *in, long n)
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



// int main(int argc, char **argv)
// {
//   long N = DEFAULT_N;
//   long n_holes;

//   getargs(argc, argv, &N);

//   printf("fill: N = %ld, debug_dump = %d\n", N, debug_dump);

//   HOST_ALLOCATE(h_input, N, int);
//   HOST_ALLOCATE(h_output, N, int);
//   DEVICE_ALLOCATE(d_input, N, int);
//   DEVICE_ALLOCATE(d_output, N, int);
//   DEVICE_ALLOCATE(d_tmp, N, int);

//   init_input(h_input, N);

//   dump("input", h_input, N);

//   printf("count_holes returns n_holes = %ld\n\n", count_holes(h_input, N));

//   n_holes = serial_fill_holes(h_output, h_input, N);

//   printf("serial_fill_holes returns n_holes = %ld\n\n", n_holes);

//   dump("h_output after serial_fill_holes", h_output, N - n_holes);

//   verify(h_input, h_output, N, n_holes);

//   //*********************************************************************
//   // copy the input data to the GPU
//   //*********************************************************************
//   cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

//   //*********************************************************************
//   // some example data parallel operations on data in GPU memory
//   //*********************************************************************

//   // set each element of d_tmp to 1 + the the value of the 
//   // corresponding value of d_input
//   add_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, d_input, N);

//   dump_device("d_tmp after add_one", d_tmp, N);

//   // increment d_tmp in place
//   add_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, d_tmp, N);

//   dump_device("d_tmp after second add_one", d_tmp, N);

//   // set the values in d_tmp
//   set_one<<<NBLOCKS(N, 1024), 1024>>>(d_tmp, N);

//   dump_device("d_tmp after set_one" , d_tmp, N);

//   // ex_sum_scan(d_output, d_tmp, N);
//   int shared_size = N * sizeof(int); // Allocate enough shared memory for n elements
//   exclusive_scan<<<1, N, shared_size>>>(d_output, d_tmp, N);
//   cudaDeviceSynchronize();

//   dump_device("d_output after exscan of d_tmp" , d_output, N);

//   //*********************************************************************
//   // copy the output data from the GPU
//   //*********************************************************************
//   cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

//   dump("h_output on host", h_output, N);

//   HOST_FREE(h_input);
//   HOST_FREE(h_output);

//   DEVICE_FREE(d_input);
//   DEVICE_FREE(d_output);
//   DEVICE_FREE(d_tmp);
// }

__global__ void mark_holes_and_non_holes(int *input, int *is_hole, int *is_non_hole, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (input[idx] < 0) {
            is_hole[idx] = 1;
            is_non_hole[idx] = 0;
        } else {
            is_hole[idx] = 0;
            is_non_hole[idx] = 1;
        }
    }
}

__global__ void place_non_holes(int *input, int *output, int *non_hole_positions, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && input[idx] >= 0) {
        int pos = non_hole_positions[idx];
        output[pos] = input[idx];
    }
}

__global__ void backfill_holes(int *input, int *output, int *hole_positions, int *non_hole_positions, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && input[idx] < 0) {
        int pos = hole_positions[idx];  // Position to start backfill
        int fill_pos = non_hole_positions[n - 1] - (pos - non_hole_positions[n - 1]);  // Mirror position from the end
        output[idx] = input[fill_pos];
    }
}



int main(int argc, char **argv) 
{
    int h_input[] = {9383, -886, 2777, 6915, 7793, 8335, -5386, -492, 6649, 1421, -3992, 20913, 23981, -2903};

    std::cout << "This is my input::";
    for (int i = 0; i < 14; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    int n = sizeof(h_input) / sizeof(h_input[0]);
    int* d_input, *d_output; //*d_flags, *d_positions;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    // cudaMalloc(&d_flags, n * sizeof(int));
    // cudaMalloc(&d_positions, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_is_hole, *d_is_non_hole, *d_non_hole_pos, *d_hole_pos;
    cudaMalloc(&d_is_hole, n * sizeof(int));
    cudaMalloc(&d_is_non_hole, n * sizeof(int));
    cudaMalloc(&d_non_hole_pos, n * sizeof(int));
    cudaMalloc(&d_hole_pos, n * sizeof(int));

    int threadsPerBlock = 1;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    //mark_non_negatives<<<blocks, threadsPerBlock>>>(d_input, d_flags, n);
    mark_holes_and_non_holes<<<blocks, threadsPerBlock>>>(d_input, d_is_hole, d_is_non_hole, n);
    //cudaDeviceSynchronize();

    int shared_size = n * sizeof(int); // Allocate enough shared memory for n elements
    exclusive_scan_in<<<1, n, shared_size>>>(d_positions, d_flags, n);
    cudaDeviceSynchronize();

    compact_array<<<blocks, threadsPerBlock>>>(d_input, d_output, d_positions, n);
    cudaDeviceSynchronize();

    place_non_holes<<<blocks, threadsPerBlock>>>(d_input, d_output, d_non_hole_pos, n);
    backfill_holes<<<blocks, threadsPerBlock>>>(d_input, d_output, d_hole_pos, d_non_hole_pos, n);

    // int lastFlag, lastPosition;
    // cudaMemcpy(&lastFlag, &d_flags[n-1], sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&lastPosition, &d_positions[n-1], sizeof(int), cudaMemcpyDeviceToHost);
    // int n_non_negatives = lastPosition + lastFlag;

    int* h_output = new int[n];
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost); //add n_non_neg instead of n

    std::cout << "This is my output::";
    for (int i = 0; i < n; i++) {  //n_non_negatives
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    // cudaFree(d_flags);
    // cudaFree(d_positions);
    delete[] h_output;

    return 0;
}
