//******************************************************************************
// cuda includes
//******************************************************************************

#include <cub/cub.cuh>


//******************************************************************************
// local includes
//******************************************************************************

#include "scan.hpp"


//******************************************************************************
// interface operations
//******************************************************************************

// compute an exclusive sum scan of d_in in O(log_2 n) steps
void ex_sum_scan
(
 int *d_out, // pointer to an output device array with space for n integers 
 int *d_in,  // pointer to an input device array containing n integers
 int n       // n - the number of elements in d_in and d_out
)
{
  void     *d_tmp = NULL; // pointer to temporary storage used by the the scan 
  size_t   tmp_bytes = 0; // the number of bytes of temporary storage needed

  // determine how many bytes of temporary storage are needed
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

  // allocate temporary storage
  cudaMalloc(&d_tmp, tmp_bytes);

  // compute the exclusive prefix sum of d_in into d_out using d_tmp as 
  // temporary storage
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_in, d_out, n);

  // feee the temporary storage
  cudaFree(d_tmp);
}