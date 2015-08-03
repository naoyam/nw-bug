#ifndef BLOCK_SIZE
#define BLOCK_SIZE (4)
#endif

int maximum(int a, int b, int c) {
  int k = a >= b ? a : b;
  k = k >= c ? k : c;
  return k;
}

/*
  Computes a BLOCK_SIZExBLOCKSIZE sub-matrix of 2D (N+1)^2 matrix. The
  top-most row and left-most column are not necessary to compute, so
  the starting offset is (1, 1).
  
  input_itemsets: (N+1)^2 integer matrix
  output_itemsets: (N+1)^2 integer matrix
  block_col_idx: horizontal index of the block
  block_row_idx: vertical index of the block
*/

/*
  Here's a straightforward nested-loop implementation with no
  blocking. This works correctly on de5net_a7.

__kernel void 
nw_kernel1(__global int* restrict reference, 
           __global int* restrict input_itemsets,
           int N,
           int penalty) 
{
  int NP = N+1;
  for (int j = 1; j < NP; ++j) {
    for (int i = 1; i < NP; ++i) {
      int index = j * NP + i;
      input_itemsets[index]= maximum(
          input_itemsets[index-1-NP]+ reference[index], 
          input_itemsets[index-1]         - penalty, 
          input_itemsets[index-NP]  - penalty);
    }
  }
}
*/

/*
  This version attempts to pipeline the outer loop by using a local
  array. It is a single-dimensional array with length BLOCK_SIZE,
  where BLOCK_SIZE is the 2D blocking dimension. This test program
  may use a very small size, but real problems would require blocking
  to keep the size of local array sufficiently small.

  The kernel is called N/BLOCK_SIZE*N/BLOCK_SIZE times. Each call sets
  the block_col_idx and block_row_idx parameters like (0, 0), (0, 1),
  (1, 0), (1, 1), etc.

  The above straightforward code has a read-after-write dependency
  across the iterations of the outer loop, so the compiler may not be
  able to pipeline the loop or even if it did, the initiation
  interval would be large. In the below version, it uses the local
  array (sr) to remove the read-after-write, so the compiler should be
  able to pipeline the outer loop with just one-cyle
  interval. However, the compiler of AOCL v15.0.2 in fact gets
  confused by load and store accesses to input_itemsets, and
  determines there is a dependency, which is in fact false. To
  work-around the false analysis, the below code uses two separate
  array parameters for load and store accesses and mark them as
  "restrict". This allows the compiler to ignore the false
  dependency and to pipeline the outer loop.

  The code is tested with Altera OpenCL version 15.0.2. Emulation
  works fine, but when executed on Terasic's Stratix 5 (de5net_a7), it
  always gives incorrect results.

  For example, when N=8 and BLOCK_SIZE=4, the first sub block at
  offset (0, 0) is fine, but the other sub blocks get incorrect
  results. For example, the coordinate at (1, 5), which is the first
  point computed by the sub block with block_col_idx being 1 and
  block_row_idx being 0, results in -20, while the correct value is
  -39. This error is then propagated to the other coordinates within
  the sub block and its dependent sub blocks. Similarly, the
  coordinate at (5, 1) also gives -20, while its correct value is
  -40. This error also affects other dependent computations.

  This works also with the AMD OpenCL SDK on CPU. Since it works with
  emulation, the cause of the error may be related to pipelining of
  the loop. 
 */
__kernel void 
nw_kernel1(__global int* restrict reference, 
           __global int* restrict input_itemsets,
           __global int* restrict output_itemsets,           
           int N,
           int penalty,
           int block_col_idx, int block_row_idx) 
{
  int NP = N + 1;
  int base = BLOCK_SIZE * block_col_idx + 1
      + NP * (BLOCK_SIZE * block_row_idx + 1);
  int sr[BLOCK_SIZE];
  
  #pragma unroll
  for (int col_idx = 0; col_idx < BLOCK_SIZE; ++col_idx) {
    sr[col_idx] = input_itemsets[base - NP + col_idx];
  }
  
  for (int row_idx = 0; row_idx < BLOCK_SIZE; ++row_idx) {
    int left = input_itemsets[base + NP * row_idx - 1];    
    int diag = input_itemsets[base + NP * row_idx - 1 - NP];

    #pragma unroll
    for (int col_idx = 0; col_idx < BLOCK_SIZE; ++col_idx) {
      int index = base + col_idx + NP * row_idx;
      int above = sr[col_idx];
      int v = maximum(diag + reference[index], 
                      left - penalty, above - penalty);
      diag = above;
      left = v;
      sr[col_idx] = v;
      
      // input_itemsets[index] = v;
      output_itemsets[index] = v;
    }
  }
}

