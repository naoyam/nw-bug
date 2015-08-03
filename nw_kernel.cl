#ifndef BLOCK_SIZE
#define BLOCK_SIZE (4)
#endif

int maximum(int a, int b, int c) {
  int k = a >= b ? a : b;
  k = k >= c ? k : c;
  return k;
}

__kernel void 
nw_kernel1(__global int* restrict reference, 
           __global int* restrict input_itemsets,
           __global int* restrict output_itemsets,           
           int N,
           int penalty,
           int bx, int by) 
{
  int base = BLOCK_SIZE * bx + 1 + N * (BLOCK_SIZE * by + 1);
  int sr[BLOCK_SIZE];
  
  #pragma unroll
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    sr[i] = input_itemsets[base - N + i];
  }
  
  for (int j = 0; j < BLOCK_SIZE; ++j) {
    int diag = input_itemsets[base + N * j - 1 - N];
    int left = input_itemsets[base + N * j - 1];

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      int index = base + i + N * j;
      int above = sr[i];
      int v = 
          maximum(
          diag + reference[index], 
          left - penalty,
          above - penalty);
      diag = above;
      left = v;
      sr[i] = v;
      output_itemsets[index] = v;
    }
  }
}

