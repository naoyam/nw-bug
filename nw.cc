#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <assert.h>

#include "opencl_util.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

//global variables

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// local variables
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

static int initialize() {
  size_t size;
  cl_int result, error;
  cl_uint platformCount;
  cl_platform_id* platforms = NULL;
  cl_context_properties ctxprop[3];

  display_device_info(&platforms, &platformCount);
#ifdef ALTERA_CL
  device_type = CL_DEVICE_TYPE_ACCELERATOR;
#else
  select_device_type(platforms, &platformCount, &device_type);
#endif
  validate_selection(platforms, &platformCount, ctxprop, &device_type);
	
  // create OpenCL context
  context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, &error );
  if( !context )
  {
    printf("ERROR: clCreateContextFromType(%s) failed with error code %d.\n", (device_type == CL_DEVICE_TYPE_ACCELERATOR) ? "FPGA" : (device_type == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", error);
    display_error_message(error, stdout);
    return -1;
  }

  // get the list of GPUs
  result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
  num_devices = (int) (size / sizeof(cl_device_id));
	
  if( result != CL_SUCCESS || num_devices < 1 )
  {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }
  device_list = new cl_device_id[num_devices];
  if( !device_list )
  {
    printf("ERROR: new cl_device_id[] failed\n");
    return -1;
  }
  result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
  if( result != CL_SUCCESS )
  {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }

  // create command queue for the first device
  cmd_queue = clCreateCommandQueue( context, device_list[0], CL_QUEUE_PROFILING_ENABLE, NULL );
  if( !cmd_queue )
  {
    printf("ERROR: clCreateCommandQueue() failed\n");
    return -1;
  }
	
  free(platforms); // platforms isn't needed in the main function

  return 0;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s [N]\n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  exit(1);
}

void init_input_matrices(int *reference, int *input_itemsets,
                         int N, int penalty) {
  ++N;
  //initialization 
  for (int i = 0 ; i < N; i++){
    for (int j = 0 ; j < N; j++){
      input_itemsets[i*N+j] = 0;
    }
  }

  srand(7);
  for( int i=1; i< N ; i++){
    input_itemsets[i*N] = rand() % 10 + 1;
  }
  for( int j=1; j< N ; j++){
    input_itemsets[j] = rand() % 10 + 1;
  }
  
  for (int i = 1 ; i < N; i++){
    for (int j = 1 ; j < N; j++){
      reference[i*(N)+j] = blosum62[input_itemsets[i*(N)]][input_itemsets[j]];      
    }
  }

  for( int i = 1; i< N; i++)
    input_itemsets[i*N] = -i * penalty;
  for( int j = 1; j< N ; j++)
    input_itemsets[j] = -j * penalty;
  
}

static int maximum(int a, int b, int c) {
  int k = a >= b ? a : b;
  k = k >= c ? k : c;
  return k;
}

static void nw_ref(int *reference,
                   int *input_itemsets,
                   int N, int penalty) {
  ++N;
  for (int j = 1; j < N; ++j) {
    for (int i = 1; i < N; ++i) {
      int index = i + j * N;
      int diag = input_itemsets[index - N - 1];
      int left = input_itemsets[index - 1];
      int above = input_itemsets[index - N];
      int v = maximum(diag + reference[index],
                      left - penalty,
                      above - penalty);
      input_itemsets[index] = v;
    }
  }
}

int main(int argc, char **argv) {
  int N = 8;
  int penalty = 10;
  
  if (argc == 2) {
    N = atoi(argv[1]);
  } 
  assert((N % BLOCK_SIZE) == 0);
  int NP = N+1;

  int *reference = (int *)malloc(NP * NP * sizeof(int) );
  int *input_itemsets = (int *)malloc(NP * NP * sizeof(int) );
  int *output_itemsets = (int *)malloc(NP * NP * sizeof(int) );

  init_input_matrices(reference, input_itemsets, N, penalty);

#ifdef ALTERA_CL
  std::string kernel_file_path = "nw_kernel.aocx";
#else
  std::string kernel_file_path = "nw_kernel.cl";
#endif
  printf("Using kernel file: %s\n", kernel_file_path.c_str());
  printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
  
  size_t sourcesize;
  char *source = read_kernel(kernel_file_path.c_str(), &sourcesize);
  // read the kernel core source
  char const * kernel_nw1  = "nw_kernel1";

  // OpenCL initialization
  if(initialize()) {
    return -1;
  }


  // compile kernel
  cl_int err = 0;
#ifndef ALTERA_CL
  const char * slist[2] = { source, 0 };
  cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
#else
  cl_program prog = clCreateProgramWithBinary(context, 1, device_list,
                                              &sourcesize, (const unsigned char**)&source, NULL, &err);
#endif
  if(err != CL_SUCCESS) {
    printf("ERROR: clCreateProgramWithSource/Binary() => %d\n", err);
    display_error_message(err, stderr);
    return -1;
  }

  char clOptions[110];
  sprintf(clOptions, "-I .");

#if defined(BLOCK_SIZE) && !defined(ALTERA_CL)
  sprintf(clOptions + strlen(clOptions),
          " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif
        
  clBuildProgram_SAFE(prog, num_devices, device_list, clOptions, NULL, NULL);

  cl_kernel kernel1;
  kernel1 = clCreateKernel(prog, kernel_nw1, &err);
  if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	
  // creat buffers
  cl_mem input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           NP * NP * sizeof(int), NULL, &err );
  if(err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", NP*NP, err);
    exit(1);
  }
#if 0  
  cl_mem output_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            NP * NP * sizeof(int), NULL, &err );
  if(err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", NP*NP, err);
    exit(1);
  }
#endif  
  cl_mem reference_d	 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          NP * NP * sizeof(int), NULL, &err );
  if(err != CL_SUCCESS) {
    fprintf(stderr, "ERROR: clCreateBuffer reference (size:%d) => %d\n", NP*NP, err);
    exit(1);
  }
	
  //write buffers
  CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue,
                                    input_itemsets_d, 1,
                                    0, NP * NP * sizeof(int),
                                    input_itemsets, 0, 0, 0));
#if 0  
  CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue,
                                    output_itemsets_d, 1,
                                    0, NP * NP * sizeof(int),
                                    input_itemsets, 0, 0, 0));
#endif  
  CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue,
                                    reference_d, 1, 0,
                                    NP * NP * sizeof(int),
                                    reference, 0, 0, 0));
        
  clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &reference_d);
  clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &input_itemsets_d);
  clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &input_itemsets_d);
  clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &N);
  clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &penalty);


  printf("==============================================================\n");
  printf("Start computation\n");  

  int nb = N  / BLOCK_SIZE;
  for (int by = 0; by < nb; ++by) {
    for (int bx = 0; bx < nb; ++bx) {
      clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*) &bx);
      clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &by);
      CL_SAFE_CALL(clEnqueueTask(cmd_queue, kernel1, 0, NULL, NULL));
    }
  }
  clFinish(cmd_queue);

  err = clEnqueueReadBuffer(cmd_queue, input_itemsets_d, 1, 0,
                            NP * NP * sizeof(int), output_itemsets, 0, 0, 0);
  clFinish(cmd_queue);

  printf("Computation done\n");
  printf("==============================================================\n");
  printf("Validating result\n");
  int *cpu_result = (int*)malloc(NP * NP * sizeof(int));
  memcpy(cpu_result, input_itemsets, NP * NP * sizeof(int));
  nw_ref(reference, cpu_result, N, penalty);
  int num_errors = 0;
  for (int row = 0; row < NP; ++row) {
    for (int col = 0; col < NP; ++col) {
      int index = col+row*NP;
      if (cpu_result[index] != output_itemsets[index]) {
        printf("Error detected at (%d, %d). CPU: %d, FPGA: %d\n",
               row, col, cpu_result[index], output_itemsets[index]);
        ++num_errors;
      }
    }
  }

  if (num_errors == 0) {
    printf("Successfully validated.\n");
  } else {
    printf("%d errors detected.\n", num_errors);
  }
  
  FILE *fout = fopen("output_itemsets.txt","w");
  for (int j = 0; j < NP; ++j) {
    fprintf(fout, "%3d", output_itemsets[j*NP]);    
    for (int i = 1; i < NP; ++i) {
      if (j == 0) {
        fprintf(fout, ", %8d", output_itemsets[j*NP + i]);
      } else {
        fprintf(fout, ", %3d", output_itemsets[j*NP + i]);        
      }
      if (i != 0 && j != 0) {
        fprintf(fout, " (%2d)", reference[j*NP+i]);
      }
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
  printf("Output itemsets saved in output_itemsets.txt\n");
}

