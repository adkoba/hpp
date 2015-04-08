include <wb.h> //@@ wb include opencl.h for you

#define wbCheckCde(code) do {                                                    \
        if (code != CL_SUCCESS) {                                             \
            wbLog(ERROR, "Got OpenCL error ...  ", code);    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define wbCheck(stmt) do {                                                    \
        cl_int err = stmt;                                               \
        if (err != CL_SUCCESS) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got OpenCL error ...  ", err);    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

//@@ OpenCL Kernel
const char* gVectorAddSrc =
"__kernel void vadd(__global float *d_A, __global float *d_B, __global float *d_C, int N) { \
	int id = get_global_id(0); \
\
	if( id < N ) \
		d_C[id] = d_A[id] + d_B[id]; \
} ";

#define BLOCK_SIZE 32

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  
	cl_int clErr = CL_SUCCESS;
	
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

	// Get plarform id
	cl_platform_id clPlatform;
	cl_uint clNumPlatforms;
	wbCheck(clGetPlatformIDs( 1, &clPlatform, &clNumPlatforms));
	
	// Create an OpenCL context
	cl_context_properties clContextProp[3];
	clContextProp[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;  // indicates that next element is platform
	clContextProp[1] = (cl_context_properties)clPlatform;  // platform is of type cl_platform_id
	clContextProp[2] = (cl_context_properties)0;   // last element must be 0
	cl_context clCtx = clCreateContextFromType(clContextProp, CL_DEVICE_TYPE_ALL, NULL, NULL, &clErr);
	wbCheckCde(clErr);
	
	// Get context infos
	size_t clParamSize;
	wbCheck(clGetContextInfo(clCtx, CL_CONTEXT_DEVICES, 0, NULL, &clParamSize));
	cl_device_id* clDevs = (cl_device_id *) malloc(clParamSize);
	wbCheck(clGetContextInfo(clCtx, CL_CONTEXT_DEVICES, clParamSize, clDevs, NULL));
	
	// Create kernel pgm
	cl_program clPgm = clCreateProgramWithSource(clCtx, 1, &gVectorAddSrc, NULL, &clErr);
	wbCheckCde(clErr);
	
	// Create command queue
	cl_command_queue clCdeQueue = clCreateCommandQueue(clCtx, clDevs[0], 0, &clErr); 
	wbCheckCde(clErr);
	
	// Compile kernel pgm
	char clCompileFlags[4096];
	sprintf(clCompileFlags, "-cl-mad-enable");
	wbCheck(clBuildProgram(clPgm, 0, NULL, clCompileFlags, NULL, NULL));
	cl_kernel clKernel = clCreateKernel(clPgm, "vadd", &clErr);
	wbCheckCde(clErr);

  //@@ Allocate GPU memory here
	wbTime_start(GPU, "Allocating GPU memory and copying input memory to the GPU");
	cl_mem d_In1, d_In2, d_Out;
	int size = inputLength* sizeof(float);
	d_In1 = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput1, &clErr);
	wbCheckCde(clErr);
	d_In2 = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput2, &clErr);
	wbCheckCde(clErr);
	d_Out = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, size, NULL, &clErr);
	wbCheckCde(clErr);
  wbTime_stop(GPU, "Allocating GPU memory and copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
	size_t globalSize = (((inputLength-1)/BLOCK_SIZE)+1)*BLOCK_SIZE;
	size_t blockSize = BLOCK_SIZE;
	
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
	wbCheck(clSetKernelArg(clKernel, 0, sizeof(cl_mem),(void *)&d_In1));
	wbCheck(clSetKernelArg(clKernel, 1, sizeof(cl_mem),(void *)&d_In2));
	wbCheck(clSetKernelArg(clKernel, 2, sizeof(cl_mem),(void *)&d_Out));
	wbCheck(clSetKernelArg(clKernel, 3, sizeof(int), &inputLength));

	cl_event clEvent=NULL;
	
	wbLog(TRACE, "Launch kernel with GSZ ", globalSize, " BSZ ", blockSize);
	
	wbCheck(clEnqueueNDRangeKernel(clCdeQueue, clKernel, 1, NULL, &globalSize, &blockSize, 0, NULL, &clEvent));
	wbCheck(clWaitForEvents(1, &clEvent));

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  clEnqueueReadBuffer(clCdeQueue, d_Out, CL_TRUE, 0, inputLength*sizeof(float), hostOutput, 0, NULL, NULL);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
	
  //@@ Free the GPU memory here
	clReleaseMemObject(d_In1);
	clReleaseMemObject(d_In2);
	clReleaseMemObject(d_Out);
			
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);
	
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
