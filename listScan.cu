// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


__global__ void addValue(float * inputVector, float * output, float outputDim)
{
	unsigned int t = threadIdx.x;
	unsigned int start = (blockIdx.x+1)*2*blockDim.x;
	
	if( start + t < outputDim )
	{
		output[start + t] += inputVector[blockIdx.x];
		
		if( start + t + blockDim.x < outputDim )
		{
			output[start + t + blockDim.x] += inputVector[blockIdx.x];
		}
	}		
}

__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

	__shared__ float InOutScan[2*BLOCK_SIZE];
	
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;
	
	if( start + t < len )
		InOutScan[t] = input[start + t];
	else
		InOutScan[t] = 0.f;
	
	if( start + blockDim.x + t < len )
		InOutScan[blockDim.x+t] = input[start+blockDim.x+t];
	else
		InOutScan[blockDim.x+t] = 0.f;

	__syncthreads();
	
	// Reduction Phase
	for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2) {
		int idx = (threadIdx.x+1)*stride*2 - 1;
		if(idx < 2*BLOCK_SIZE)
			InOutScan[idx] += InOutScan[idx-stride];
		__syncthreads();
	}
	
	// Post Reduction Reverse Phase
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index+stride < 2*BLOCK_SIZE) {
			InOutScan[index + stride] += InOutScan[index];
		}
	}
	__syncthreads();
	
	// Copy shared memory to output
	if (start + t < len)
		output[start + t] = InOutScan[t];
	if (start + blockDim.x + t < len)
		output[start+blockDim.x+t] = InOutScan[blockDim.x+t];

	// Copy last value from shared memory to input for reuse 
	// in next stage in case where blockDim was superior to 1
	// this allow to reduce host/device interaction using the
	// fact that input data are only updated in device side
	// and are still present in device memory for a next call
	if( t == 0 )
		input[blockIdx.x] = InOutScan[2*BLOCK_SIZE-1];
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	float * scanBlockSums; // Device global memory used to 
	
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	
	// scanBlockSums's size doesn't exceed the number 
	// of block used by first call of scan
	wbCheck(cudaMalloc((void**)&scanBlockSums, ((numElements-1)/(BLOCK_SIZE*2) + 1)*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
	wbCheck(cudaMemset(scanBlockSums, 0, ((numElements-1)/(BLOCK_SIZE*2) + 1)*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((numElements-1)/(BLOCK_SIZE*2) + 1);

	wbLog(TRACE, "Launch kernel with GridDim.x == ", dimGrid.x , " blockDim.x == ", dimBlock.x);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the device
	scan<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, numElements);

	if( dimGrid.x > 1 )
	{
		int newInputSize = dimGrid.x;
		int newDim = ((dimGrid.x -1)/(BLOCK_SIZE*2) + 1);

		wbLog(TRACE, "Launch kernel with GridDim.x == ", newDim , " blockDim.x == ", BLOCK_SIZE);

		scan<<<newDim,BLOCK_SIZE>>>(deviceInput, scanBlockSums, newInputSize);

		addValue<<<newInputSize-1,BLOCK_SIZE>>>(scanBlockSums, deviceOutput, numElements);
	}

	wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
	cudaFree(scanBlockSums);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

