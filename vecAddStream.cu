#include	<wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
	} while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//@@ Insert code to implement vector addition here
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( i < len )
		out[i] = in1[i] + in2[i];
}

#define NB_STREAMS 8
#define BLOCK_SIZE 64

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
 
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	
	// Create streams
	cudaStream_t streams[NB_STREAMS];
	for ( int ii = 0 ; ii < NB_STREAMS; ++ii)
	{
		wbCheck(cudaStreamCreate(&streams[ii]));
	}
	
	int SegSize = (inputLength-1)/NB_STREAMS + 1;
	wbLog(TRACE, "The SegSize length is ", SegSize);
	
	// Allocate device buffers
	float *d_In1[NB_STREAMS];
	float *d_In2[NB_STREAMS];
	float *d_Out[NB_STREAMS];
	int size = SegSize * sizeof(float);
	for ( int ii = 0 ; ii < NB_STREAMS; ++ii)
	{
		wbCheck(cudaMalloc((void **) &d_In1[ii], size));
		wbCheck(cudaMalloc((void **) &d_In2[ii], size));
		wbCheck(cudaMalloc((void **) &d_Out[ii], size));
	}
	
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((SegSize-1)/BLOCK_SIZE + 1);
	wbLog(TRACE, "GridDim.x == ", dimGrid.x, " BlockDim.x == ", dimBlock.x);
	
	cudaMemcpyAsync(d_In1[0], hostInput1, size, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(d_In2[0], hostInput2, size, cudaMemcpyHostToDevice, streams[0]);
	
	vecAdd<<<dimGrid, dimBlock, 0, streams[0]>>>(d_In1[0], d_In2[0], d_Out[0], SegSize);

	cudaMemcpyAsync(d_In1[1], hostInput1+SegSize, size, cudaMemcpyHostToDevice, streams[1]);
	cudaMemcpyAsync(d_In2[1], hostInput2+SegSize, size, cudaMemcpyHostToDevice, streams[1]);
	
	// Launch work on gpu queue
	int idxStream=1;
	for ( ; idxStream<NB_STREAMS-1; idxStream++) 
	{
		cudaMemcpyAsync(hostOutput+((idxStream-1)*SegSize), d_Out[idxStream-1], size, cudaMemcpyDeviceToHost, streams[idxStream-1]);

		vecAdd<<<dimGrid, dimBlock, 0, streams[idxStream]>>>(d_In1[idxStream], d_In2[idxStream], d_Out[idxStream], SegSize);
		
		cudaMemcpyAsync(d_In1[idxStream+1], hostInput1+((idxStream+1)*SegSize), size, cudaMemcpyHostToDevice, streams[idxStream+1]);
		cudaMemcpyAsync(d_In2[idxStream+1], hostInput2+((idxStream+1)*SegSize), size, cudaMemcpyHostToDevice, streams[idxStream+1]);
	}

	cudaMemcpyAsync(hostOutput+((idxStream-1)*SegSize), d_Out[idxStream-1], size, cudaMemcpyDeviceToHost, streams[idxStream-1]);
	vecAdd<<<dimGrid, dimBlock, 0, streams[idxStream]>>>(d_In1[idxStream], d_In2[idxStream], d_Out[idxStream], SegSize);
	cudaMemcpyAsync(hostOutput+(idxStream*SegSize), d_Out[idxStream], size, cudaMemcpyDeviceToHost, streams[idxStream]);

	// Wait for it
	cudaDeviceSynchronize();
	
	wbTime_start(GPU, "Freeing GPU Memory");
	for ( int ii = 0 ; ii < NB_STREAMS; ++ii)
	{
		wbCheck(cudaFree(d_In1[ii]));
		wbCheck(cudaFree(d_In2[ii]));
	    wbCheck(cudaFree(d_Out[ii]));
	}
	wbTime_stop(GPU, "Freeing GPU Memory");

	
	for ( int ii = 0 ; ii < NB_STREAMS; ++ii)
	{
		wbCheck(cudaStreamDestroy(streams[ii]));
	}
		
    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

