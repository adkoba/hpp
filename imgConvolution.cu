#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + (Mask_width-1))

//@@ INSERT CODE HERE

#define clamp(x, start, end) min(max(x, start), end)


__global__ void imageConvolution( float* inputData, float* outputData
							, int width, int height, int channels
							, float* const __restrict__ M )
{
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH][3];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// Compute output position
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;
	
	// Compute input position
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;
	
	// Load input data and ghost in shared memory
	for(int chIt = 0; chIt < channels; chIt++) {
		if( row_i >= 0 && row_i < height && col_i >= 0 && col_i < width ) {
			Ns[ty][tx][chIt] = inputData[(row_i*width + col_i)*channels + chIt];
		} else {
			Ns[ty][tx][chIt] = 0.0f;
		}
	}
	
	// We must to insure that each data is launch into share memory
	// before to start convolution computation
	__syncthreads();
		
	if( ty < O_TILE_WIDTH && tx < O_TILE_WIDTH )
	{
		// We compute output value
		float output;

		for(int chIt = 0; chIt < channels; chIt++) {
			output = 0.0f;

			for(int rowIt = 0; rowIt < Mask_width; rowIt++) {
				for(int colIt = 0; colIt < Mask_width; colIt++) {
					output += M[rowIt*Mask_width + colIt] * Ns[rowIt+ty][colIt+tx][chIt];
				}
			}
				
			// Finaly we set the output value for the current channel
			if(row_o < height && col_o < width ) {
				outputData[(row_o*width + col_o)*channels + chIt] = clamp(output, 0.0, 1.0);		
			}
		}
	}
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

	wbLog(TRACE, "The dimensions of Input Image are ", imageWidth, " x ", imageHeight, " x ", imageChannels);
	
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
	dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1);
	
	wbLog(TRACE, "dimBlock = ", dimBlock.x, " x ", dimBlock.y);
	wbLog(TRACE, "dimGrid = ", dimGrid.x, " x ", dimGrid.y);
	
	imageConvolution<<<dimGrid,dimBlock>>>( deviceInputImageData, deviceOutputImageData
											, imageWidth, imageHeight, imageChannels
											, deviceMaskData);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}