// Histogram Equalization

#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
	
#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128


//@@ insert code here
__global__ void castUcharImage( float* inputDataImg, unsigned char* outUcharImage, int width, int height, int channels )
{
	int rowIt = blockIdx.y*blockDim.y + threadIdx.y;
	int colIt = blockIdx.x*blockDim.x + threadIdx.x; 

	if( rowIt < height && colIt < width )
	{
		int idx =  (rowIt * width + colIt)*channels;
		
		for( int itChannel = 0 ; itChannel < channels; itChannel++ )
		{
			outUcharImage[idx + itChannel] = (unsigned char)(255*inputDataImg[idx + itChannel]);
		}
	}
}


__global__ void rgbToGrayScaleImage( unsigned char* inUcharImage, unsigned char* outGrayImage, int width, int height, int channels )
{
	int rowIt = blockIdx.y*blockDim.y + threadIdx.y;
	int colIt = blockIdx.x*blockDim.x + threadIdx.x; 
	
	if( rowIt < height && colIt < width )
	{
		int idx =  (rowIt * width + colIt)*channels;
		
        unsigned char r = inUcharImage[idx];
        unsigned char g = inUcharImage[idx + 1];
        unsigned char b = inUcharImage[idx + 2];
        outGrayImage[rowIt * width + colIt] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}


__global__ void histogram( unsigned char* inGrayImage, unsigned int* outHistogram, int width, int height)
{
	__shared__ unsigned int private_histo[HISTOGRAM_LENGTH];
	
	int rowIt = blockIdx.y*blockDim.y + threadIdx.y;
	int colIt = blockIdx.x*blockDim.x + threadIdx.x; 
	
	if (threadIdx.x < HISTOGRAM_LENGTH && threadIdx.y == 0) 
		private_histo[threadIdx.x] = 0;
	
	__syncthreads();
	
	if( rowIt < height && colIt < width )
	{
		int idx =  rowIt * width + colIt;
		atomicAdd(&(private_histo[inGrayImage[idx]]), 1);
	}

	__syncthreads();
									  
	if (threadIdx.x < HISTOGRAM_LENGTH && threadIdx.y == 0) 
		atomicAdd(&(outHistogram[threadIdx.x]),private_histo[threadIdx.x]);
								  
}

__device__ float probability(float value, int width, int height)
{
    return value / ((float)(width * height));
}
		
__global__ void cdf(unsigned int * inHistogram, float * outCDF, int width, int height) 
{
	__shared__ float InOutScan[2*BLOCK_SIZE];
	
	unsigned int t = threadIdx.x;

	// Load value in shared memory
	InOutScan[t] = probability((float)inHistogram[t], width, height);
	InOutScan[BLOCK_SIZE+t] = probability((float)inHistogram[BLOCK_SIZE+t], width, height);

	__syncthreads();
	
	// Reduction Phase
	for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2) {
		int idx = (t+1)*stride*2 - 1;
		if(idx < 2*BLOCK_SIZE)
			InOutScan[idx] += InOutScan[idx-stride];
		__syncthreads();
	}
	
	// Post Reduction Reverse Phase
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (t+1)*stride*2 - 1;
		if(index+stride < 2*BLOCK_SIZE) {
			InOutScan[index + stride] += InOutScan[index];
		}
	}
	__syncthreads();
	
	// Copy shared memory to output
	outCDF[t] = InOutScan[t];
	outCDF[BLOCK_SIZE+t] = InOutScan[BLOCK_SIZE+t];
}

__global__ void cdfmin(float * inCDFValues, float * output)
{
	__shared__ float minCdfValue[2*BLOCK_SIZE];
	
	unsigned int t = threadIdx.x;
	
	minCdfValue[t] = inCDFValues[t];
	minCdfValue[BLOCK_SIZE+t] = inCDFValues[BLOCK_SIZE+t];

	for (unsigned int stride = BLOCK_SIZE; stride > 0; stride /= 2) {
		__syncthreads();
		if (t < stride)
			minCdfValue[t] = min( minCdfValue[t], minCdfValue[t+stride]);
	}
	
	if( t == 0 )
		*output = minCdfValue[t];
}

__device__ unsigned char clamp(unsigned char value, unsigned char start, unsigned char end )
{
    return min(max(value, start), end);
}

__device__ unsigned char correct_color(unsigned char value, float* inCDF, float inCdfMin )
{
	return clamp((unsigned char)(255*(inCDF[value] - inCdfMin)/(1 - inCdfMin)), 0, 255);
}

__global__ void applyHistoEgalFunc( unsigned char* inOutUcharImage, float* inCDF, float *inCdfMin
								   	, int width, int height, int channels )
{
	int rowIt = blockIdx.y*blockDim.y + threadIdx.y;
	int colIt = blockIdx.x*blockDim.x + threadIdx.x; 

	if( rowIt < height && colIt < width )
	{
		int idx =  (rowIt * width + colIt)*channels;
		
		for( int itChannel = 0 ; itChannel < channels; itChannel++ )
		{
			inOutUcharImage[idx + itChannel] = correct_color(inOutUcharImage[idx + itChannel], inCDF, *inCdfMin);
		}
	}
}

__global__ void castUcharToFloat( unsigned char* inUcharImage, float* outDataImg
								 , int width, int height, int channels )
{
	int rowIt = blockIdx.y*blockDim.y + threadIdx.y;
	int colIt = blockIdx.x*blockDim.x + threadIdx.x; 

	if( rowIt < height && colIt < width )
	{
		int idx =  (rowIt * width + colIt)*channels;
		
		for( int itChannel = 0 ; itChannel < channels; itChannel++ )
		{
			outDataImg[idx + itChannel] = (float) (inUcharImage[idx + itChannel]/255.0);
		}
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	float *d_InputImage;
	float *d_OutputImage;
	unsigned char *d_UcharImage;
	unsigned char *d_GrayImage;
	unsigned int *d_Histogram;
	float *d_CDF;
	float *d_MinCdfValue;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of Input Image are ", imageWidth, " x ", imageHeight, " x ", imageChannels);
	
    //@@ insert code here
	
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU memory allocation");
	wbCheck(cudaMalloc((void **) &d_InputImage, imageWidth * imageHeight * imageChannels * sizeof(float)));
	wbCheck(cudaMalloc((void **) &d_OutputImage, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &d_UcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &d_GrayImage, imageWidth * imageHeight * sizeof(unsigned char)));
    wbCheck(cudaMalloc((void **) &d_Histogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
	wbCheck(cudaMalloc((void **) &d_CDF, HISTOGRAM_LENGTH * sizeof(float)));
	wbCheck(cudaMalloc((void **) &d_MinCdfValue, sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(d_InputImage,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");

	dim3 dimBlock(32,32);
	dim3 dimGrid((imageWidth-1)/32+1, (imageHeight-1)/32+1);
	
	wbLog(TRACE, "dimBlock = ", dimBlock.x, " x ", dimBlock.y, " x ",  dimBlock.z);
	wbLog(TRACE, "dimGrid = ", dimGrid.x, " x ", dimGrid.y);
	
	castUcharImage<<<dimGrid,dimBlock>>>( d_InputImage, d_UcharImage, imageWidth, imageHeight, imageChannels );
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());

	rgbToGrayScaleImage<<<dimGrid,dimBlock>>>( d_UcharImage, d_GrayImage, imageWidth, imageHeight, imageChannels );
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());

	dim3 dimBlockH(HISTOGRAM_LENGTH,4);
	dim3 dimGridH((imageWidth-1)/HISTOGRAM_LENGTH+1, (imageHeight-1)/4+1);
	
	wbLog(TRACE, "dimBlockH = ", dimBlockH.x, " x ", dimBlockH.y, " x ",  dimBlockH.z);
	wbLog(TRACE, "dimGridH = ", dimGridH.x, " x ", dimGridH.y);

	histogram<<<dimGridH,dimBlockH>>>( d_GrayImage, d_Histogram, imageWidth, imageHeight);
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());
	
	cdf<<<1,BLOCK_SIZE>>>(d_Histogram, d_CDF, imageWidth, imageHeight);
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());
	
	cdfmin<<<1,BLOCK_SIZE>>>(d_CDF, d_MinCdfValue);
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());

	wbLog(TRACE, "dimBlock = ", dimBlock.x, " x ", dimBlock.y, " x ",  dimBlock.z);
	wbLog(TRACE, "dimGrid = ", dimGrid.x, " x ", dimGrid.y);

	applyHistoEgalFunc<<<dimGrid,dimBlock>>>( d_UcharImage, d_CDF, d_MinCdfValue, imageWidth, imageHeight, imageChannels );
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());

	castUcharToFloat<<<dimGrid,dimBlock>>>( d_UcharImage, d_OutputImage, imageWidth, imageHeight, imageChannels);
	wbCheck(cudaGetLastError());
	wbCheck(cudaDeviceSynchronize());
	
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostOutputImageData,
               d_OutputImage,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost));
	
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    //@@ insert code here
    cudaFree(d_InputImage);
    cudaFree(d_UcharImage);
    cudaFree(d_GrayImage);
 	cudaFree(d_Histogram);
	cudaFree(d_CDF);
	cudaFree(d_MinCdfValue);
	
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

