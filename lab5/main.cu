#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CSC(call) 														                        \
while (1) 																				        \
{																							    \
	cudaError_t status = call;									                                \
	if (status != cudaSuccess)                                                                  \
    {								                                                            \
		printf("ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
		exit(0);																	            \
	}																						    \
	break;																					    \
}

const int NUM_OF_BLOCKS = 16;
const int SIZE_OF_BLOCKS = 1024;

__device__ void MergeStep(int* data, int size, int left, int right, int step, int posX) 
{
    __shared__ int sharedMem[SIZE_OF_BLOCKS];
    int* tmp = data;

    for (int i = left; i < right; i += step) 
    {
        int idx;
        tmp = data + i;

        if (posX >= SIZE_OF_BLOCKS / 2) {
            idx = SIZE_OF_BLOCKS * 3 / 2 - 1 - posX;
        } 
        else {
            idx = posX;
        }

        if (posX >= SIZE_OF_BLOCKS / 2) {
            sharedMem[posX] = tmp[idx];
        }
        else {
            sharedMem[posX] = tmp[posX];
        }
        
        __syncthreads();
        
        for (int j = SIZE_OF_BLOCKS / 2; j > 0; j /= 2) 
        {
            unsigned int XOR = posX ^ j;
        
			if (XOR > posX) 
            {
				if ((posX & SIZE_OF_BLOCKS) != 0) 
                {
					if (sharedMem[posX] < sharedMem[XOR]){
						thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
				} 
                else 
                {
					if (sharedMem[posX] > sharedMem[XOR]){
						thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
				}
			}
        
            __syncthreads();
        }

        tmp[posX] = sharedMem[posX];
    }
}

__global__ void BitonicMerge(int* data, int size, bool isOdd) 
{
    unsigned int posX = threadIdx.x;
    int blockId = blockIdx.x;
    int shift = gridDim.x;

    if (isOdd) {
        MergeStep(data, size, (SIZE_OF_BLOCKS / 2) + blockId * SIZE_OF_BLOCKS, size - SIZE_OF_BLOCKS, shift * SIZE_OF_BLOCKS, posX);
    } 
    else {
        MergeStep(data, size, blockId * SIZE_OF_BLOCKS, size, shift * SIZE_OF_BLOCKS, posX);
    }
}

__global__ void SortStep(int* data, int j, int k, int size)
{
    __shared__ int sharedMem[SIZE_OF_BLOCKS];
    int* tmp = data;
    
    unsigned int posX = threadIdx.x;
    int blockId = blockIdx.x;
    int shift = gridDim.x;
    
    for (int i = blockId * SIZE_OF_BLOCKS; i < size; i += shift * SIZE_OF_BLOCKS) 
    {
        tmp = data + i;
        sharedMem[posX] = tmp[posX];
        
        __syncthreads();
        
        for (j = k / 2; j > 0; j /= 2) 
        {
            unsigned int XOR = posX ^ j;
            if (XOR > posX) 
            {
                if ((posX & k) != 0) 
                {
                    if (sharedMem[posX] < sharedMem[XOR]){
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                } 
                else 
                {
                    if (sharedMem[posX] > sharedMem[XOR]){
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                }
            }

            __syncthreads();

            tmp[posX] = sharedMem[posX];
        }
    }
}

void BitonicSort(int* devData, int partitionSize) 
{
    for (int i = 2; i <= partitionSize; i *= 2) 
    {
        if (i > SIZE_OF_BLOCKS){
            break;
        }
        
        for (int j = i / 2; j > 0; j /= 2) 
        {
            SortStep <<<NUM_OF_BLOCKS, SIZE_OF_BLOCKS>>> (devData, j, i, partitionSize);
            CSC(cudaGetLastError());
        }
    }

    for (int i = 0; i < 2 * (partitionSize / SIZE_OF_BLOCKS); ++i) 
    {
        BitonicMerge <<<NUM_OF_BLOCKS, SIZE_OF_BLOCKS>>> (devData, partitionSize, (i % 2 == 0));
        CSC(cudaGetLastError());
    }
}


int main() 
{
    std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int n;
    std::cin.read(reinterpret_cast<char*>(&n), sizeof(int));
    
    int partitionSize = ceil((double) n / SIZE_OF_BLOCKS) * SIZE_OF_BLOCKS;
    std::vector<int> data(partitionSize, INT_MAX);
    
    std::cin.read(reinterpret_cast<char*>(data.data()), n * sizeof(int));
    
    int* devData;
    CSC(cudaMalloc((void**)&devData, sizeof(int) * partitionSize));
    CSC(cudaMemcpy(devData, data.data(), sizeof(int) * partitionSize, cudaMemcpyHostToDevice));
    
    BitonicSort(devData, partitionSize);

    CSC(cudaMemcpy(data.data(), devData, sizeof(int) * n, cudaMemcpyDeviceToHost));
    
    std::cout.write(reinterpret_cast<char*>(data.data()), n * sizeof(int));
    
    CSC(cudaFree(devData));
    
    return 0;
}
