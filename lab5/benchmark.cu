#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

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

void GenerateRandomArray(std::vector<int>& data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % 100000; // случайные числа в диапазоне [0, 100000]
    }
}

__device__ void MergeStep(int* data, int size, int left, int right, int step, int posX) {
    __shared__ int sharedMem[SIZE_OF_BLOCKS];
    for (int i = left; i < right; i += step) {
        int* tmp = data + i;
        if (posX >= SIZE_OF_BLOCKS || i + posX >= size) return;
        sharedMem[posX] = tmp[posX];
        __syncthreads();
        for (int j = SIZE_OF_BLOCKS / 2; j > 0; j /= 2) {
            unsigned int XOR = posX ^ j;
            if (XOR > posX && XOR < SIZE_OF_BLOCKS) {
                if ((posX & SIZE_OF_BLOCKS) != 0) {
                    if (sharedMem[posX] < sharedMem[XOR]) {
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                } else {
                    if (sharedMem[posX] > sharedMem[XOR]) {
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                }
            }
            __syncthreads();
        }
        tmp[posX] = sharedMem[posX];
    }
}

__global__ void BitonicMerge(int* data, int size, bool isOdd) {
    unsigned int posX = threadIdx.x;
    int blockId = blockIdx.x;
    int shift = gridDim.x;
    if (blockId * SIZE_OF_BLOCKS + posX < size)
        MergeStep(data, size, blockId * SIZE_OF_BLOCKS, size, shift * SIZE_OF_BLOCKS, posX);
}

__global__ void SortStep(int* data, int j, int k, int size) {
    __shared__ int sharedMem[SIZE_OF_BLOCKS];
    unsigned int posX = threadIdx.x;
    int blockId = blockIdx.x;
    int shift = gridDim.x;
    for (int i = blockId * SIZE_OF_BLOCKS; i < size; i += shift * SIZE_OF_BLOCKS) {
        if (i + posX >= size) return;
        sharedMem[posX] = data[i + posX];
        __syncthreads();
        for (j = k / 2; j > 0; j /= 2) {
            unsigned int XOR = posX ^ j;
            if (XOR > posX && XOR < SIZE_OF_BLOCKS) {
                if ((posX & k) != 0) {
                    if (sharedMem[posX] < sharedMem[XOR]) {
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                } else {
                    if (sharedMem[posX] > sharedMem[XOR]) {
                        thrust::swap(sharedMem[posX], sharedMem[XOR]);
                    }
                }
            }
            __syncthreads();
        }
        data[i + posX] = sharedMem[posX];
    }
}

void BitonicSort(int* devData, int partitionSize) {
    for (int i = 2; i <= partitionSize; i *= 2) {
        for (int j = i / 2; j > 0; j /= 2) {
            SortStep<<<NUM_OF_BLOCKS, SIZE_OF_BLOCKS>>>(devData, j, i, partitionSize);
            CSC(cudaGetLastError());
        }
    }
    for (int i = 0; i < 2 * (partitionSize / SIZE_OF_BLOCKS); ++i) {
        BitonicMerge<<<NUM_OF_BLOCKS, SIZE_OF_BLOCKS>>>(devData, partitionSize, (i % 2 == 0));
        CSC(cudaGetLastError());
    }
}

void RunSortingTest(int size) {
    std::vector<int> data(size);
    GenerateRandomArray(data, size);
    int* devData;
    CSC(cudaMalloc((void**)&devData, sizeof(int) * size));
    CSC(cudaMemcpy(devData, data.data(), sizeof(int) * size, cudaMemcpyHostToDevice));
    auto start = std::chrono::high_resolution_clock::now();
    BitonicSort(devData, size);
    CSC(cudaMemcpy(data.data(), devData, sizeof(int) * size, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Sorting " << size << " elements took " << elapsed.count() << " seconds\n";
    CSC(cudaFree(devData));
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    RunSortingTest(10);
    //RunSortingTest(10000);
    //RunSortingTest(100000);
    //RunSortingTest(1000000);

    return 0;
}
