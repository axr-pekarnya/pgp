#include <stdlib.h> 
#include <stdio.h> 
#include <stddef.h> 
#include <stdbool.h> 
#include <math.h> 
#include <cmath> 
#include <iomanip>

#include <thrust/extrema.h> 
#include <thrust/device_vector.h> 

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

class TComparator 
{ 
    public: 
        __host__ __device__ bool operator()(const double num1, const double num2) const { 
            return fabs(num1) < fabs(num2); 
        }
    private:
    protected:
}; 

__global__ void SwapLines(double *matrix, double *unitedMatrix, int n, int i, int j) 
{ 
    int posX = blockIdx.x * blockDim.x + threadIdx.x;  
    int shift = gridDim.x * blockDim.x; 

    double tmp;

    for (int k = posX; k < n; k += shift) 
    { 
        tmp = matrix[n * k + i];
        matrix[n * k + i] = matrix[n * k + j];
        matrix[n * k + j] = tmp;
        
        tmp = unitedMatrix[n * k + i];
        unitedMatrix[n * k + i] = unitedMatrix[n * k + j];
        unitedMatrix[n * k + j] = tmp;
    }
} 

__global__ void Divide(double* matrix, double* unitedMatrix, int n) 
{ 
    int posX = blockIdx.x * blockDim.x + threadIdx.x; 
    int posY = blockIdx.y * blockDim.y + threadIdx.y; 
    int shiftX = gridDim.x * blockDim.x; 
    int shiftY = gridDim.y * blockDim.y; 

    for (int i = posX; i < n; i += shiftX){ 
        for (int j = posY; j < n; j += shiftY){ 
            unitedMatrix[j * n + i] /= matrix[i * n + i]; 
        } 
    }  
} 

__global__ void DelLower (double* matrix, double* unitedMatrix, int n, int sep) 
{ 
    int posX = blockIdx.x * blockDim.x + threadIdx.x; 
    int posY = blockIdx.y * blockDim.y + threadIdx.y; 
    int shiftX = gridDim.x * blockDim.x; 
    int shiftY = gridDim.y * blockDim.y; 

    for (int i = sep + 1 + posX; i < n; i += shiftX) 
    { 
        double div = -matrix[sep * n + i] / matrix[sep * n + sep]; 
  
        for (int j = sep + 1 + posY; j < n; j += shiftY) { 
            matrix[j * n + i] += div * matrix[j * n + sep]; 
        } 
 
        for (int j = posY; j < n; j += shiftY) { 
            unitedMatrix[j * n + i] += div * unitedMatrix[j * n + sep]; 
        } 
    } 
} 

__global__ void DelUpper (double* matrix, double* unitedMatrix, int n, int sep) 
{ 
    int posX = threadIdx.x + blockIdx.x * blockDim.x; 
    int posY = threadIdx.y + blockIdx.y * blockDim.y; 
    int shiftX = gridDim.x * blockDim.x; 
    int shiftY = gridDim.y * blockDim.y; 

    for (int i = sep - posX - 1; i >= 0; i -= shiftX) 
    { 
        double div = -matrix[sep * n + i] / matrix[sep * n + sep]; 
 
        for (int j = posY; j < n; j += shiftY) { 
            unitedMatrix[j * n + i] += div * unitedMatrix[j * n + sep]; 
        } 
    } 
} 

int main() 
{
    int n; 
    std::cin >> n;  

    if (n < 1){
        return 0;
    }

    double* matrix = (double*)malloc(n * n * sizeof(double)); 
    
    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < n; ++j) { 
            std::cin >> matrix[j * n + i]; 
        } 
    } 
    
    double* unitedMatrix = (double*)malloc(n * n * sizeof(double)); 
    
    for (int i = 0; i < n; ++i) 
    { 
        for (int j = 0; j < n; ++j) 
        { 
            if (i == j){
                unitedMatrix[i * n + j] = 1;
            }
            else {
                unitedMatrix[i * n + j] = 0;
            }
        } 
    }   
    
    double* devMatrix; 
    double* devUnitedMatrix; 
    
    cudaMalloc(&devMatrix, sizeof(double) * n * n); 
    cudaMalloc(&devUnitedMatrix, sizeof(double) * n * n); 
    
    cudaMemcpy(devMatrix, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice); 
    cudaMemcpy(devUnitedMatrix, unitedMatrix, sizeof(double) * n * n, cudaMemcpyHostToDevice); 
    
    dim3 block(32, 16); 
    dim3 thread(32, 16); 

    const thrust::device_ptr<double> ptr = thrust::device_pointer_cast(devMatrix); 
    const TComparator cmp; 

    for (int i = 0; i < n - 1; ++i) 
    { 
        const int maxIdx = thrust::max_element(ptr + i * n + i, ptr + (i + 1) * n, cmp) - ptr - i * n; 
        
        if (maxIdx != i){ 
            SwapLines<<<256, 256>>>(devMatrix, devUnitedMatrix, n, i, maxIdx); 
        } 
        
        DelLower <<<block, thread>>>(devMatrix, devUnitedMatrix, n, i); 
    }

    for (int i = n - 1; i > 0; i--) { 
        DelUpper <<<block, thread>>>(devMatrix, devUnitedMatrix, n, i); 
    } 

    Divide <<<block, thread>>>(devMatrix, devUnitedMatrix, n); 

    cudaMemcpy(unitedMatrix, devUnitedMatrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost); 
    
    cudaFree(devMatrix); 
    cudaFree(devUnitedMatrix); 

    for (int i = 0; i < n; ++i) { 
        for (int j = 0; j < n; ++j) { 
            std::cout << std::fixed << std::setprecision(10) << unitedMatrix[j * n + i] << ' '; 
        } 
        std::cout << '\n'; 
    } 
    
    free(matrix); 
    free(unitedMatrix); 

    return 0; 
}