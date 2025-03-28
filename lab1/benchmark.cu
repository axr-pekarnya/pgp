#include <iomanip>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <random>

const int CONFIGS[5][2] = {{1, 32}, {16, 64}, {64, 256}, {512, 512}, {1024, 1024}};
const int SIZES[3] = {100, 100000, 10000000};

__device__ double Multiply(double x, double y){
    return x * y;
}

__global__ void kernel(double *a, double *b, double *res, int size) 
{
    int shift = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += shift) {
        res[i] = Multiply(a[i], b[i]);
    }
}

__host__ void PrintVector(double *vec, int n)
{
    for (int i = 0; i < n; ++i){
        std::cout << std::fixed << std::setprecision(10) << vec[i] << ' ';
    }
    std::cout << '\n';
}

int main() 
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int s = 0; s < 3; ++s) {
        int n = SIZES[s];
        int size = sizeof(double) * n;
        
        double *a = (double*)malloc(size);
        double *b = (double*)malloc(size);
        double *res = (double*)malloc(size);

        for (int i = 0; i < n; ++i){
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        
        double *devA, *devB, *devRes;
        cudaMalloc(&devA, size);
        cudaMalloc(&devB, size);
        cudaMalloc(&devRes, size);

        cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);

        std::cout << "Testing for n=" << n << std::endl;
        
        for (int i = 0; i < 5; ++i) {
            int blocks = CONFIGS[i][0];
            int threads = CONFIGS[i][1];
            
            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();
            
            kernel<<<blocks, threads>>>(devA, devB, devRes, n);
            
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Config <<<" << blocks << ", " << threads << ">>>: " 
                      << elapsed.count() << " seconds" << std::endl;
        }

        cudaMemcpy(res, devRes, size, cudaMemcpyDeviceToHost);
        
        free(a);
        free(b);
        free(res);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devRes);
    }

    return 0;
}
