#include <iomanip>
#include <iostream>

const int THREADS_AMOUNT = 32;
const int BLOCKS_AMOUNT = 32;

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
    int n;
    std::cin >> n;
    
    int size = sizeof(double) * n;

    double *a = (double*)malloc(size);
    double *b = (double*)malloc(size);

    for (int i = 0; i < n; ++i){
        std::cin >> a[i];
    }

    for (int i = 0; i < n; ++i){
        std::cin >> b[i];
    }
    
    double *devA;
    cudaMalloc(&devA, size);
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);

    double *devB;
    cudaMalloc(&devB, size);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);

    double *devRes;
    cudaMalloc(&devRes, size);

    kernel<<<BLOCKS_AMOUNT, THREADS_AMOUNT>>>(devA, devB, devRes, n);

    double *res = (double*)malloc(size);
    cudaMemcpy(res, devRes, size, cudaMemcpyDeviceToHost);

    PrintVector(res, n);

    free(a);
    free(b);
    free(res);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devRes);

    return 0;
}