#include <iostream>
#include <fstream>
#include <vector>
#include <float.h>
#include <cuda_runtime.h>
#include <chrono>

__constant__ float3 devAvg[32];

#define CSC(call) \
while (1)  \
{ \
    cudaError_t status = call; \
    if (status != cudaSuccess) \
    { \
        printf("ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
    break; \
}

__global__ void kernel(uchar4 *data, int width, int height, int nc) 
{
    int posX = blockDim.x * blockIdx.x + threadIdx.x;
    int shiftX = blockDim.x * gridDim.x;

    for (int i = posX; i < width * height; i += shiftX) 
    {
        float max = -FLT_MAX;
    
        for (int j = 0; j < nc; j++) 
        {
            float cur = -(data[i].x - devAvg[j].x) * (data[i].x - devAvg[j].x) \
                        - (data[i].y - devAvg[j].y) * (data[i].y - devAvg[j].y) \
                        - (data[i].z - devAvg[j].z) * (data[i].z - devAvg[j].z);

            if (cur > max) 
            {
                max = cur;
                data[i].w = j;
            }
        }
    }
}

std::vector<float3> GetAvg(const std::vector<int>& np, std::vector<std::vector<std::pair<int, int>>>& pos, std::vector<uchar4>& data, int width) 
{
    std::vector<float3> res(np.size());
    float3 acc;

    for (size_t i = 0; i < np.size(); i++) 
    {
        acc = {0, 0, 0};
        
        for (size_t j = 0; j < np[i]; j++) 
        {
            uchar4 cur = data[pos[i][j].first + pos[i][j].second * width];

            acc.x += cur.x;
            acc.y += cur.y;
            acc.z += cur.z;
        }
        
        acc.x /= np[i];
        acc.y /= np[i];
        acc.z /= np[i];

        res[i] = acc;
    }

    return res;
}

int main() 
{
    int width, height, nc;
    std::string inputFile, resputFile;
    
    std::cin >> inputFile >> resputFile;
    std::cin >> nc;

    std::vector<int> np(nc);
    std::vector<std::vector<std::pair<int, int>>> pos(nc);
    
    for (int i = 0; i < nc; i++) 
    {
        std::cin >> np[i];
        pos[i].resize(np[i]);
    
        for (int j = 0; j < np[i]; j++) {
            std::cin >> pos[i][j].first >> pos[i][j].second;
        }
    }

    std::ifstream in(inputFile, std::ios::binary);
    in.read(reinterpret_cast<char*>(&width), sizeof(int));
    in.read(reinterpret_cast<char*>(&height), sizeof(int));

    std::vector<uchar4> data(width * height);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(uchar4) * width * height);
    in.close();
    
    std::vector<float3> avg = GetAvg(np, pos, data, width);
    
    uchar4* devData;
    CSC(cudaMalloc(&devData, sizeof(uchar4) * width * height));
    CSC(cudaMemcpy(devData, data.data(), sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));

    CSC(cudaMemcpyToSymbol(devAvg, avg.data(), sizeof(float3) * 32));

    int grid_sizes[] = {64, 128, 256, 512, 1024};
    for (int i = 0; i < 5; i++) {
        int grid_size = grid_sizes[i];
        int block_size = grid_size;

        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<grid_size, block_size>>>(devData, width, height, nc);
        CSC(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        
        std::cout << "Configuration <<<" << grid_size << ", " << block_size << ">>> Execution time: " 
                  << duration.count() << " ms" << std::endl;
    }
    
    CSC(cudaMemcpy(data.data(), devData, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
    CSC(cudaFree(devData));
    
    std::ofstream out(resputFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&height), sizeof(int));
    out.write(reinterpret_cast<const char*>(data.data()), sizeof(uchar4) * width * height);
    out.close();
    
    return 0;
}
