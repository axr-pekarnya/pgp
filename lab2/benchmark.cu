#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define CSC(call) 														                        \
while (1) 																				        \
{																							    \
	cudaError_t status = call;									                                \
	if (status != cudaSuccess) {								                                \
		printf("ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));\
		exit(0);																	            \
	}																						    \
	break;																					    \
}

__global__ void kernel(cudaTextureObject_t texture, int width, int height, uchar4 *res) {
    int posX = blockDim.x * blockIdx.x + threadIdx.x;
    int posY = blockDim.y * blockIdx.y + threadIdx.y;

    int shiftX = blockDim.x * gridDim.x;
    int shiftY = blockDim.y * gridDim.y;
    
    for(int y = posY; y < height; y += shiftY) {
        for(int x = posX; x < width; x += shiftX) {
            uchar4 tmp;
            float accX = 0;
            float accY = 0;

            y = max(min(y, height), 0);
            x = max(min(x, width), 0);

            for (int curY = -1; curY <= 1; curY++) {
                for (int curX = -1; curX <= 1; curX += 2) {
                    tmp = tex2D<uchar4>(texture, x + curX, y + curY);
                    float Y = 0.299 * tmp.x + 0.587 * tmp.y + 0.114 * tmp.z;
                    accX += curX * Y;
                }
            }

            for (int curY = -1; curY <= 1; curY += 2) {
                for (int curX = -1; curX <= 1; curX++) {
                    tmp = tex2D<uchar4>(texture, x + curX, y + curY);
                    float Y = 0.299 * tmp.x + 0.587 * tmp.y + 0.114 * tmp.z;
                    accY += curY * Y;
                }
            }
        
            float grad = min(max(sqrt(accX * accX + accY * accY), 0.0f), 255.0f);
            res[y * width + x] = make_uchar4(grad, grad, grad, tmp.w);
        }
    }
}

int main() {
    std::string inputFile, resputFile;
    std::cin >> inputFile >> resputFile;

    int width, height;
    std::ifstream in(inputFile, std::ios::binary);
    in.read(reinterpret_cast<char*>(&width), sizeof(int));
    in.read(reinterpret_cast<char*>(&height), sizeof(int));
    
    std::vector<uchar4> data(width * height);
    in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
    in.close();

    cudaArray *arr;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &channel, width, height));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data.data(), width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *res;
    CSC(cudaMalloc(&res, sizeof(uchar4) * width * height));

    int configs[5] = {64, 128, 256, 512, 1024};
    for (int i = 0; i < 5; i++) {
        dim3 blocks(configs[i], configs[i]);
        dim3 threads(32, 32);
        
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<blocks, threads>>>(tex, width, height, res);
        CSC(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        
        std::cout << "Execution time for <<<" << configs[i] << ", " << configs[i] << ">>>: " << duration.count() << " ms" << std::endl;
    }

    CSC(cudaMemcpy(data.data(), res, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(res));

    std::ofstream out(resputFile, std::ios::binary);
    out.write(reinterpret_cast<char*>(&width), sizeof(int));
    out.write(reinterpret_cast<char*>(&height), sizeof(int));
    out.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(uchar4));
    out.close();

    return 0;
}
