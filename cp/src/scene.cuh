#ifndef TScene_CUH
#define TScene_CUH

#include "params.cuh"
#include "figures.cuh"
#include "floor.cuh"
#include "ray.cuh"

#include "gpu.cuh"
#include "cpu.cuh"
#include "camera.cuh"

#include <fstream>
#include <chrono>

class TScene 
{
public:
    TScene(TCamera cameraVal, PFrame frameVal, PFigure tetrahedronVal, PFigure hexahedronVal, PFigure icosahedronVal, PFloor floorVal, PLight lightVal) 
    {
        camera = cameraVal;
        frame = frameVal;
        tetrahedron = tetrahedronVal;
        hexahedron = hexahedronVal;
        icosahedron = icosahedronVal;
        floor = floorVal;
        light = lightVal;
    }
    
    void SaveData()
    {
        for (int i = 0; i < (int)dataFrames.size(); ++i)
        {
            size_t pos = frame.path.find("%d");
            std::string fileName = frame.path.substr(0, pos) + std::to_string(i) + frame.path.substr(pos + 2);
    
            std::ofstream outFile(fileName, std::ios::binary);
            
            outFile.write(reinterpret_cast<const char*>(&frame.width), sizeof(int));
            outFile.write(reinterpret_cast<const char*>(&frame.height), sizeof(int));
            
            for (int j = 0; j < frame.width * frame.height; ++j)
            {
                outFile.put(dataFrames[i][j].x);
                outFile.put(dataFrames[i][j].y);
                outFile.put(dataFrames[i][j].z);
                outFile.put(dataFrames[i][j].w);
            }
            
            outFile.close();    
        }
    }
    
    void AddFigures()
    {
        TFloor fl(floor);
        fl.AddFigure(canvas);
        
        THexahedron hex(hexahedron);
        hex.AddFigure(canvas);
        
        TIcosahedron icos(icosahedron);
        icos.AddFigure(canvas);
        
        TTetrahedron tet(tetrahedron);
        tet.AddFigure(canvas);    
    }
    
    TRayParams GetRayParams(){
        return TRayParams(frame.width * light.rayPerPixel, frame.height * light.rayPerPixel, frame.angle, light.position, light.colour, polygons, numOfPolygons);
    }
    
    TSmoothParams GetSmoothParams(){
        return TSmoothParams(frame.width, frame.height, light.rayPerPixel);
    }
     
    void GPU() 
    {
        if (!canvas.size()){
            return;
        }
        
        dataFrames = std::vector<uchar4*>(frame.amount);
    
        polygons = canvas.data();
        numOfPolygons = canvas.size();
        
        int rayCnt = 0;
        int *devRayCnt;
        
        cudaMalloc((void**)&devRayCnt, sizeof(int));
        cudaMemcpy(devRayCnt, &rayCnt, sizeof(int), cudaMemcpyHostToDevice);
        
        data = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height * light.rayPerPixel * light.rayPerPixel);
        smoothedData = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height);    
    
        cudaMalloc(&devData, sizeof(uchar4) * frame.width * frame.height * light.rayPerPixel * light.rayPerPixel);
        cudaMalloc(&devSmoothedData, sizeof(uchar4) * frame.width * frame.height);
        
        TRayParams rayParams = GetRayParams();
        TSmoothParams smoothParams = GetSmoothParams();
        
        int sumOfRays = 0;
        
        for (int i = 0; i < frame.amount; ++i) 
        {
            double time = 2 * M_PI * i / frame.amount;
            
            TPoint camCurPos = TPoint(camera.GetPosRadialX(time), camera.GetPosRadialY(time), camera.GetPosHeightZ(time));
            TPoint camCurDir = TPoint(camera.GetDirRadialX(time), camera.GetDirRadialY(time), camera.GetDirHeightZ(time));

            cudaEvent_t start, stop;
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            float milliseconds = 0;

            GPURender <<<dim3(blocksX, threadsX), dim3(blocksY, threadsY)>>>(
                devRayCnt, devData, camCurPos, camCurDir, rayParams);

            GPUSmoothing <<<dim3(blocksX, threadsX), dim3(blocksY, threadsY)>>>(
                devData, devSmoothedData, smoothParams);

            cudaMemcpy(smoothedData, devSmoothedData, sizeof(uchar4) * frame.width * frame.height, cudaMemcpyDeviceToHost);
            cudaMemcpy(&rayCnt, devRayCnt, sizeof(int), cudaMemcpyDeviceToHost);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            sumOfRays += rayCnt;
            std::cout << i << '\t' << milliseconds << '\t' << sumOfRays << '\n';
            
            std::ofstream outFile("1.txt", std::ios::app);
            outFile << milliseconds << '\n';
            outFile.close();
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            dataFrames[i] = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height);
            memcpy(dataFrames[i], smoothedData, sizeof(uchar4) * frame.width * frame.height);
        }

        free(data);
        free(smoothedData);
        
        cudaFree(devRayCnt);
        cudaFree(devData);
        cudaFree(devSmoothedData);
    }
    
    void CPU() 
    {
        if (!canvas.size()){
            return;
        }
        
        dataFrames = std::vector<uchar4*>(frame.amount);
    
        polygons = canvas.data();
        numOfPolygons = canvas.size();
        
        int rayCnt = 0;
    
        data = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height * light.rayPerPixel * light.rayPerPixel);
        smoothedData = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height);    
                
        TRayParams rayParams = GetRayParams();
        TSmoothParams smoothParams = GetSmoothParams();
        
        int sumOfRays = 0;
        
        for (int i = 0; i < frame.amount; ++i) 
        {
            double time = 2 * M_PI * i / frame.amount;
            
            TPoint camCurPos = TPoint(camera.GetPosRadialX(time), camera.GetPosRadialY(time), camera.GetPosHeightZ(time));
            TPoint camCurDir = TPoint(camera.GetDirRadialX(time), camera.GetDirRadialY(time), camera.GetDirHeightZ(time));

            auto start = std::chrono::high_resolution_clock::now();

            CPURender(rayCnt, data, camCurPos, camCurDir, rayParams);
            CPUSmoothing(data, smoothedData, smoothParams);
            
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            
            sumOfRays += rayCnt;
            
            std::cout << i << '\t' << duration.count() << '\t' << sumOfRays << '\n'; 
            
            std::ofstream outFile("2.txt", std::ios::app);
            outFile << duration.count() << '\n';
            outFile.close();
            
            dataFrames[i] = (uchar4*)malloc(sizeof(uchar4) * frame.width * frame.height);
            memcpy(dataFrames[i], smoothedData, sizeof(uchar4) * frame.width * frame.height);
        }

        free(data);
        free(smoothedData);
    }
    
    ~TScene(){
        for (uchar4* elem : dataFrames){
            free(elem);
        }
    }

private:
    TCamera camera;

    PFrame frame;
    PFigure tetrahedron;
    PFigure hexahedron;
    PFigure icosahedron;
    PFloor floor;
    PLight light;
    
    uchar4* data;
    uchar4* smoothedData;
    uchar4* devData;
    uchar4* devSmoothedData;
    
    std::vector<TPolygon> canvas;
    
    TPolygon *polygons;
    int numOfPolygons;
    
    std::vector<uchar4*> dataFrames;
    
    int blocksX = 8;
    int threadsX = 8;
    
    int blocksY = 8;
    int threadsY = 8;
};

#endif
