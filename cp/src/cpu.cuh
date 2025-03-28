#ifndef CPU_CUH
#define CPU_CUH

#include "ray.cuh"

__host__ __device__ void CPURender(int rayCnt, uchar4* pixelBuffer, TPoint cameraPosition, TPoint cameraDirection, TRayParams params) 
{
    double pixelWidth = 2.0 / (params.width - 1.0);
    double pixelHeight = 2.0 / (params.height - 1.0);
    double focalLength = 1.0 / tan(params.angle * M_PI / 360.0);

    TPoint forwardVector = (cameraDirection - cameraPosition).normalize();
    TPoint rightVector = (VectorProd(forwardVector, {0.0, 0.0, 1.0})).normalize();
    TPoint upVector = (VectorProd(rightVector, forwardVector)).normalize();

    for (int x = 0; x < params.width; ++x) 
    {
        for (int y = 0; y < params.height; ++y) 
        {
            TPoint screenCoordinate = TPoint(-1.0 + pixelWidth * x, (-1.0 + pixelHeight * y) * params.height / params.width, focalLength);
            TPoint rayDirection = MatrixProd(rightVector, upVector, forwardVector, screenCoordinate);
            
            pixelBuffer[(params.height - 1 - y) * params.width + x] = TraceRay(cameraPosition, rayDirection.normalize(), params);
            rayCnt += 4;
        }
    }
}

__host__ __device__ void CPUSmoothing(uchar4* inputBuffer, uchar4* outputBuffer, TSmoothParams params) 
{
    for (int x = 0; x < params.width; ++x) 
    {
        for (int y = 0; y < params.height; ++y) 
        {
            uint4 colorAccumulator = make_uint4(0, 0, 0, 0);
            
            for (int i = 0; i < params.rayPerPixel; ++i) 
            {
                for (int j = 0; j < params.rayPerPixel; ++j) 
                {
                    uchar4 currentPixel = inputBuffer[(params.width * params.rayPerPixel * (y * params.rayPerPixel + j) + (x * params.rayPerPixel + i))];
                    
                    colorAccumulator.x += currentPixel.x;
                    colorAccumulator.y += currentPixel.y;
                    colorAccumulator.z += currentPixel.z;
                }
            }
            
            int totalSamples = params.rayPerPixel * params.rayPerPixel;
            outputBuffer[y * params.width + x] = make_uchar4(colorAccumulator.x / totalSamples, colorAccumulator.y / totalSamples, colorAccumulator.z / totalSamples, 255);
        }
    }
}

#endif