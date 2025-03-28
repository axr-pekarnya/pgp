#ifndef TRANS_PARAMS_CUH
#define TRANS_PARAMS_CUH

#include "polygon.cuh"

struct TRayParams
{
    int width;
    int height;
    double angle;
    
    TPoint lightPos;
    uchar4 lightColour;
    
    TPolygon* canvas;
    int canvasSize;
    
    TRayParams(int widthVal, int heightVal, double angleVal, TPoint lightPosVal, uchar4 lightColourVal, TPolygon* canvasVal, int canvasSizeVal)
    {
        width = widthVal;
        height = heightVal;
        angle = angleVal;
        
        lightPos = lightPosVal;
        lightColour = lightColourVal;
        
        canvas = canvasVal;
        canvasSize = canvasSizeVal;
    }
};

struct TSmoothParams
{
    int width;
    int height;
    int rayPerPixel;
    
    TSmoothParams(int widthVal, int heightVal, double rayPerPixelVal)
    {
        width = widthVal;
        height = heightVal;
        rayPerPixel = rayPerPixelVal;
    }
};


#endif
