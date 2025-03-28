#ifndef POLYGON_CUH
#define POLYGON_CUH

#include "point.cuh"

struct TPolygon 
{
public:
    __host__ __device__ TPolygon() {}
    
    __host__ __device__ TPolygon(TPoint p1Val, TPoint p2Val, TPoint p3Val, uchar4 colourVal) 
    {
        p1 = p1Val;
        p2 = p2Val;
        p3 = p3Val;
        
        colour = colourVal;
    }
    
    __host__ __device__ TPoint GetP1(){
        return p1;
    }
    
    __host__ __device__ TPoint GetP2(){
        return p2;
    }
    
    __host__ __device__ TPoint GetP3(){
        return p3;
    }
    
    __host__ __device__ uchar4 GetColour(){
        return colour;
    }
    
    __host__ __device__ void SetP1(TPoint pVal){
        p1 = pVal;
    }
    
    __host__ __device__ void SetP2(TPoint pVal){
        p2 = pVal;
    }
    
    __host__ __device__ void SetP3(TPoint pVal){
        p3 = pVal;
    }
    
    __host__ __device__ void SetColour(uchar4 colourVal){
        colour = colourVal;
    }
    
private:
    TPoint p1, p2, p3;
    uchar4 colour;
};

#endif
