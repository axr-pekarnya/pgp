#ifndef POINT_CUH
#define POINT_CUH

#include <iostream>

class TPoint
{
public:
    __host__ __device__ TPoint() {}
    
    __host__ __device__ TPoint(double xVal, double yVal, double zVal) 
    {
        x = xVal;
        y = yVal;
        z = zVal;
    }
    
    __host__ __device__ double GetX(){
        return x;
    } 
    
    __host__ __device__ double GetY(){
        return y;
    } 
    
    __host__ __device__ double GetZ(){
        return z;
    } 
    
    __host__ __device__ void SetX(double val){
        x = val;
    } 
    
    __host__ __device__ void SetY(double val){
        y = val;
    } 
    
    __host__ __device__ void SetZ(double val){
        z = val;
    } 
    
    __host__ __device__ TPoint operator+(TPoint p) {
        return TPoint(x + p.x, y + p.y, z + p.z);
    }
    
    __host__ __device__ TPoint operator-(TPoint p) {
        return TPoint(x - p.x, y - p.y, z - p.z);
    }
    
    __host__ __device__ TPoint operator*(double num) {
        return TPoint(x * num, y * num, z * num);
    }
    
    __host__ __device__ TPoint normalize() 
    {
        double l = sqrt(ScalarProd(*this, *this));
        return TPoint(x / l, y / l, z / l);
    }
    
    friend std::istream& operator>>(std::istream& fin, TPoint& p);
    friend std::ostream& operator<<(std::ostream& fout, TPoint& p);
    
    friend __host__ __device__ double ScalarProd(TPoint p1, TPoint p2);
    
private:
    double x, y, z;
};

std::istream& operator>>(std::istream& fin, TPoint& p) {
    fin >> p.x >> p.y >> p.z;
    return fin;
}

std::ostream& operator<<(std::ostream& fout, TPoint& p) {
    fout << p.x << " " <<  p.y << " " << p.z;
    return fout;
}

__host__ __device__ double ScalarProd(TPoint p1, TPoint p2) {
    return p1.GetX() * p2.GetX() + p1.GetY() * p2.GetY() + p1.GetZ() * p2.GetZ();
}

__host__ __device__ TPoint VectorProd(TPoint p1, TPoint p2) {
    return TPoint(p1.GetY() * p2.GetZ() - p1.GetZ() * p2.GetY(), 
                  p1.GetZ() * p2.GetX() - p1.GetX() * p2.GetZ(), 
                  p1.GetX() * p2.GetY() - p1.GetY() * p2.GetX());
}

__host__ __device__ TPoint MatrixProd(TPoint p1, TPoint p2, TPoint p3, TPoint p4) {
    return TPoint(p1.GetX() * p4.GetX() + p2.GetX() * p4.GetY() + p3.GetX() * p4.GetZ(), 
                  p1.GetY() * p4.GetX() + p2.GetY() * p4.GetY() + p3.GetY() * p4.GetZ(), 
                  p1.GetZ() * p4.GetX() + p2.GetZ() * p4.GetY() + p3.GetZ() * p4.GetZ());
}

#endif