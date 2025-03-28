#ifndef PARAMS_CUH
#define PARAMS_CUH

#include "point.cuh"

class PFrame
{
public:
    int amount;
    std::string path;
    int width, height;
    double angle;
};

std::istream& operator>>(std::istream& fin, PFrame& data) 
{
    fin >> data.amount >> data.path >> data.width >> data.height >> data.angle;
    return fin;
}

class PCamera 
{
public:
    double r0c, z0c, phi0c, arc, azc, wrc, wzc, wphic, prc, pzc;
    double r0n, z0n, phi0n, arn, azn, wrn, wzn, wphin, prn, pzn;
};

std::istream& operator>>(std::istream& fin, PCamera& data) 
{
    fin >> data.r0c >> data.z0c >> data.phi0c >> data.arc >> data.azc >> data.wrc >> data.wzc >> data.wphic >> data.prc >> data.pzc;
    fin >> data.r0n >> data.z0n >> data.phi0n >> data.arn >> data.azn >> data.wrn >> data.wzn >> data.wphin >> data.prn >> data.pzn;
    
    return fin;
}

class PFloor 
{
public:
    TPoint p1, p2, p3, p4;
    uchar4 colour;
};

std::istream& operator>>(std::istream& fin, PFloor& data) 
{
    double r, g, b;
    
    fin >> data.p1 >> data.p2 >> data.p3 >> data.p4;
    fin >> r >> g >> b;
    
    data.colour = make_uchar4(r * 255, g * 255, b * 255, 255);

    return fin;
}

class PLight 
{
public:
    TPoint position;
    uchar4 colour;
    double rayPerPixel;
};

std::istream& operator>>(std::istream& in, PLight& l) {
    in >> l.position;

    double r, g, b;
    in >> r >> g >> b;
    l.colour = make_uchar4(r * 255, g * 255, b * 255, 255);

    in >> l.rayPerPixel;
    return in;
}

class PFigure 
{
public:
    TPoint center;
    uchar4 colour;
    double radius;
};

std::istream& operator>>(std::istream& fin, PFigure& params) 
{
    fin >> params.center;

    double r, g, b;
    fin >> r >> g >> b;
    params.colour = make_uchar4(r * 255, g * 255, b * 255, 255);

    fin >> params.radius;
    return fin;
}

#endif