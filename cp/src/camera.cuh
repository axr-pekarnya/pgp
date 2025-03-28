#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "params.cuh"

class TCamera
{
public:
    TCamera() = default;

    TCamera(PCamera paramsVal){
        params = paramsVal;
    }
    
    double GetPosRadialX(double time){
        return (params.r0c + params.arc * sin(params.wrc * time + params.prc)) * cos(params.phi0c + params.wphic * time);
    }
    
    double GetPosRadialY(double time){
        return (params.r0c + params.arc * sin(params.wrc * time + params.prc)) * sin(params.phi0c + params.wphic * time);
    }
    
    double GetPosHeightZ(double time){
        return params.z0c + params.azc * sin(params.wzc * time + params.pzc);
    }
    
    double GetDirRadialX(double time){
        return (params.r0n + params.arn * sin(params.wrn * time + params.prn)) * cos(params.phi0n + params.wphin * time);
    }
    
    double GetDirRadialY(double time){
        return (params.r0n + params.arn * sin(params.wrn * time + params.prn)) * sin(params.phi0n + params.wphin * time);
    }
    
    double GetDirHeightZ(double time){
        return params.z0n + params.azn * sin(params.wzn * time + params.pzn);
    }

private:
    PCamera params;
};

#endif
