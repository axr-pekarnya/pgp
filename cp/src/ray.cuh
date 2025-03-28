#ifndef RENDER_CUH
#define RENDER_CUH

#include "transParams.cuh"

__host__ __device__ uchar4 TraceRay(TPoint position, TPoint direction, TRayParams params, int depth = 3) 
{
    int closestPolygonIndex = -1;
    double closestIntersection;

    for (int i = 0; i < params.canvasSize; ++i) 
    {
        TPoint edge1 = params.canvas[i].GetP2() - params.canvas[i].GetP1();
        TPoint edge2 = params.canvas[i].GetP3() - params.canvas[i].GetP1();
        
        TPoint crossProduct = VectorProd(direction, edge2);
        double determinant = ScalarProd(crossProduct, edge1);
        
        if (fabs(determinant) < 1e-10) {
            continue;
        }
        
        TPoint translationVector = position - params.canvas[i].GetP1();
        TPoint crossTranslation = VectorProd(translationVector, edge1);
        
        double barycentricU = ScalarProd(crossProduct, translationVector) / determinant;
        double barycentricV = ScalarProd(crossTranslation, direction) / determinant;
        
        if ((barycentricU < 0.0 || barycentricU > 1.0) || (barycentricV < 0.0 || barycentricV + barycentricU > 1.0)) {
            continue;
        }
        
        double intersectionDistance = ScalarProd(crossTranslation, edge2) / determinant; 
        
        if (intersectionDistance < 0.0) {
            continue;
        }
        
        if (closestPolygonIndex == -1 || intersectionDistance < closestIntersection) 
        {
            closestPolygonIndex = i;
            closestIntersection = intersectionDistance;
        }
    }

    if (closestPolygonIndex == -1) {
        return make_uchar4(0, 0, 0, 255);
    }

    TPoint intersectionPoint = direction * closestIntersection + position;
    
    TPoint edge1 = params.canvas[closestPolygonIndex].GetP2() - params.canvas[closestPolygonIndex].GetP1();
    TPoint edge2 = params.canvas[closestPolygonIndex].GetP3() - params.canvas[closestPolygonIndex].GetP1();
    TPoint normalVector = VectorProd(edge1, edge2).normalize();

    TPoint reflectedDirection = direction - normalVector * (2.0 * ScalarProd(direction, normalVector));
    TPoint lightDirection = (params.lightPos - intersectionPoint).normalize();
    double diffuseIntensity = fmax(0.25, ScalarProd(normalVector, lightDirection));

    uchar4 polygonColor = params.canvas[closestPolygonIndex].GetColour();

    uchar4 directColor = make_uchar4(
        polygonColor.x * params.lightColour.x * diffuseIntensity,
        polygonColor.y * params.lightColour.y * diffuseIntensity,
        polygonColor.z * params.lightColour.z * diffuseIntensity,
        255
    );

    double reflectionCoefficient = 0.5;
    
    if (depth > 0 && reflectionCoefficient > 0.0) 
    {
        uchar4 reflectedColor = TraceRay(intersectionPoint, reflectedDirection, params, depth - 1);

        directColor.x = directColor.x * (1 - reflectionCoefficient) + reflectedColor.x * reflectionCoefficient;
        directColor.y = directColor.y * (1 - reflectionCoefficient) + reflectedColor.y * reflectionCoefficient;
        directColor.z = directColor.z * (1 - reflectionCoefficient) + reflectedColor.z * reflectionCoefficient;
    }

    return directColor;
}

#endif