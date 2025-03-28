#ifndef FLOOR_CUH
#define FLOOR_CUH

#include "params.cuh"
#include "polygon.cuh"
#include <vector>

class IFloor
{
    public:
        virtual void AddFigure(std::vector<TPolygon> &canvas) = 0;
        virtual ~IFloor() {};

    protected:
        PFloor params;
};

class TFloor : public IFloor
{
    public:
        TFloor() {};

        TFloor(PFloor paramsValues){
            params = paramsValues;
        }

        void AddFigure(std::vector<TPolygon> &canvas) override
        {
            canvas.emplace_back(TPolygon(params.p1, params.p2, params.p3, params.colour));
            canvas.emplace_back(TPolygon(params.p1, params.p3, params.p4, params.colour));
        }
};

#endif