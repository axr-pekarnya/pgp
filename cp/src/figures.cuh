#ifndef FIGURES_CUH
#define FIGURES_CUH

#include "params.cuh"
#include "polygon.cuh"
#include <vector>

class IFigure
{
public:
    virtual void AddFigure(std::vector<TPolygon>& canvas) = 0;
    virtual ~IFigure() = default;

protected:
    PFigure params;
};


class TTetrahedron : public IFigure
{
public:
    explicit TTetrahedron(PFigure paramsValues) { params = paramsValues; }

    void AddFigure(std::vector<TPolygon>& canvas) override
    {
        double rad = 4 / sqrt(6) * params.radius;
        std::vector<TPoint> v = {
            {params.center.GetX(), params.center.GetY() + params.radius, params.center.GetZ()},
            {params.center.GetX() + 0.6 * rad, params.center.GetY() - params.radius / 3, params.center.GetZ()},
            {params.center.GetX() - 0.3 * rad, params.center.GetY() - params.radius / 3, params.center.GetZ() + 0.5 * rad},
            {params.center.GetX() - 0.3 * rad, params.center.GetY() - params.radius / 3, params.center.GetZ() - 0.5 * rad}
        };

        int faces[4][3] = {
            {0, 1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 3}
        };

        for (const auto& face : faces) {
            canvas.emplace_back(v[face[0]], v[face[1]], v[face[2]], params.colour);
        }
    }
};


class THexahedron : public IFigure
{
public:
    explicit THexahedron(PFigure paramsValues) { params = paramsValues; }

    void AddFigure(std::vector<TPolygon>& canvas) override
    {
        double a = params.radius * 2;
        std::vector<TPoint> v = {
            {params.center.GetX() - a / 2, params.center.GetY() - a / 2, params.center.GetZ() - a / 2},
            {params.center.GetX() + a / 2, params.center.GetY() - a / 2, params.center.GetZ() - a / 2},
            {params.center.GetX() + a / 2, params.center.GetY() + a / 2, params.center.GetZ() - a / 2},
            {params.center.GetX() - a / 2, params.center.GetY() + a / 2, params.center.GetZ() - a / 2},
            {params.center.GetX() - a / 2, params.center.GetY() - a / 2, params.center.GetZ() + a / 2},
            {params.center.GetX() + a / 2, params.center.GetY() - a / 2, params.center.GetZ() + a / 2},
            {params.center.GetX() + a / 2, params.center.GetY() + a / 2, params.center.GetZ() + a / 2},
            {params.center.GetX() - a / 2, params.center.GetY() + a / 2, params.center.GetZ() + a / 2}
        };

        int faces[6][4] = {
            {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
            {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 3, 7, 4}
        };

        for (const auto& face : faces) 
        {
            canvas.emplace_back(v[face[0]], v[face[1]], v[face[2]], params.colour);
            canvas.emplace_back(v[face[0]], v[face[2]], v[face[3]], params.colour);
        }
    }
};


class TIcosahedron : public IFigure
{
public:
    explicit TIcosahedron(PFigure paramsValues) { params = paramsValues; }

    void AddFigure(std::vector<TPolygon>& canvas) override
    {
        double phi = (1.0 + sqrt(5.0)) / 2.0;
        double scale = params.radius / sqrt(1 + phi * phi);
        
        std::vector<TPoint> v = {
            {params.center.GetX(), params.center.GetY() - scale, params.center.GetZ() - phi * scale},
            {params.center.GetX(), params.center.GetY() - scale, params.center.GetZ() + phi * scale},
            {params.center.GetX(), params.center.GetY() + scale, params.center.GetZ() - phi * scale},
            {params.center.GetX(), params.center.GetY() + scale, params.center.GetZ() + phi * scale},
            {params.center.GetX() - scale, params.center.GetY() - phi * scale, params.center.GetZ()},
            {params.center.GetX() - scale, params.center.GetY() + phi * scale, params.center.GetZ()},
            {params.center.GetX() + scale, params.center.GetY() - phi * scale, params.center.GetZ()},
            {params.center.GetX() + scale, params.center.GetY() + phi * scale, params.center.GetZ()},
            {params.center.GetX() - phi * scale, params.center.GetY(), params.center.GetZ() - scale},
            {params.center.GetX() - phi * scale, params.center.GetY(), params.center.GetZ() + scale},
            {params.center.GetX() + phi * scale, params.center.GetY(), params.center.GetZ() - scale},
            {params.center.GetX() + phi * scale, params.center.GetY(), params.center.GetZ() + scale}
        };

        int faces[20][3] = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        for (const auto& face : faces) {
            canvas.emplace_back(v[face[0]], v[face[1]], v[face[2]], params.colour);
        }
    }
};

#endif