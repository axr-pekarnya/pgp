#include "scene.cuh"
#include <string>
#include <sstream>

int main(int argc, char* argv[]) 
{
    bool gpu = false;
    bool cpu = false;
    bool demo = false;
    
    for (int i = 1; i < argc; ++i) 
    {        
        if (std::string(argv[i]) == "--cpu") {
            cpu = true;
        } 
        
        if (std::string(argv[i]) == "--gpu") {
            gpu = true;
        } 
        
        if (std::string(argv[i]) == "--default"){
            demo = true;
        }
    }
    
    std::istream *fin;
    std::istringstream demoStream;
    
    if (demo)
    {
        gpu = true;
    
        std::string demoInput = 
            "24\n"
            "data-bin/%d.data\n"
            "640 480 90\n"
            "7 2 -0.5   1   1     1 1 1   0 0\n"
            "2 0 0      0.5 0.1   1 1 1   0 0\n"
            "2 -3 0   0 1 0   1\n"
            "0 0 1    0 0 1   1\n"
            "-2 3 0   1 0 0   1\n"
            "-4 -4 -1 -4 4 -1 4 4 -1 4 -4 -1 1 1 1\n"
            "10 0 15 0.294118 0.196078 0.0980392 4\n";
        
        std::cout << demoInput << '\n';
        
        demoStream.str(demoInput);
        fin = &demoStream;
    }
    else {
        fin = &std::cin;
    }

    PFrame frameParams;
    PCamera cameraParams;
    PFigure tetrahedronParams;
    PFigure hexahedronParams;
    PFigure icosahedronParams;
    PFloor floorParams;
    PLight lightParams;

    *fin >> frameParams;
    *fin >> cameraParams;
    *fin >> tetrahedronParams;
    *fin >> hexahedronParams;
    *fin >> icosahedronParams;
    *fin >> floorParams;
    *fin >> lightParams;
    
    TCamera camera(cameraParams);
    TScene scene = TScene(camera, frameParams, tetrahedronParams, hexahedronParams, icosahedronParams, floorParams, lightParams);
        
    scene.AddFigures();
    
    if (gpu || !cpu){
        scene.GPU();
    }
    else {
        scene.CPU();
    }
    
    scene.SaveData();
        
    return 0;
}