#include <iostream>
#include <iomanip>
#include <math.h>

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    
    float a, b, c;
    std::cin >> a >> b >> c;
    
    if (a == 0 && b == 0)
    {
        if (c == 0){
            std::cout << "any" << '\n';
        }
        else {
            std::cout << "incorrect" << '\n';
        }
        
        return 0;
    }
    
    if (a == 0)
    {
        std::cout << std::fixed << std::setprecision(6) << -c / b << '\n';
        return 0;
    }
    
    float d = b * b - 4 * a * c;
    
    if (d > 0)
    {
        std::cout << std::fixed << std::setprecision(6) << (-b + sqrt(d)) / (2 * a) << ' ' << (-b - sqrt(d)) / (2 * a) << '\n';
        return 0;
    }
    
    if (d == 0)
    {
        std::cout << std::fixed << std::setprecision(6) << -b / (2 * a) << '\n';
        return 0;
    }
    
    if (d < 0)
    {
        std::cout << "imaginary" << '\n';
        return 0;
    }
}