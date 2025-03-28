#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    
    int n;
    std::cin >> n;
    
    std::vector<float> data(n);
    
    for (int i = 0; i < n; ++i){
        std::cin >> data[i];
    }
    
    std::sort(data.begin(), data.end());
    
    for (int i = 0; i < n; ++i){
        std::cout << std::fixed << std::setprecision(6) << data[i] << ' ';
    }
    std::cout << '\n';
    
    return 0;
}