#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>

const int SIZES[3] = {100, 100000, 10000000};

void MultiplyVectors(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res, int size) {
    for (int i = 0; i < size; ++i) {
        res[i] = a[i] * b[i];
    }
}

void PrintVector(const std::vector<double>& vec, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(10) << vec[i] << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int s = 0; s < 3; ++s) {
        int n = SIZES[s];
        std::vector<double> a(n);
        std::vector<double> b(n);
        std::vector<double> res(n);

        for (int i = 0; i < n; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        MultiplyVectors(a, b, res, n);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Execution time for n=" << n << ": " << elapsed.count() << " seconds" << std::endl;
    }
    
    return 0;
}
