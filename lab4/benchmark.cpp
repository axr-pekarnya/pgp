#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>

void SwapLines(std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& unitedMatrix, int i, int j) {
    std::swap(matrix[i], matrix[j]);
    std::swap(unitedMatrix[i], unitedMatrix[j]);
}

void Divide(std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& unitedMatrix, int n) {
    for (int i = 0; i < n; i++) {
        double divisor = matrix[i][i];
        for (int j = 0; j < n; j++) {
            unitedMatrix[i][j] /= divisor;
        }
    }
}

void DelLower(std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& unitedMatrix, int n, int sep) {
    for (int i = sep + 1; i < n; i++) {
        double div = -matrix[i][sep] / matrix[sep][sep];
        for (int j = sep + 1; j < n; j++) {
            matrix[i][j] += div * matrix[sep][j];
        }
        for (int j = 0; j < n; j++) {
            unitedMatrix[i][j] += div * unitedMatrix[sep][j];
        }
    }
}

void DelUpper(std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& unitedMatrix, int n, int sep) {
    for (int i = sep - 1; i >= 0; i--) {
        double div = -matrix[i][sep] / matrix[sep][sep];
        for (int j = 0; j < n; j++) {
            unitedMatrix[i][j] += div * unitedMatrix[sep][j];
        }
    }
}

std::vector<std::vector<double>> GenerateMatrix(int n) {
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
    return matrix;
}

void InvertMatrix(int n) {
    auto matrix = GenerateMatrix(n);
    std::vector<std::vector<double>> unitedMatrix(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        unitedMatrix[i][i] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n - 1; ++i) {
        int maxIdx = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(matrix[k][i]) > std::abs(matrix[maxIdx][i])) {
                maxIdx = k;
            }
        }
        if (maxIdx != i) {
            SwapLines(matrix, unitedMatrix, i, maxIdx);
        }
        DelLower(matrix, unitedMatrix, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        DelUpper(matrix, unitedMatrix, n, i);
    }

    Divide(matrix, unitedMatrix, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "Matrix size: " << n << "x" << n << " - Execution time: " << duration.count() << " ms" << std::endl;
}

int main() {
    InvertMatrix(100);
    InvertMatrix(500);
    InvertMatrix(1000);
    return 0;
}
