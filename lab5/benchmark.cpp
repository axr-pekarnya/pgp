#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

void GenerateRandomArray(std::vector<int>& data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % 100000; // случайные числа в диапазоне [0, 100000]
    }
}

void BitonicMerge(std::vector<int>& data, int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if ((data[i] > data[i + k]) == ascending) {
                std::swap(data[i], data[i + k]);
            }
        }
        BitonicMerge(data, low, k, ascending);
        BitonicMerge(data, low + k, k, ascending);
    }
}

void BitonicSort(std::vector<int>& data, int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        BitonicSort(data, low, k, true);
        BitonicSort(data, low + k, k, false);
        BitonicMerge(data, low, cnt, ascending);
    }
}

void RunSortingTest(int size) {
    std::vector<int> data(size);
    GenerateRandomArray(data, size);
    auto start = std::chrono::high_resolution_clock::now();
    BitonicSort(data, 0, size, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Sorting " << size << " elements took " << elapsed.count() << " seconds\n";
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    RunSortingTest(1000);
    RunSortingTest(10000);
    RunSortingTest(100000);
    RunSortingTest(1000000);
    return 0;
}
