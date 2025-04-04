#include <iostream>
#include <cstring>
#include <utility>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <format>
#include "kaizen.h"
#include <iomanip>
#include <numeric>
#include <immintrin.h> // AVX2
#include <memory>


// Parse command-line arguments
std::pair<size_t,int> process_args(int argc, char* argv[]) {
    zen::cmd_args args(argv, argc);
    auto size_options = args.get_options("--size");
    auto offset_options = args.get_options("--offset");

    if (offset_options.empty() || size_options.empty()) {
        zen::print("Error: --size/--offset arguments are absent, using default", 1000000, " and ", 3, '\n');
        return {1000000, 3};  // Default size of 1 million elements
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0])};
}
// Aligned allocator (32-byte for AVX2)
template<typename T>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 32); // 32-byte aligned for AVX2
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) { _mm_free(ptr); }
};

// Misaligned allocator
template<typename T>
struct MisalignedAllocator {
    using value_type = T;
    int offset;
    MisalignedAllocator(int off = 3) : offset(off) {}
    T* allocate(std::size_t n) {
        void* raw = _mm_malloc(n * sizeof(T) + 32, 32); // Overallocate
        if (!raw) throw std::bad_alloc();
        return reinterpret_cast<T*>(static_cast<char*>(raw) + offset); // Misalign by offset
    }
    void deallocate(T* ptr, std::size_t) {
        void* raw = static_cast<char*>(ptr) - offset;
        _mm_free(raw);
    }
};

// Sum function with AVX2
template<typename Allocator>
double sum(size_t size, std::vector<double, Allocator>& container) {
    __m256d sum_vec = _mm256_setzero_pd(); // 4 doubles

    // Process 4 doubles at a time with AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(container.data() + i); // Unaligned load
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }

    // Handle remainder
    double sum = 0;
    for (; i < size; ++i) {
        sum += container[i];
    }

    // Reduce SIMD vector to scalar
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    for (int j = 0; j < 4; ++j) {
        sum += result[j];
    }
    return sum;
}

// Warm-up function
template <typename Allocator>
void warmUp(std::vector<double, Allocator>& container){
    double sum = 0;
    for(int i = 0; i < container.size();++i){
        sum += container[i];
    }

}
int main(int argc, char* argv[]) {
    auto [size, offset] = process_args(argc, argv);  // Get user-specified size and offset
    
    zen::timer timer;

    // Aligned Data
    std::vector<double, AlignedAllocator<double>> alignedData(size, 0);  // size is in terms of elements

    std::fill(alignedData.begin(), alignedData.end(), zen::random_int(0, 10000));
    warmUp(alignedData);

    timer.start();
    sum(size, alignedData);
    timer.stop();
    double alignedDataTime = timer.duration<zen::timer::nsec>().count();  // Fix variable name conflict
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.6f} | {:<9} |\n", "Aligned data SIMD", alignedDataTime, "ns")));

    std::vector<double, MissAlignedAllocator<double>> misalignedData(size, 0);  // Ensure correct allocator
    std::fill(misalignedData.begin(), misalignedData.end(), zen::random_int(0, 10000));
    warmUp(misalignedData);

    // Misaligned Data (using MissAlignedAllocator)
    timer.start();
    sum(size, misalignedData);
    timer.stop();
    double misalignedDataTime = timer.duration<zen::timer::nsec>().count();  // Fix variable name conflict
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.6f} | {:<9} |\n", "Misaligned data SIMD", misalignedDataTime, "ns")));
}
