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
        return {10000000, 3};  // Default size of 1 million elements
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0])};
}

// Aligned allocator
template<typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U>&) {}

    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 8); // 8-byte aligned malloc
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) {
        _mm_free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U>;
    };
};

template<typename T>
struct MissAlignedAllocator {
    using value_type = T;

    MissAlignedAllocator() = default;

    template <typename U>
    MissAlignedAllocator(const MissAlignedAllocator<U>&) {}

    T* allocate(std::size_t n) {
        void* raw = _mm_malloc(n * sizeof(T) + 8, 8); // Overallocate
        if (!raw) throw std::bad_alloc();

        // Offset by 3 bytes to force misalignment
        void* misaligned_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(raw) + 3);
        return reinterpret_cast<T*>(misaligned_ptr);
    }

    void deallocate(T* ptr, std::size_t) {
        // Recover the actual pointer before free
        void* raw_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) - 3);
        _mm_free(raw_ptr);
    }

    template <typename U>
    struct rebind {
        using other = MissAlignedAllocator<U>;
    };
};


template <typename Allocator>
double sum(int size, std::vector<double, Allocator>& container) {
    __m256d sum_vec = _mm256_setzero_pd();  // Initialize sum to 0

    // SIMD (AVX256) for 8 doubles at a time
    for (int i = 0; i + 7 < size / 8; i += 8) {
        __m256d vec = _mm256_load_pd(container.data() + i);  // Load 8 doubles
        sum_vec = _mm256_add_pd(sum_vec, vec);  // SIMD addition
    }

    // Handle any remaining elements that couldn't be processed in the loop
    double sum = 0;
    for (int i = size - size % 8; i < size; ++i) {
        sum += container[i];  // Process the remaining elements one by one
    }

    // Reduce the SIMD result to a scalar sum
    double result[8];
    _mm256_storeu_pd(result, sum_vec);  // Store the result in an array

    // Sum the elements in the result array
    for (int i = 0; i < 8; ++i) {
        sum += result[i];
    }

    return sum;
}


int main(int argc, char* argv[]) {
    auto [size, offset] = process_args(argc, argv);  // Get user-specified size and offset
    
    zen::timer timer;

    // Aligned Data
    timer.start();
    std::vector<double, AlignedAllocator<double>> alignedData(size, 0);  // size is in terms of elements
    std::fill(alignedData.begin(), alignedData.end(), zen::random_int(0, 10000));
    sum(size, alignedData);
    timer.stop();
    double alignedDataTime = timer.duration<zen::timer::nsec>().count();  // Fix variable name conflict
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.6f} | {:<9} |\n", "Aligned data SIMD", alignedDataTime, "ns")));

    // Misaligned Data (using MissAlignedAllocator)
    timer.start();
    std::vector<double, MissAlignedAllocator<double>> misalignedData(size, 0);  // Ensure correct allocator
    std::fill(misalignedData.begin(), misalignedData.end(), zen::random_int(0, 10000));
    sum(size, misalignedData);
    timer.stop();
    double misalignedDataTime = timer.duration<zen::timer::nsec>().count();  // Fix variable name conflict
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.6f} | {:<9} |\n", "Misaligned data SIMD", misalignedDataTime, "ns")));
}
