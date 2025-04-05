#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <cassert>
#include "kaizen.h"


double random_double(double min, double max) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Parse command-line arguments
std::tuple<size_t, int, int> process_args(int argc, char* argv[]) {
    zen::cmd_args args(argv, argc);
    auto size_options = args.get_options("--size");
    auto offset_options = args.get_options("--offset");
    auto iter_options = args.get_options("--iterations");

    if (size_options.empty() || offset_options.empty() || iter_options.empty()) {
        zen::print("Error: --size, --offset, or --iterations arguments are absent, using defaults: size=1000000, offset=4, iterations=100\n");
        return {100000, 4, 1000}; // Increased defaults
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0]), std::stoi(iter_options[0])};
}

// Aligned allocator (8-byte alignment for AVX)
template<typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 8); // 8-byte aligned for AVX
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { _mm_free(ptr); }

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U>;
    };
};


// Aligned sum with AVX
double sum_aligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_load_pd(data + i); // Aligned load
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) sum += data[i];
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    return sum + result[0] + result[1] + result[2] + result[3];
}


// Misaligned sum with AVX
double sum_misaligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(data + i); // Unaligned load
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) sum += data[i];
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    return sum + result[0] + result[1] + result[2] + result[3];
}

// Initialize vector
void initialize_vector(double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = random_double(-100.0, 100.0);
    }
}


int main(int argc, char* argv[]) {
    auto [size, offset, iterations] = process_args(argc, argv);
    offset = (offset / sizeof(double)) * sizeof(double); // Ensure offset aligns with double

    // Allocate enough space for size + offset in doubles
    size_t total_size = size + (offset / sizeof(double)) + 1;
    std::vector<double, AlignedAllocator<double>> data(total_size);
    initialize_vector(data.data(), total_size);

    double* aligned_ptr = data.data();
    double* unaligned_ptr = reinterpret_cast<double*>(reinterpret_cast<char*>(data.data()) + offset);

    // Verify bounds
    if (unaligned_ptr + size > data.data() + total_size) {
        std::cerr << "Error: Unaligned pointer exceeds buffer!\n";
        return 1;
    }

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        sum_aligned(aligned_ptr, size);
        sum_misaligned(unaligned_ptr, size);
    }

    // Measure aligned
    double aligned_time = 0;
    double aligned_sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        aligned_sum += sum_aligned(aligned_ptr, size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    aligned_time = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
        "Aligned sum: ", aligned_sum, "ns")));

    // Measure unaligned
    double unaligned_time = 0;
    double unaligned_sum = 0; // Use double instead of long double for consistency
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        unaligned_sum += sum_misaligned(unaligned_ptr, size);
    }
    end = std::chrono::high_resolution_clock::now();
    unaligned_time = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
        "Unaligned sum: ", unaligned_sum, "ns")));

    std::cout << std::fixed << std::setprecision(3);
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
        "Aligned time: ", aligned_time, "ns")));
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
        "Unaligned time: ", unaligned_time, "ns")));
    zen::print(zen::color::yellow(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
            "SpeedupFactor ", aligned_time/unaligned_time, "ns")));

    zen::print("Addresses - Aligned: 0x" , std::hex , reinterpret_cast<uintptr_t>(aligned_ptr)
              , ", Unaligned: 0x" , reinterpret_cast<uintptr_t>(unaligned_ptr) , std::dec , "\n");

    return 0;
}