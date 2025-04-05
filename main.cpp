#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include "kaizen.h"
#include <tuple>
double random_double(double min, double max) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}
// Parse command-line arguments
std::tuple<size_t, int, int, int> process_args(int argc, char* argv[]) {
    zen::cmd_args args(argv, argc);
    auto size_options = args.get_options("--size");
    auto offset_options = args.get_options("--offset");
    auto iter_options = args.get_options("--iterations");
    auto trial_options = args.get_options("--trials");

    if (size_options.empty() || offset_options.empty() || iter_options.empty()) {
        zen::print("Error: --size, --offset, , --iterations or --trials arguments are absent, using defaults: size=1000000, offset=4, iterations=100\n");
        return {100000, 4, 1000,5}; // Increased defaults
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0]), std::stoi(iter_options[0]), std::stoi(trial_options[0])};
}

template<typename T>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 32);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { _mm_free(ptr); }
    template<typename U> struct rebind { using other = AlignedAllocator<U>; };
};

double sum_aligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_load_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) sum += data[i];
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    return sum + result[0] + result[1] + result[2] + result[3];
}

double sum_misaligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) {
        sum += data[i];
    }
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    return sum + result[0] + result[1] + result[2] + result[3];
}

void initialize_vector(double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = random_double(-100.0, 100.0);
    }
}

void flush_data(const double* ptr, size_t size) {
    for (size_t i = 0; i < size * sizeof(double); i += 64) { // 64-byte cache line
        _mm_clflush(reinterpret_cast<const char*>(ptr) + i);
    }
}

int main(int argc, char* argv[]) {
    auto [size,offset,iterations,trials] = process_args(argc,argv);
    size_t extra = (offset + sizeof(double) - 1) / sizeof(double);  // Ceiling division
    size_t total_size = size + extra;

    std::vector<double, AlignedAllocator<double>> data(total_size);
    initialize_vector(data.data(), total_size);

    double* aligned_ptr = data.data();
    double* unaligned_ptr = aligned_ptr + extra;

    std::vector<double> aligned_times(trials), unaligned_times(trials);

    for (int trial = 0; trial < trials; ++trial) {
        std::cout << "Trial " << trial << ":\n";

        // Warm-up (optional, can remove if cache eviction is sufficient)
        for (int i = 0; i < 5; ++i) {
            sum_aligned(aligned_ptr, size);
            sum_misaligned(unaligned_ptr, size);
        }

        // Measure aligned
        flush_data(aligned_ptr,total_size); // Ensure cache is cold
        double aligned_sum = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            aligned_sum += sum_aligned(aligned_ptr, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        aligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Aligned sum = " << aligned_sum << "\n";

        // Evict cache before misaligned
        flush_data(unaligned_ptr,total_size); // Ensure cache is cold
        double unaligned_sum = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            unaligned_sum += sum_misaligned(unaligned_ptr, size);
        }
        end = std::chrono::high_resolution_clock::now();
        unaligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Unaligned sum = " << unaligned_sum << "\n";
    }

    // Compute averages
    double avg_aligned = std::accumulate(aligned_times.begin(), aligned_times.end(), 0.0) / trials;
    double avg_unaligned = std::accumulate(unaligned_times.begin(), unaligned_times.end(), 0.0) / trials;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average Aligned time: " << avg_aligned << " ms\n";
    std::cout << "Average Unaligned time: " << avg_unaligned << " ms\n";
    std::cout << "Performance ratio (unaligned/aligned): " << avg_unaligned / avg_aligned << "\n";

    return 0;
}