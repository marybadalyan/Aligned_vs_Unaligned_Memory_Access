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
        return {1000000, 46, 100, 20}; // Increased defaults
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0]), std::stoi(iter_options[0]), std::stoi(trial_options[0])};
}


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

double sum_misaligned(const double* data, size_t size){
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
        data[i] = random_double(-1000.0, 1000.0);
    }
}

void flush_data(const double* ptr, size_t size) {
    for (size_t i = 0; i < size * sizeof(double); i += 64) { // 64-byte cache line
        _mm_clflush(reinterpret_cast<const char*>(ptr) + i);
    }
}


void evict_cache() {
    const size_t evict_size = 32 * 1024 * 1024 / sizeof(double); // 32 MB, larger than most L3 caches
    std::vector<double> evict_buffer(evict_size);
    for (size_t i = 0; i < evict_size; ++i) {
        evict_buffer[i] = static_cast<double>(i);  
    }
    volatile double sink = 0;
    for (size_t i = 0; i < evict_size; ++i) {
        sink += evict_buffer[i]; 
    }
    (void)sink; // Prevent optimization
}

int main(int argc, char* argv[]) {
    auto [size, offset, iterations, trials] = process_args(argc, argv); // offset in bytes

    // Calculate total size to accommodate offset
    size_t extra_bytes = (offset + sizeof(double) - 1) / sizeof(double); // Ceiling division
    size_t total_size = size + extra_bytes;

    // Allocate with 8-byte alignment using _mm_malloc
    double* raw_data = static_cast<double*>(_mm_malloc(total_size * sizeof(double), 8));
    if (!raw_data) {
        std::cerr << "Memory allocation failed\n";
        return 1;
    }
    if (reinterpret_cast<uintptr_t>(raw_data) % 8 != 0) {
        std::cerr << "Aligned allocation failed\n";
        _mm_free(raw_data);
        return 1;
    }

    std::vector<double> aligned_times(trials), unaligned_times(trials);

    for (int trial = 0; trial < trials; ++trial) {
        initialize_vector(raw_data, total_size);

        double* aligned_ptr = raw_data;
        char* offset_bytes = reinterpret_cast<char*>(aligned_ptr) + offset;
        double* unaligned_ptr = reinterpret_cast<double*>(offset_bytes);

        // Verify alignment
        std::cout << "Trial " << trial << ": Aligned ptr = " << aligned_ptr
                  << ", Unaligned ptr = " << unaligned_ptr << "\n";
        std::cout << "Aligned mod 8 = " << (reinterpret_cast<uintptr_t>(aligned_ptr) % 8)
                  << ", Unaligned mod 8 = " << (reinterpret_cast<uintptr_t>(unaligned_ptr) % 8) << "\n";

        flush_data(aligned_ptr, size);
        evict_cache();
        double aligned_sum = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            aligned_sum += sum_aligned(aligned_ptr, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        aligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Aligned sum = " << aligned_sum << "\n";

        flush_data(unaligned_ptr, size);
        evict_cache();
        double unaligned_sum = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            unaligned_sum += sum_misaligned(unaligned_ptr, size);
        }
        end = std::chrono::high_resolution_clock::now();
        unaligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Unaligned sum = " << unaligned_sum << "\n";

        // Verify sums match
        double baseline = std::accumulate(aligned_ptr, aligned_ptr + size, 0.0);
        if (std::abs(baseline - aligned_sum) > 1e-10 || std::abs(baseline - unaligned_sum) > 1e-10) {
            std::cout << "Sum mismatch: baseline = " << baseline << "\n";
        }
    }

    double avg_aligned = std::accumulate(aligned_times.begin(), aligned_times.end(), 0.0) / trials;
    double avg_unaligned = std::accumulate(unaligned_times.begin(), unaligned_times.end(), 0.0) / trials;
    double percentage = ((avg_unaligned - avg_aligned) / avg_unaligned) * 100;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "| Average Aligned time:   | " << avg_aligned << " ns |\n";
    std::cout << "| Average Unaligned time: | " << avg_unaligned << " ns |\n";
    std::cout << "| Speedup Factor:         | " << percentage << " % |\n";

    _mm_free(raw_data);
    return 0;
}