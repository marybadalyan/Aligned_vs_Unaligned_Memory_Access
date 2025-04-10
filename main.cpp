#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <tuple>

// Assuming kaizen.h provides zen::cmd_args and zen::print
#include "kaizen.h"

double random_double(double min, double max) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

std::tuple<size_t, int, int, int> process_args(int argc, char* argv[]) {
    zen::cmd_args args(argv, argc);
    auto size_options = args.get_options("--size");
    auto offset_options = args.get_options("--offset");
    auto iter_options = args.get_options("--iterations");
    auto trial_options = args.get_options("--trials");

    if (size_options.empty() || offset_options.empty() || iter_options.empty()) {
        zen::print("Error: Missing arguments, using defaults: size=1000000, offset=4, iterations=100, trials=20\n");
        return {1000000, 4, 100, 20};
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0]), std::stoi(iter_options[0]), std::stoi(trial_options[0])};
}

double horizontal_sum(__m256d vec) {
    __m128d vlow = _mm256_castpd256_pd128(vec);
    __m128d vhigh = _mm256_extractf128_pd(vec, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
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
    return sum + horizontal_sum(sum_vec);
}

double sum_misaligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) sum += data[i];
    return sum + horizontal_sum(sum_vec);
}

void initialize_vector(double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = random_double(-1000.0, 1000.0);
    }
}

void flush_data(const double* ptr, size_t size) {
    for (size_t i = 0; i < size * sizeof(double); i += 64) {
        _mm_clflush(reinterpret_cast<const char*>(ptr) + i);
    }
}

void evict_cache() {
    const size_t evict_size = 32 * 1024 * 1024 / sizeof(double);
    std::vector<double> evict_buffer(evict_size);
    for (size_t i = 0; i < evict_size; ++i) {
        evict_buffer[i] = static_cast<double>(i);
    }
    volatile double sink = 0;
    for (size_t i = 0; i < evict_size; ++i) {
        sink += evict_buffer[i];
    }
    (void)sink;
}

int main(int argc, char* argv[]) {
    auto [size, offset, iterations, trials] = process_args(argc, argv);
    size_t off_bytes = (offset + sizeof(double) - 1) / sizeof(double); // Ceiling division
    size_t total_size = size + off_bytes;

    std::vector<double> aligned_times(trials), unaligned_times(trials);
    std::vector<double> data(total_size);

    for (int trial = 0; trial < trials; ++trial) {
        initialize_vector(data.data(), total_size); // Match allocation size

        double* aligned_ptr = data.data();
        char* unaligned_ptr_char = reinterpret_cast<char*>(aligned_ptr) + offset;
        double* unaligned_ptr = reinterpret_cast<double*>(unaligned_ptr_char);

        std::cout << "Trial " << trial << ":\n";
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
    }

    double avg_aligned = std::accumulate(aligned_times.begin(), aligned_times.end(), 0.0) / trials;
    double avg_unaligned = std::accumulate(unaligned_times.begin(), unaligned_times.end(), 0.0) / trials;
    double percentage = ((avg_unaligned - avg_aligned) / avg_unaligned) * 100;

    zen::print(zen::color::green(std::format("| {:<24} | {:>12.3f} ns|\n", "Average Aligned time:", avg_aligned)));
    zen::print(zen::color::red(std::format("| {:<20} | {:>12.3f} ns|\n", "Average Unaligned time:", avg_unaligned)));
    zen::print(zen::color::yellow(std::format("| {:<24} | {:>12.3f} %|\n", "Speedup Factor:", percentage)));

    return 0;
}