#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <cstring> 

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
        zen::print("Error: --size, --offset, --iterations or --trials arguments are absent, using defaults: size=1000000, offset=32, iterations=1000, trials=6\n");
        return {10000000, 14, 10, 3}; // sensible defaults
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

double sum_misaligned(const double* vec, std::size_t count) {
    __m256d sum = _mm256_setzero_pd();
    std::size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        __m256d chunk = _mm256_loadu_pd(vec + i); // unaligned load
        sum = _mm256_add_pd(sum, chunk);
    }

    double temp[4];
    _mm256_storeu_pd(temp, sum);
    double total = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < count; ++i)
        total += vec[i];

    return total;
}

void initialize_vector(double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = random_double(-100.0, 100.0);
    }
}

void flush_data(const double* ptr, size_t size) {
    for (size_t i = 0; i < size * sizeof(double); i += 64) {
        _mm_clflush(reinterpret_cast<const char*>(ptr) + i);
    }
}


int main(int argc, char* argv[]) {
    auto [size, offset, iterations, trials] = process_args(argc, argv);
    std::vector<double> aligned_times(trials), unaligned_times(trials);

    for (int trial = 0; trial < trials; ++trial) {
        std::vector<double> data(size);
        initialize_vector(data.data(), size);
    
        const double* aligned_ptr = data.data();
    
        void* raw = std::malloc(size * sizeof(double) + 64);
        uint8_t* bytes = reinterpret_cast<uint8_t*>(raw);
        double* unaligned_ptr = reinterpret_cast<double*>(bytes + offset);
    
        std::memcpy(unaligned_ptr, aligned_ptr, size * sizeof(double));
    
        std::cout << "Trial " << trial << ":\n";
    
        // Flush before aligned sum  
        _mm_mfence();
        flush_data(aligned_ptr, size);
        flush_data(unaligned_ptr, size); // ensure no overlap cache reuse
        _mm_mfence();
    
        double aligned_sum = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            aligned_sum += sum_aligned(aligned_ptr, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        aligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Aligned sum   = " << aligned_sum << "\n";
    
        // Flush before unaligned sum  
        _mm_mfence();
        flush_data(aligned_ptr, size);
        flush_data(unaligned_ptr, size); // again to avoid cache overlap
        _mm_mfence();
    
        double unaligned_sum = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            unaligned_sum += sum_misaligned(unaligned_ptr, size);
        }
        end = std::chrono::high_resolution_clock::now();
        unaligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Unaligned sum = " << unaligned_sum << "\n";
    
        std::free(raw);
    }
    

    double avg_aligned = std::accumulate(aligned_times.begin(), aligned_times.end(), 0.0) / trials;
    double avg_unaligned = std::accumulate(unaligned_times.begin(), unaligned_times.end(), 0.0) / trials;
    double speedUP_factor = ((avg_unaligned - avg_aligned)/avg_unaligned)*100;

    zen::print(zen::color::green(std::format("| {:<24} | {:>12.3f} ns |\n", "Average Aligned time:", avg_aligned)));
    zen::print(zen::color::red(std::format("| {:<24} | {:>12.3f} ns |\n", "Average Unaligned time:", avg_unaligned)));
    zen::print(zen::color::yellow(std::format("| {:<24} | {:>12.3f}   %|\n", "Speedup Percentage:", speedUP_factor)));

    return 0;
}
