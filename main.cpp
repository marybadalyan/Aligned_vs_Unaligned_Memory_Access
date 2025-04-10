#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <tuple>

double random_double(double min, double max) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

std::tuple<size_t, size_t, int, int> process_args(int argc, char* argv[]) {
    // Simplified for this example; assume defaults if no args
    if (argc < 5) {
        std::cout << "Using defaults: size=1000000, offset=1, iterations=100, trials=20\n";
        return {1000000, 1, 100, 20}; // offset in doubles, not bytes
    }
    return {std::stoul(argv[1]), std::stoul(argv[2]), std::stoi(argv[3]), std::stoi(argv[4])};
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

double sum_misaligned(const double* data, size_t size) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) sum += data[i];
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    return sum + result[0] + result[1] + result[2] + result[3];
}

void initialize_vector(double* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = random_double(-1000.0, 1000.0);
    }
}

void evict_cache() {
    const size_t evict_size = 32 * 1024 * 1024 / sizeof(double);
    std::vector<double> evict_buffer(evict_size);
    volatile double sink = 0;
    for (size_t i = 0; i < evict_size; ++i) {
        evict_buffer[i] = static_cast<double>(i);
        sink += evict_buffer[i];
    }
    (void)sink;
}

int main(int argc, char* argv[]) {
    auto [size, offset, iterations, trials] = process_args(argc, argv); // offset in doubles
    size_t total_size = size + offset; // Enough space for offset + data

    std::vector<double> aligned_times(trials), unaligned_times(trials);
    std::vector<double> data(total_size);
    initialize_vector(data.data(), total_size);

    for (int trial = 0; trial < trials; ++trial) {
        double* aligned_ptr = data.data();
        double* unaligned_ptr = data.data() + offset; // Shift by offset doubles

        std::cout << "Trial " << trial << ":\n";

        evict_cache();
        double aligned_sum = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            aligned_sum += sum_aligned(aligned_ptr, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        aligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Aligned sum = " << aligned_sum << "\n";

        evict_cache();
        double unaligned_sum = 0;
        start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < iterations; ++i) {
            unaligned_sum += sum_misaligned(unaligned_ptr, size);
        }
        end = std::chrono::high_resolution_clock::now();
        unaligned_times[trial] = std::chrono::duration<double, std::nano>(end - start).count() / iterations;
        std::cout << "  Unaligned sum = " << unaligned_sum << "\n";
    }

    double avg_aligned = std::accumulate(aligned_times.begin(), aligned_times.end(), 0.0) / trials;
    double avg_unaligned = std::accumulate(unaligned_times.begin(), unaligned_times.end(), 0.0) / trials;
    double percentage = ((avg_unaligned - avg_aligned) / avg_unaligned) * 100;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "| Average Aligned time:   | " << avg_aligned << " ns |\n";
    std::cout << "| Average Unaligned time: | " << avg_unaligned << " ns |\n";
    std::cout << "| Speedup Factor:         | " << percentage << " % |\n";

    return 0;
}