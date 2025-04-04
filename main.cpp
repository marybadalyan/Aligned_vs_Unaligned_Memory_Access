#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h> // AVX2
#include <memory>
#include <chrono>
#include <iomanip>
#include <tuple>
#include "kaizen.h" // Assuming this provides zen::timer, zen::print, zen::color, zen::cmd_args
#include <string>
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
        zen::print("Error: --size, --offset, or --iterations arguments are absent, using defaults: size=1000000, offset=1, iterations=100\n");
        return {1000000, 3, 1000}; // Increased defaults
    }
}

// Aligned allocator (8-byte alignment)
template<typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 8); // 8-byte aligned
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { _mm_free(ptr); }

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U>;
    };
};

// Aligned sum with AVX2
double sum_aligned(size_t size, std::vector<double, AlignedAllocator<double>>& container, int repeats = 10) {
    double total_sum = 0;
    for (int r = 0; r < repeats; ++r) { // Repeat to amplify workload
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256d vec = _mm256_load_pd(container.data() + i); // Aligned load
            sum_vec = _mm256_add_pd(sum_vec, vec);
        }
        double sum = 0;
        for (; i < size; ++i) {
            sum += container[i];
        }
        double result[8];
        _mm256_store_pd(result, sum_vec);
        for (int j = 0; j < 8; ++j) {
            sum += result[j];
        }
        total_sum += sum;
    }
    return total_sum / repeats; // Average to avoid overflow
}

// Misaligned sum with AVX2
double sum_misaligned(size_t size, std::vector<double, AlignedAllocator<double>>& container, int offset, int repeats = 10) {
    double total_sum = 0;
    for (int r = 0; r < repeats; ++r) {
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 7 < size - offset; i += 8) { // Ensure no out-of-bounds access
            __m256d vec = _mm256_loadu_pd(container.data() + i + offset); // Forced misalignment
            sum_vec = _mm256_add_pd(sum_vec, vec);
        }
        double sum = 0;
        for (; i < size - offset; ++i) {
            sum += container[i + offset];
        }
        double result[8];
        _mm256_storeu_pd(result, sum_vec);
        for (int j = 0; j < 8; ++j) {
            sum += result[j];
        }
        total_sum += sum;
    }
    return total_sum / repeats; // Average to avoid overflow
}

// Warm-up function
template<typename Allocator>
void warm_up(std::vector<double, Allocator>& container) {
    volatile double sum = 0; // Prevent optimization
    for (size_t i = 0; i < container.size(); ++i) {
        sum += container[i];
    }
}

int main(int argc, char* argv[]) {
    auto [size, offset, iter] = process_args(argc, argv);
    if (offset >= size) {
        std::cerr << "Offset (" << offset << ") must be less than size (" << size << "), adjusting to 1\n";
        offset = 1;
    }
    zen::timer timer;

    // Aligned data (both use AlignedAllocator)
    std::vector<double, AlignedAllocator<double>> data(size);
    for (auto& val : data) {
        val = random_double(0.0, 10000.0);
    }
    warm_up(data);

    // Aligned access
    double aligned_time_total = 0;
    double aligned_sum_result = 0;
    for (int r = 0; r < iter; ++r) {
        timer.start();
        aligned_sum_result = sum_aligned(size, data);
        timer.stop();
        aligned_time_total += timer.duration<zen::timer::nsec>().count();
    }
    double aligned_time_avg = aligned_time_total / iter;
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Aligned data SIMD", aligned_time_avg, "ns")));

    // Misaligned access
    double misaligned_time_total = 0;
    double misaligned_sum_result = 0;
    for (int r = 0; r < iter; ++r) {
        timer.start();
        misaligned_sum_result = sum_misaligned(size, data, offset);
        timer.stop();
        misaligned_time_total += timer.duration<zen::timer::nsec>().count();
    }
    double misaligned_time_avg = misaligned_time_total / iter;
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Misaligned access SIMD", misaligned_time_avg, "ns")));

    // Verify results
    zen::print("Aligned sum: " , aligned_sum_result , ", Misaligned sum: " , misaligned_sum_result , "\n");
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(data.data());
    uintptr_t misaligned_addr = reinterpret_cast<uintptr_t>(data.data() + offset);
    zen::print("Addresses - Base: 0x" , std::hex , base_addr 
              , ", Misaligned access: 0x" , misaligned_addr , std::dec , "\n");
    double speedup = aligned_time_avg/misaligned_time_avg;
    std::string speedupFactor = std::to_string(speedup);

    zen::print(zen::color::yellow("Aligned vs Misaligned speedup factor: "));
    zen::print(zen::color::yellow(speedupFactor));
    zen::print(zen::color::yellow("\n"));


   

    return 0;
}