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
#pragma pack(1)

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
        return {100000, 25, 100}; // Increased defaults
    }
    return {std::stoi(size_options[0]), std::stoi(offset_options[0]), std::stoi(iter_options[0])};
}

// Aligned allocator (32-byte alignment for AVX)
template<typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = _mm_malloc(n * sizeof(T), 32); // 32-byte aligned for AVX
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
template<typename Allocator>
double sum_aligned(size_t size, std::vector<double, Allocator>& container, int repeats = 1) {
    volatile  double total_sum = 0;
    for (int r = 0; r < repeats; ++r) {
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_load_pd(container.data() + i); // Aligned load
            sum_vec = _mm256_add_pd(sum_vec, vec);
            _mm_clflush(container.data() + i); // Evict immediately
            _mm_mfence(); // Ensure flush completes
        }
        double sum = 0;
        for (; i < size; ++i) {
            sum += container[i];
            _mm_clflush(&container[i]); // Evict scalar access
            _mm_mfence();
        }
        double result[4];
        _mm256_storeu_pd(result, sum_vec);
        for (int j = 0; j < 4; ++j) sum += result[j];
        total_sum += sum;
    }
    return total_sum / repeats;
}

// Misaligned sum with AVX2
template<typename Allocator>
double sum_misaligned(size_t size, std::vector<double, Allocator>& container, int byte_offset, int repeats = 1) {
    volatile double total_sum = 0;
    double* base = container.data();
    double* misaligned_ptr = reinterpret_cast<double*>(
        reinterpret_cast<char*>(base) + byte_offset);
    size_t adjusted_size = size - (byte_offset / sizeof(double));
    for (int r = 0; r < repeats; ++r) {
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 3 < adjusted_size; i += 4) {
            __m256d vec = _mm256_loadu_pd(misaligned_ptr + i); // Misaligned load
            sum_vec = _mm256_add_pd(sum_vec, vec);
            _mm_clflush(misaligned_ptr + i); // Evict immediately
            _mm_mfence(); // Ensure flush completes
        }
        double sum = 0;
        for (; i < adjusted_size; ++i) {
            sum += misaligned_ptr[i];
            _mm_clflush(&misaligned_ptr[i]); // Evict scalar access
            _mm_mfence();
        }
        double result[4];
        _mm256_storeu_pd(result, sum_vec);
        for (int j = 0; j < 4; ++j) sum += result[j];
        total_sum += sum;
    }
    return total_sum / repeats;
}
// Initialize vector with non-temporal stores
template<typename Allocator>
void initialize_vector(std::vector<double, Allocator>& container) {
    size_t size = container.size();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_set_pd(
            random_double(0.0, 10000.0),
            random_double(0.0, 10000.0),
            random_double(0.0, 10000.0),
            random_double(0.0, 10000.0)
        );
        _mm256_stream_pd(container.data() + i, vec); // Non-temporal store
    }
    for (; i < size; ++i) {
        container[i] = random_double(0.0, 10000.0); // Scalar for remainder
    }
    _mm_mfence(); // Ensure stores complete
}

int main(int argc, char* argv[]) {
    auto [size, offset, iter] = process_args(argc, argv);
    if (offset >= size * sizeof(double)) {
        std::cerr << "Offset (" << offset << ") must be less than size in bytes (" << size * sizeof(double) << "), adjusting to 4\n";
        offset = 4;
    }
    zen::timer timer;

    // Misaligned data
    std::vector<double, AlignedAllocator<double>> missaliged_data(size);
    initialize_vector(missaliged_data); // Use non-temporal stores

    // Misaligned access
    double misaligned_time_total = 0;
    for (int r = 0; r < iter; ++r) {
        timer.start();
        volatile double sum = sum_misaligned(size, missaliged_data, offset);
        timer.stop();
        misaligned_time_total += timer.elapsed<zen::timer::nsec>().count();
    }
    double misaligned_time_avg = misaligned_time_total / iter;
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Misaligned access SIMD", misaligned_time_avg, "ns")));

    // Aligned data
    std::vector<double, AlignedAllocator<double>> alligned_data(size);
    initialize_vector(alligned_data); // Use non-temporal stores

    // Aligned access
    double aligned_time_total = 0;
    for (int r = 0; r < iter; ++r) {
        timer.start();
        volatile double sum = sum_aligned(size, alligned_data);
        timer.stop();
        aligned_time_total += timer.elapsed<zen::timer::nsec>().count();
    }
    double aligned_time_avg = aligned_time_total / iter;
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Aligned data SIMD", aligned_time_avg, "ns")));

    // Verification
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(missaliged_data.data());
    uintptr_t misaligned_addr = reinterpret_cast<uintptr_t>(
        reinterpret_cast<char*>(missaliged_data.data()) + offset);
    zen::print("Addresses - Base: 0x", std::hex, base_addr, 
               ", Misaligned: 0x", misaligned_addr, std::dec, "\n");
    double speedup = aligned_time_avg / misaligned_time_avg;
    zen::print(zen::color::yellow("Speedup factor: " + std::to_string(speedup) + "\n"));

    return 0;
}