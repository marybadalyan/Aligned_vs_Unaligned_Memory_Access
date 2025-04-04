#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h> // AVX2
#include <memory>
#include <chrono>
#include <iomanip>
#include "kaizen.h" // Assuming this provides zen::timer, zen::print, zen::color

double random_double(double min, double max) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Aligned allocator (8-byte alignment per task)
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

// Misaligned allocator (offset by 1 byte)
template<typename T>
struct MisalignedAllocator {
    using value_type = T;
    int offset;

    MisalignedAllocator(int off = 1) noexcept : offset(off) {}
    template<typename U> MisalignedAllocator(const MisalignedAllocator<U>& other) noexcept : offset(other.offset) {}

    T* allocate(std::size_t n) {
        void* raw = _mm_malloc(n * sizeof(T) + 8, 8); // Overallocate
        if (!raw) throw std::bad_alloc();
        void* misaligned_ptr = reinterpret_cast<char*>(raw) + offset; // Offset by 'offset' bytes
        return reinterpret_cast<T*>(misaligned_ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept {
        void* raw = reinterpret_cast<char*>(ptr) - offset; // Reverse offset
        _mm_free(raw);
    }

    template<typename U>
    struct rebind {
        using other = MisalignedAllocator<U>;
    };
};

// Aligned sum with AVX2
double sum_aligned(size_t size, std::vector<double, AlignedAllocator<double>>& container) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_load_pd(container.data() + i); // Aligned load
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) {
        sum += container[i];
    }
    double result[4];
    _mm256_store_pd(result, sum_vec);
    for (int j = 0; j < 4; ++j) {
        sum += result[j];
    }
    return sum;
}

// Unaligned sum with AVX2
double sum_misaligned(size_t size, std::vector<double, MisalignedAllocator<double>>& container) {
    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(container.data() + i); // Unaligned load
        sum_vec = _mm256_add_pd(sum_vec, vec);
    }
    double sum = 0;
    for (; i < size; ++i) {
        sum += container[i];
    }
    double result[4];
    _mm256_storeu_pd(result, sum_vec);
    for (int j = 0; j < 4; ++j) {
        sum += result[j];
    }
    return sum;
}

// Warm-up function
template<typename Allocator>
void warm_up(std::vector<double, Allocator>& container) {
    volatile double sum = 0; // Prevent optimization
    for (size_t i = 0; i < container.size(); ++i) {
        sum += container[i];
    }
}

int main() {
    const size_t size = 1000000; // Fixed size per task
    const int runs = 10; // Multiple runs for consistency
    zen::timer timer;

    // Aligned data
    std::vector<double, AlignedAllocator<double>> aligned_data(size);
    for (auto& val : aligned_data) {
        val = random_double(0.0, 10000.0);
    }
    warm_up(aligned_data);

    double aligned_time_total = 0;
    double aligned_sum_result = 0;
    for (int r = 0; r < runs; ++r) {
        timer.start();
        aligned_sum_result = sum_aligned(size, aligned_data);
        timer.stop();
        aligned_time_total += timer.duration<zen::timer::nsec>().count();
    }
    double aligned_time_avg = aligned_time_total / runs;
    zen::print(zen::color::green(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Aligned data SIMD", aligned_time_avg, "ns")));

    // Misaligned data
    std::vector<double, MisalignedAllocator<double>> misaligned_data(size, MisalignedAllocator<double>(1));
    for (auto& val : misaligned_data) {
        val = random_double(0.0, 10000.0);
    }
    warm_up(misaligned_data);

    double misaligned_time_total = 0;
    double misaligned_sum_result = 0;
    for (int r = 0; r < runs; ++r) {
        timer.start();
        misaligned_sum_result = sum_misaligned(size, misaligned_data);
        timer.stop();
        misaligned_time_total += timer.duration<zen::timer::nsec>().count();
    }
    double misaligned_time_avg = misaligned_time_total / runs;
    zen::print(zen::color::red(std::format("| {:<36} | {:>12.2f} | {:<9} |\n", 
               "Misaligned data SIMD", misaligned_time_avg, "ns")));

    // Verify results
    std::cout << "Aligned sum: " << aligned_sum_result << ", Misaligned sum: " << misaligned_sum_result << "\n";
    std::cout << "Addresses - Aligned: " << aligned_data.data() << ", Misaligned: " << misaligned_data.data() << "\n";

    return 0;
}