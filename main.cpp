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

