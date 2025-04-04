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
