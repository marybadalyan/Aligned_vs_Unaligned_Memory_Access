cmake_minimum_required(VERSION 3.10)

project(Aligned_vs_Unaligned_Memory_Access)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(MSVC)
    add_compile_options(/arch:AVX2)
elseif(UNIX)
    add_compile_options(-mavx2)
endif()

add_executable(Aligned_vs_Unaligned_Memory_Access main.cpp)