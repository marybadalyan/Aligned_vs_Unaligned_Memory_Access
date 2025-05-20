cmake_minimum_required(VERSION 3.10)

project(Aligned_vs_Unaligned_Memory_Access)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force RelWithDebInfo to include full debug info
set(CMAKE_BUILD_TYPE "RelWithDebInfo")


if(MSVC)
    add_compile_options(/arch:AVX2)
elseif(UNIX)
    add_compile_options(-mavx2)
endif()




# MSVC-specific flags
if (MSVC)
    add_compile_options(/Zi /Zo /Od) # /Zi: debug info, /Zo: enhanced debug, /Od: no optimization (optional)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG") # ensure .pdb is generated
endif()

add_executable(Aligned_vs_Unaligned_Memory_Access main.cpp)