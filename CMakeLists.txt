cmake_minimum_required(VERSION 3.18)
project(AstroStacker VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /GL /arch:AVX2 /openmp /EHsc /fp:fast)
else()
    add_compile_options(-O3 -march=znver3 -mfma -mno-avx512f -ffast-math -fopenmp)
endif()

find_package(OpenCV REQUIRED)

add_executable(astro_stacker src/main.cpp)
target_link_libraries(astro_stacker PRIVATE ${OpenCV_LIBS})
