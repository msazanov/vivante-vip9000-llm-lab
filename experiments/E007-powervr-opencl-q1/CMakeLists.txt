cmake_minimum_required(VERSION 3.16)
project(powervr_opencl_q1_runner C)

add_executable(powervr-opencl-q1-runner powervr_opencl_q1_runner.c)
target_compile_features(powervr-opencl-q1-runner PRIVATE c_std_11)
target_compile_options(powervr-opencl-q1-runner PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)
target_link_libraries(powervr-opencl-q1-runner PRIVATE ${CMAKE_DL_LIBS} m)
