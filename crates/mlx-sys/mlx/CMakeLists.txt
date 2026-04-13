cmake_minimum_required(VERSION 3.24)
project(mlx_fetch LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

option(MLX_BUILD_METAL "Build with Metal" ON)
option(MLX_BUILD_ACCELERATE "Build with Accelerate" ON)

include(FetchContent)
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG v0.31.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mlx)
