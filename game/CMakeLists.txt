cmake_minimum_required(VERSION 3.14)
project(game_py_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add pybind11 from submodule
add_subdirectory(${CMAKE_SOURCE_DIR}/../extern/pybind11 extern/pybind11_build)

# Source files for your chess engine
set(SOURCES
    bindings.cpp
    bit_ops.cpp
    constants.cpp
    utils.cpp
    move.cpp
    bitboard.cpp
    possible_piece_moves.cpp
    legal_moves.cpp
    legal_king_moves.cpp
    pinned.cpp
    turn.cpp
)

# Create Python module
pybind11_add_module(game_py ${SOURCES})

# Include headers in same directory
target_include_directories(game_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
