cmake_minimum_required(VERSION 3.15)
project(ChessModels LANGUAGES CXX)

# Set standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force MSVC to use C++20 in all build types
if (MSVC)
    foreach(FLAG_VAR
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL)
        set(${FLAG_VAR} "${${FLAG_VAR}} /std:c++20")
    endforeach()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")



# ======== Torch Setup ========
set(Torch_DIR "C:/Users/Erics/Documents/Chess/libtorch/share/cmake/Torch")


add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
set_target_properties(CUDA::nvToolsExt PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
    INTERFACE_LINK_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/nvToolsExt64_1.lib"
)


find_package(Torch REQUIRED)

# ======== pybind11 include path via pip ========
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${PYBIND11_INCLUDE_DIRS})

include_directories("C:/Users/Erics/AppData/Local/Programs/Python/Python313/include")


# ======== Shared Source Paths ========
set(GAME_SRCS
    ../../game/bitboard.cpp
    ../../game/move.cpp
    ../../game/legal_moves.cpp
    ../../game/turn.cpp
    ../../game/constants.cpp
    ../../game/legal_king_moves.cpp
    ../../game/pinned.cpp
    ../../game/possible_piece_moves.cpp
    ../../game/utils.cpp
    ../../game/bit_ops.cpp
)

# target_include_directories(model_trainer PRIVATE "${CMAKE_SOURCE_DIR}/../..")
# target_link_libraries(model_trainer PRIVATE "${TORCH_LIBRARIES}")
# target_link_libraries(model_trainer PRIVATE "C:/Users/Erics/AppData/Local/Programs/Python/Python313/libs/python313.lib")

# ======== index_to_move Python Module ========
add_library(index_to_move SHARED
    index_to_move.cpp
    ${GAME_SRCS}
)

# Set suffix to .pyd for Python module
set_target_properties(index_to_move PROPERTIES PREFIX "" SUFFIX ".pyd")

# Include parent project and game headers
target_include_directories(index_to_move PRIVATE "${CMAKE_SOURCE_DIR}/../..")

# Link LibTorch and Python
target_link_libraries(index_to_move PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(index_to_move PRIVATE "C:/Users/Erics/AppData/Local/Programs/Python/Python313/libs/python313.lib")


# ======== mcts Python Module ========
add_library(mcts_module SHARED
    ../AlphaGoZero/mcts.cpp
    ../AlphaGoZero/alphanet.cpp  # Needed because MCTS uses Net
    index_to_move.cpp
    ${GAME_SRCS}
)
set_target_properties(mcts_module PROPERTIES PREFIX "" SUFFIX ".pyd")
target_include_directories(mcts_module PRIVATE "${CMAKE_SOURCE_DIR}/../..")
target_link_libraries(mcts_module PRIVATE "${TORCH_LIBRARIES}")



# ======== alphanet Python Module ========
add_library(alphanet_module SHARED
    ../AlphaGoZero/alphanet.cpp
)
set_target_properties(alphanet_module PROPERTIES PREFIX "" SUFFIX ".pyd")
target_include_directories(alphanet_module PRIVATE "${CMAKE_SOURCE_DIR}/../..")
target_link_libraries(alphanet_module PRIVATE "${TORCH_LIBRARIES}")



# Tell CMake where to find python313.lib
link_directories("C:/Users/Erics/AppData/Local/Programs/Python/Python313/libs")
target_link_libraries(mcts_module PRIVATE "C:/Users/Erics/AppData/Local/Programs/Python/Python313/libs/python313.lib")
target_link_libraries(alphanet_module PRIVATE "C:/Users/Erics/AppData/Local/Programs/Python/Python313/libs/python313.lib")

# ======== DLL Copy on Windows ========
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    # add_custom_command( #TARGET model_trainer POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #     ${TORCH_DLLS}
    #     $<TARGET_FILE_DIR:model_trainer>
    #     COMMENT "Copying LibTorch DLLs to output directory..."
    # )
endif()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
