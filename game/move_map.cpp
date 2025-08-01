#include "move_map.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <iostream>

namespace py = pybind11;

/**
 * @brief Loads the move data from the specified file into the two maps.
 * This logic is identical to the Python version's file parsing.
 * @param filename The path to the move map file.
 * @return True if loading was successful, false otherwise.
 */
bool MoveMapping::load_from_file(const std::string& filename) {
    index_to_move.clear();
    move_to_index.clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open move map file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int idx, piece_type_int, from_sq_int, to_sq_int, is_capture_int, promotion_type_int, is_castle_int;

        if (!(ss >> idx >> piece_type_int >> from_sq_int >> to_sq_int >> is_capture_int >> promotion_type_int >> is_castle_int)) {
            std::cerr << "Warning: Skipping malformed line in move map: " << line << std::endl;
            continue;
        }

        Move move;
        move.piece_type = static_cast<PieceType>(piece_type_int);
        move.from_sq = static_cast<Square>(from_sq_int);
        move.to_sq = static_cast<Square>(to_sq_int);
        move.is_capture = static_cast<bool>(is_capture_int);
        move.promotion_type = static_cast<PieceType>(promotion_type_int);
        move.is_castle = static_cast<bool>(is_castle_int);
        move.is_check = false;       
        move.is_en_passant = false;  
        index_to_move[idx] = move;


        auto key = std::make_tuple(
            static_cast<int>(move.from_sq),
            static_cast<int>(move.to_sq),
            move.is_castle,
            static_cast<int>(move.promotion_type)
        );
        move_to_index[key] = idx;
    }

    std::cout << "Successfully loaded " << index_to_move.size() << " moves from " << filename << std::endl;
    return true;
}


int MoveMapping::get_index_for_move(const Move& move) const {
    auto key = std::make_tuple(
        static_cast<int>(move.from_sq),
        static_cast<int>(move.to_sq),
        move.is_castle,
        static_cast<int>(move.promotion_type)
    );

    auto it = move_to_index.find(key);
    if (it != move_to_index.end()) {
        return it->second;
    }
    
    throw std::runtime_error("get_index_for_move: Move not found in C++ map.");
}


Move MoveMapping::get_move_for_index(int index) const {
    auto it = index_to_move.find(index);
    if (it != index_to_move.end()) {
        return it->second;
    }
    throw std::runtime_error("get_move_for_index: Index not found in C++ map.");
}


// --- Pybind11 Bindings ---
static MoveMapping g_move_mapping;

void load_move_map_from_python(const std::string& filename) {
    if (!g_move_mapping.load_from_file(filename)) {
        throw std::runtime_error("C++ failed to load move map from file: " + filename);
    }
}

int get_index_for_move_from_python(const Move& move) {
    return g_move_mapping.get_index_for_move(move);
}

Move get_move_for_index_from_python(int index) {
    return g_move_mapping.get_move_for_index(index);
}

