#pragma once

#include <array>
#include <tuple>
#include <string>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "constants.h"
#include "move.h"

// Combine hash values
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
    template <>
    struct hash<std::tuple<uint8_t, std::array<uint64_t, 12>, uint8_t, uint64_t>> {
        std::size_t operator()(const std::tuple<uint8_t, std::array<uint64_t, 12>, uint8_t, uint64_t>& tup) const {
            std::size_t seed = 0;

            const uint8_t& toMove = std::get<0>(tup);
            const std::array<uint64_t, 12>& boards = std::get<1>(tup);
            const uint8_t& castling = std::get<2>(tup);
            const uint64_t& en_passant = std::get<3>(tup);

            hash_combine(seed, std::hash<uint8_t>{}(toMove));
            for (std::size_t i = 0; i < 12; ++i) {
                hash_combine(seed, std::hash<uint64_t>{}(boards[i]));
            }
            hash_combine(seed, std::hash<uint8_t>{}(castling));
            hash_combine(seed, std::hash<uint64_t>{}(en_passant));

            return seed;
        }
    };
}

struct State {
    uint8_t toMove = 0;
    std::array<uint64_t, 12> boards;
    std::vector<std::tuple<Move, PieceType, uint8_t, Square, uint8_t>> moves; // Move, captured_piece, castling, en_passant, fifty_moves 
    uint8_t castling; // trailing 4 bits represent castling capabilities
    Square en_passant_sq; // en passant Square (if possible)
    uint8_t fifty_move;
    std::unordered_map<std::string, int> repetition_table;

    State(); 

    std::string str() const;
    bool operator==(const State& other) const;
    std::tuple<uint8_t, std::array<uint64_t, 12>, uint8_t, uint64_t> hash() const;
    std::string get_string_key() const;

    void reset();
    void printBoard() const;

    uint64_t get_all_occupied_squares() const;
    uint64_t get_occupied_by_color(const Color color) const;
    PieceType piece_on_square(const Square sq) const;
    PieceType piece_on_square_by_color(const Square sq, const Color color) const;
    Color color_on_square(const Square sq) const;

    void move_pieces(const Move move);
    void castle(const Move move);
    void promote(const Move move);
    void en_passant(const Move move);
    void unmake_move();
};
