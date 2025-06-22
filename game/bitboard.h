#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <cstdint>
#include <optional>

#include "constants.h"
#include "move.h"

struct State {
    uint8_t toMove = 0;
    std::vector<std::vector<uint64_t>> boards;
    std::vector<std::tuple<Move, PieceType, uint8_t, Square, uint8_t>> moves; // Move, captured_piece, castling, en_passant, fifty_moves 
    uint8_t castling; // trailing 4 bits represent castling capabilities
    Square en_passant_sq; // en passant Square (if possible)
    uint8_t fifty_move;

    State(); 

    std::string str() const;
    bool operator==(const State& other) const;
    std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t> hash() const;


    void reset();
    void printBoard() const;

    uint64_t get_all_occupied_squares() const;
    uint64_t get_occupied_by_color(const Color color);
    std::optional<PieceType> piece_on_square(const Square sq);
    std::optional<PieceType> piece_on_square_by_color(const Square sq, const Color color);
    std::optional<Color> color_on_square(const Square sq);

    void move_pieces(const Move move);
    void castle(const Move move);
    void promote(const Move move);
    void en_passant(const Move move);

    void unmake_move();

};
