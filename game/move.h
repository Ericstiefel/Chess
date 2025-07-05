#pragma once

#include <string>
#include "constants.h"

struct Move {
    PieceType piece_type;
    Square from_sq;
    Square to_sq;
    bool is_capture = false;
    PieceType promotion_type = PieceType::NONE;
    bool is_castle = false;
    bool is_check = false;
    bool is_en_passant = false;

    // Default constructor
    Move()
    : piece_type(PieceType::NONE), from_sq(Square::NO_SQUARE), to_sq(Square::NO_SQUARE),
      is_capture(false), promotion_type(PieceType::NONE),
      is_castle(false), is_check(false), is_en_passant(false) {}

    // Full constructor with defaults
    Move(PieceType p, Square f, Square t, bool capture,
        PieceType promo = PieceType::NONE,
        bool castle = false, bool check = false, bool enpassant = false)
    : piece_type(p), from_sq(f), to_sq(t), is_capture(capture),
      promotion_type(promo), is_castle(castle), is_check(check),
      is_en_passant(enpassant) {}

    Move(PieceType p, Square f, Square t)
    : piece_type(p), from_sq(f), to_sq(t) {}

    std::string notation(Color move_color) const;
     std::string str(Color move_color) const;

    bool operator==(const Move& other) const;
    std::size_t hash() const;
};
