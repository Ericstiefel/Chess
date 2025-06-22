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

    std::string notation(Color move_color) const;
     std::string str(Color move_color) const;

    bool operator==(const Move& other) const;
    std::size_t hash() const;
};
