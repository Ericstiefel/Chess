#include "move.h"
#include "constants.h"

std::string Move::notation(Color move_color) const {
    std::string from = SQUARE_NAMES[static_cast<int>(from_sq)];
    std::string to = SQUARE_NAMES[static_cast<int>(to_sq)];
    std::string piece_char;

    if (piece_type != PieceType::PAWN) {
        switch (piece_type) {
            case PieceType::KNIGHT: piece_char = "N"; break;
            case PieceType::BISHOP: piece_char = "B"; break;
            case PieceType::ROOK:   piece_char = "R"; break;
            case PieceType::QUEEN:  piece_char = "Q"; break;
            case PieceType::KING:   piece_char = "K"; break;
            default: break;
        }
    }

    if (is_castle) {
        if (to_sq == Square::G1 || to_sq == Square::G8) return std::string("O-O") + (is_check ? "+" : "");
        if (to_sq == Square::C1 || to_sq == Square::C8) return std::string("O-O-O") + (is_check ? "+" : "");
    }

    std::string result = piece_char;
    if (is_capture) {
        if (piece_type == PieceType::PAWN) result += from[0];
        result += "x";
    }

    result += to;

    if (promotion_type != PieceType::NONE) { 
        result += "=";
        result += PIECE_SYMBOLS[static_cast<int>(move_color)][static_cast<int>(promotion_type)];
    }

    if (is_check) result += "+";
    return result;
}


std::string Move::str(Color move_color) const {
    return notation(move_color);
}

bool Move::operator==(const Move& other) const {
    return from_sq == other.from_sq &&
           to_sq == other.to_sq &&
           is_castle == other.is_castle &&
           promotion_type == other.promotion_type;
}

std::size_t Move::hash() const {

    std::size_t h1 = std::hash<int>()(static_cast<int>(from_sq));
    std::size_t h2 = std::hash<int>()(static_cast<int>(to_sq));
    std::size_t h3 = std::hash<int>()(static_cast<int>(promotion_type));
    return h1 ^ (h2 << 1) ^ (h3 << 2); 
}