#include <iostream>

#include "constants.h"
#include "move.h"

std::vector<Move> CAPTURE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::D7, Square::D5},
    {PieceType::PAWN, Square::E4, Square::D5, true}

};


std::vector<Move> REGULAR_KINGSIDE_CASTLE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::BISHOP, Square::F1, Square::C4},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::KING, Square::E1, Square::G1, false, PieceType::NONE, true} // is_castle = true
};


std::vector<Move> REGULAR_QUEENSIDE_CASTLE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::D7, Square::D5},
    {PieceType::PAWN, Square::E4, Square::D5, true},
    {PieceType::QUEEN, Square::D8, Square::D5, true},
    {PieceType::KNIGHT, Square::B1, Square::C3},
    {PieceType::QUEEN, Square::D5, Square::A5},
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::BISHOP, Square::C8, Square::G4},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::BISHOP, Square::F1, Square::E2},
    {PieceType::KING, Square::E8, Square::C8, false, PieceType::NONE, true, false} // is_castle = true
};

std::vector<Move> EN_PASSANT_SIMPLE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::A7, Square::A6},
    {PieceType::PAWN, Square::E4, Square::E5},
    {PieceType::PAWN, Square::D7, Square::D5},
    {PieceType::PAWN, Square::E5, Square::D6, true, PieceType::NONE, false, false, true}
};


std::vector<Move> REGULAR_PROMOTION_TO_QUEEN = {
    {PieceType::PAWN, Square::E2, Square::E4}, 
    {PieceType::PAWN, Square::A7, Square::A6}, 
    {PieceType::PAWN, Square::E4, Square::E5},  
    {PieceType::PAWN, Square::F7, Square::F6}, 
    {PieceType::PAWN, Square::E5, Square::F6, true},
    {PieceType::PAWN, Square::H7, Square::H5}, 
    {PieceType::PAWN, Square::F6, Square::G7, true} ,
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::PAWN, Square::G7, Square::H8, true, PieceType::QUEEN}
};

std::vector<Move> EN_PASSANT_REVEALS_CHECK = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::PAWN, Square::D7, Square::D5},
    {PieceType::BISHOP, Square::C1, Square::G5},
    {PieceType::BISHOP, Square::F8, Square::E7},
    {PieceType::PAWN, Square::D4, Square::D5},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::QUEEN, Square::D1, Square::D4},
    {PieceType::PAWN, Square::C7, Square::C5},
    {PieceType::PAWN, Square::E4, Square::E5},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::PAWN, Square::D5, Square::C6, true},
    {PieceType::PAWN, Square::B7, Square::C6, true},
    {PieceType::BISHOP, Square::G5, Square::F4},
    {PieceType::QUEEN, Square::D8, Square::C7},
    {PieceType::PAWN, Square::F2, Square::F4},
    {PieceType::PAWN, Square::E5, Square::F4, true, PieceType::NONE, false, true, true}
};


std::vector<Move> CASTLE_KING_IN_CHECK = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::PAWN, Square::F7, Square::F6},
    {PieceType::KNIGHT, Square::F3, Square::E5, true},
    {PieceType::PAWN, Square::F6, Square::E5, true},
    {PieceType::QUEEN, Square::D1, Square::H5, false, PieceType::NONE, false, true},
    {PieceType::PAWN, Square::G7, Square::G6},
    {PieceType::QUEEN, Square::H5, Square::E5, true, PieceType::NONE, false, true}
};


std::vector<Move> CASTLE_THROUGH_CHECK = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::PAWN, Square::E5, Square::D4, true},
    {PieceType::KNIGHT, Square::F3, Square::D4, true},
    {PieceType::BISHOP, Square::F8, Square::C5}
};


std::vector<Move> CASTLE_DELIVERS_CHECKMATE = {
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::PAWN, Square::C2, Square::C4},
    {PieceType::PAWN, Square::E7, Square::E6},
    {PieceType::KNIGHT, Square::B1, Square::C3},
    {PieceType::BISHOP, Square::F8, Square::B4},
    {PieceType::QUEEN, Square::D1, Square::C2},
    {PieceType::PAWN, Square::D7, Square::D5},
    {PieceType::PAWN, Square::C4, Square::D5, true},
    {PieceType::PAWN, Square::E6, Square::D5, true},
    {PieceType::PAWN, Square::A2, Square::A3},
    {PieceType::BISHOP, Square::B4, Square::C3, true, PieceType::NONE, false, true},
    {PieceType::PAWN, Square::B2, Square::C3, true},
    {PieceType::PAWN, Square::B7, Square::B6},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::BISHOP, Square::C8, Square::B7},
    {PieceType::BISHOP, Square::C1, Square::D2},
    {PieceType::QUEEN, Square::D8, Square::C7},
    {PieceType::PAWN, Square::E2, Square::E3},
    {PieceType::KING, Square::E8, Square::G8, false, PieceType::NONE, true}, // Black castles kingside
    {PieceType::BISHOP, Square::F1, Square::D3},
    {PieceType::ROOK, Square::A8, Square::C8},
    {PieceType::KING, Square::E1, Square::C1, false, PieceType::NONE, true, true} // White castles queenside, delivering checkmate.
};



std::vector<Move> PROMOTION_WITH_CAPTURE = {
    {PieceType::PAWN, Square::C2, Square::C4},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::PAWN, Square::E7, Square::E6},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::BISHOP, Square::F8, Square::E7},
    {PieceType::BISHOP, Square::C1, Square::G5},
    {PieceType::PAWN, Square::H7, Square::H6},
    {PieceType::BISHOP, Square::G5, Square::H4},
    {PieceType::KING, Square::E8, Square::G8, false, PieceType::NONE, true},
    {PieceType::PAWN, Square::E2, Square::E3},
    {PieceType::PAWN, Square::B7, Square::B6},
    {PieceType::KNIGHT, Square::B1, Square::D2},
    {PieceType::PAWN, Square::C7, Square::C5},
    {PieceType::PAWN, Square::C4, Square::D5, true},
    {PieceType::PAWN, Square::E6, Square::D5, true},
    {PieceType::BISHOP, Square::F1, Square::D3},
    {PieceType::BISHOP, Square::C8, Square::B7},
    {PieceType::PAWN, Square::A2, Square::A4},
    {PieceType::PAWN, Square::A7, Square::A5},
    {PieceType::QUEEN, Square::D1, Square::B3},
    {PieceType::BISHOP, Square::B7, Square::C8},
    {PieceType::PAWN, Square::D4, Square::D5, true},
    {PieceType::PAWN, Square::C5, Square::D5, true},
    {PieceType::PAWN, Square::B2, Square::B4},
    {PieceType::QUEEN, Square::D8, Square::C7},
    {PieceType::PAWN, Square::B4, Square::A5, true},
    {PieceType::PAWN, Square::B6, Square::A5, true},
    {PieceType::PAWN, Square::G2, Square::G4},
    {PieceType::PAWN, Square::H6, Square::G5, true},
    {PieceType::PAWN, Square::H4, Square::G5, true},
    {PieceType::PAWN, Square::F7, Square::F5},
    {PieceType::PAWN, Square::G5, Square::F6, true, PieceType::NONE, false, false, true},
    {PieceType::BISHOP, Square::E7, Square::F6, true},
    {PieceType::QUEEN, Square::B3, Square::B7},
    {PieceType::BISHOP, Square::F6, Square::G7},
    {PieceType::PAWN, Square::E3, Square::E4},
    {PieceType::PAWN, Square::F5, Square::E4, true},
    {PieceType::KNIGHT, Square::F3, Square::E5},
    {PieceType::QUEEN, Square::C7, Square::E5, true},
    {PieceType::PAWN, Square::F2, Square::F4},
    {PieceType::QUEEN, Square::E5, Square::G3},
    {PieceType::KING, Square::E1, Square::G1, false, PieceType::NONE, true},
    {PieceType::BISHOP, Square::C8, Square::F5},
    {PieceType::PAWN, Square::D5, Square::D6, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::G8, Square::H7},
    {PieceType::PAWN, Square::D6, Square::D7},
    {PieceType::ROOK, Square::A8, Square::E8},
    {PieceType::PAWN, Square::D7, Square::E8, true, PieceType::QUEEN, false, true},
};


std::vector<Move> UNDERPROMOTION_TO_KNIGHT_CHECK = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::PAWN, Square::F7, Square::F5},
    {PieceType::PAWN, Square::E4, Square::F5, true},
    {PieceType::PAWN, Square::E5, Square::F5, true},
    {PieceType::KNIGHT, Square::F3, Square::E5},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::PAWN, Square::D2, Square::D4},
    {PieceType::KNIGHT, Square::F6, Square::D5},
    {PieceType::PAWN, Square::C2, Square::C3},
    {PieceType::PAWN, Square::C7, Square::C6},
    {PieceType::BISHOP, Square::C1, Square::F4},
    {PieceType::BISHOP, Square::F8, Square::C5},
    {PieceType::QUEEN, Square::D1, Square::B3},
    {PieceType::QUEEN, Square::D8, Square::C7},
    {PieceType::KING, Square::E1, Square::D1},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::BISHOP, Square::F1, Square::C4},
    {PieceType::PAWN, Square::A7, Square::A6},
    {PieceType::KNIGHT, Square::B1, Square::D2},
    {PieceType::KING, Square::E8, Square::G8, false, PieceType::NONE, true},
    {PieceType::PAWN, Square::G2, Square::G3},
    {PieceType::PAWN, Square::H7, Square::H6},
    {PieceType::PAWN, Square::F5, Square::F6},
    {PieceType::PAWN, Square::G7, Square::F6, true},
    {PieceType::PAWN, Square::E5, Square::F6, true, PieceType::NONE, false, true},
    {PieceType::BISHOP, Square::C5, Square::F2, true, PieceType::NONE, false, true},
    {PieceType::ROOK, Square::H1, Square::F2, true},
    {PieceType::QUEEN, Square::C7, Square::F4, true},
    {PieceType::QUEEN, Square::B3, Square::F7, true},
    {PieceType::KING, Square::G8, Square::H8},
    {PieceType::PAWN, Square::F6, Square::G7, true, PieceType::KNIGHT, false, true}
};


std::vector<Move> SCHOLARS_MATE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::QUEEN, Square::D1, Square::H5},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::BISHOP, Square::F1, Square::C4},
    {PieceType::KNIGHT, Square::G8, Square::F6},
    {PieceType::QUEEN, Square::H5, Square::F7, true, PieceType::NONE, false, true} // Checkmate
};

std::vector<Move> FOOLS_MATE = {
    {PieceType::PAWN, Square::F2, Square::F3},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::PAWN, Square::G2, Square::G4},
    {PieceType::QUEEN, Square::D8, Square::H4, false, PieceType::NONE, false, true} // Checkmate
};


std::vector<Move> SMOTHERED_MATE = {
    {PieceType::PAWN, Square::E2, Square::E4},
    {PieceType::PAWN, Square::E7, Square::E5},
    {PieceType::KNIGHT, Square::G1, Square::F3},
    {PieceType::KNIGHT, Square::B8, Square::C6},
    {PieceType::BISHOP, Square::F1, Square::C4},
    {PieceType::PAWN, Square::D7, Square::D6},
    {PieceType::KNIGHT, Square::F3, Square::G5},
    {PieceType::BISHOP, Square::C8, Square::G4},
    {PieceType::PAWN, Square::H2, Square::H3},
    {PieceType::BISHOP, Square::G4, Square::H5},
    {PieceType::KNIGHT, Square::G5, Square::F7, true},
    {PieceType::ROOK, Square::H8, Square::F7, true},
    {PieceType::QUEEN, Square::D1, Square::H5, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::E8, Square::F8},
    {PieceType::BISHOP, Square::C4, Square::F7, true, PieceType::NONE, false, true},
    {PieceType::KING, Square::F8, Square::G8},
    {PieceType::QUEEN, Square::H5, Square::H7, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::G8, Square::F8},
    {PieceType::KNIGHT, Square::B1, Square::C3},
    {PieceType::PAWN, Square::H7, Square::H6},
    {PieceType::QUEEN, Square::H7, Square::H8, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::F8, Square::E7},
    {PieceType::QUEEN, Square::H8, Square::F6, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::E7, Square::D8},
    {PieceType::QUEEN, Square::F6, Square::F7, false, PieceType::NONE, false, true},
    {PieceType::KING, Square::D8, Square::C8},
    {PieceType::KNIGHT, Square::C3, Square::D5},
    {PieceType::PAWN, Square::A7, Square::A6},
    {PieceType::KNIGHT, Square::D5, Square::E7, false, PieceType::NONE, false, true}, // Smothered mate
};