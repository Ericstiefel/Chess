#pragma once
#include <array>
#include <string>
#include <cstdint>

// -------------------- ENUM DEFINITIONS --------------------
enum class Square : int {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQUARE = -1
};

enum class PieceType : int {
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NONE
};

enum class Color : int {
    WHITE = 0, BLACK = 1
};

extern const std::array<std::array<char, 6>, 2> PIECE_SYMBOLS;
extern const std::array<std::string, 64> SQUARE_NAMES;

extern const uint64_t RANK_1;
extern const uint64_t RANK_2;
extern const uint64_t RANK_7;
extern const uint64_t RANK_8;
extern const uint64_t FILE_A;
extern const uint64_t FILE_H;
extern const uint64_t FILE_B;
extern const uint64_t FILE_G;

extern const std::array<int, 8> KNIGHT_OFFSETS;

constexpr int BOARD_SIZE = 8;


uint64_t knight_attack_mask(uint64_t sq);
uint64_t king_attack_mask(uint64_t sq);
uint64_t bishop_attack_mask(uint64_t sq, uint64_t occupancy);
uint64_t rook_attack_mask(uint64_t sq, uint64_t occupancy);
