#include "constants.h"

const std::array<std::array<char, 6>, 2> PIECE_SYMBOLS = {{
    {'P', 'N', 'B', 'R', 'Q', 'K'},  // White
    {'p', 'n', 'b', 'r', 'q', 'k'}   // Black
}};


const std::array<std::string, 64> SQUARE_NAMES = []() {
    std::array<std::string, 64> names{};
    const char* files = "abcdefgh";
    for (int i = 0; i < 64; ++i) {
        names[i] = std::string{files[i % 8]} + std::to_string((i / 8) + 1);
    }
    return names;
}();

const uint64_t RANK_1 = 0x00000000000000FFULL;
const uint64_t RANK_2 = 0x000000000000FF00ULL;
const uint64_t RANK_7 = 0x00FF000000000000ULL;
const uint64_t RANK_8 = 0xFF00000000000000ULL;
const uint64_t FILE_A = 0x0101010101010101ULL;
const uint64_t FILE_H = 0x8080808080808080ULL;
const uint64_t FILE_B = 0x0202020202020202ULL;
const uint64_t FILE_G = 0x4040404040404040ULL;

const std::array<int, 8> KNIGHT_OFFSETS = {-17, -15, -10, -6, 6, 10, 15, 17};


uint64_t knight_attack_mask(uint64_t sq) {
    uint64_t bb = 1ULL << sq;
    uint64_t attacks = 0ULL;

    if (sq > 63) return 0; // Safety check

    if ((bb >> 17) & ~0x8080808080808080ULL) attacks |= (bb >> 17);
    if ((bb >> 15) & ~0x0101010101010101ULL) attacks |= (bb >> 15);
    if ((bb >> 10) & ~0xC0C0C0C0C0C0C0C0ULL) attacks |= (bb >> 10);
    if ((bb >> 6)  & ~0x0303030303030303ULL) attacks |= (bb >> 6);
    if ((bb << 17) & ~0x0101010101010101ULL) attacks |= (bb << 17);
    if ((bb << 15) & ~0x8080808080808080ULL) attacks |= (bb << 15);
    if ((bb << 10) & ~0x0303030303030303ULL) attacks |= (bb << 10);
    if ((bb << 6)  & ~0xC0C0C0C0C0C0C0C0ULL) attacks |= (bb << 6);

    return attacks;
}

uint64_t king_attack_mask(uint64_t sq) {
    uint64_t bb = 1ULL << sq;
    uint64_t attacks = 0ULL;

    if ((bb >> 8)) attacks |= (bb >> 8);
    if ((bb << 8)) attacks |= (bb << 8);
    if ((bb >> 1) & ~0x8080808080808080ULL) attacks |= (bb >> 1);
    if ((bb << 1) & ~0x0101010101010101ULL) attacks |= (bb << 1);
    if ((bb >> 9) & ~0x8080808080808080ULL) attacks |= (bb >> 9);
    if ((bb >> 7) & ~0x0101010101010101ULL) attacks |= (bb >> 7);
    if ((bb << 9) & ~0x0101010101010101ULL) attacks |= (bb << 9);
    if ((bb << 7) & ~0x8080808080808080ULL) attacks |= (bb << 7);

    return attacks;
}

uint64_t bishop_attack_mask(uint64_t sq, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int rank = sq / 8;
    int file = sq % 8;

    // Directions: NE, NW, SE, SW
    // NE
    for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; ++r, ++f) {
        int idx = r * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // NW
    for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; ++r, --f) {
        int idx = r * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // SE
    for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; --r, ++f) {
        int idx = r * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // SW
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; --r, --f) {
        int idx = r * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    return attacks;
}


uint64_t rook_attack_mask(uint64_t sq, uint64_t occupancy) {
    uint64_t attacks = 0ULL;
    int rank = sq / 8;
    int file = sq % 8;

    // Directions: N, S, E, W
    // North
    for (int r = rank + 1; r <= 7; ++r) {
        int idx = r * 8 + file;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // South
    for (int r = rank - 1; r >= 0; --r) {
        int idx = r * 8 + file;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // East
    for (int f = file + 1; f <= 7; ++f) {
        int idx = rank * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    // West
    for (int f = file - 1; f >= 0; --f) {
        int idx = rank * 8 + f;
        attacks |= (1ULL << idx);
        if (occupancy & (1ULL << idx)) break;
    }

    return attacks;
}
