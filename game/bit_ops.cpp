#include "bit_ops.h"
#include <vector>
#include <cstdint>

uint64_t get_bit(uint64_t bb, uint64_t square) {
    return (bb >> square) & 1;
}

void set_bit(uint64_t& bb, uint64_t square) {
    bb |= (1ULL << square);
}

void clear_bit(uint64_t& bb, uint64_t square) {
    bb &= ~(1ULL << square);
}

// Portable lsb_index (count trailing zeros)
int lsb_index(uint64_t bb) {
    if (bb == 0) return -1;
    int idx = 0;
    while ((bb & 1) == 0) {
        bb >>= 1;
        idx++;
    }
    return idx;
}

// Portable msb_index (position of highest set bit)
int msb_index(uint64_t bb) {
    if (bb == 0) return -1;
    int idx = 0;
    while (bb >>= 1) {
        ++idx;
    }
    return idx;
}

// Portable popcount (Hamming weight)
uint64_t popcount(uint64_t bb) {
    uint64_t count = 0;
    while (bb) {
        bb &= (bb - 1);
        ++count;
    }
    return count;
}

std::vector<uint64_t> bitscan(uint64_t bb) {
    std::vector<uint64_t> indices;
    while (bb) {
        int idx = lsb_index(bb);
        indices.push_back(idx);
        bb &= bb - 1;
    }
    return indices;
}

std::vector<Square> bitboard_to_squares(uint64_t bb) {
    std::vector<Square> squares;
    while (bb) {
        int sq_idx = lsb_index(bb);
        squares.push_back(static_cast<Square>(sq_idx));
        bb &= bb - 1;
    }
    return squares;
}
