#include "bit_ops.h"
#include <bit>       
#include <algorithm> 

int get_bit(uint64_t bb, int square) {
    return (bb >> square) & 1;
}

void set_bit(uint64_t* bb, int square) {
    *bb |= (1ULL << square); 
}


void clear_bit(uint64_t* bb, int square) {
    (*bb) = (*bb) & ~(1ULL << square);
}

int lsb_index(uint64_t bitboard) {
    if (bitboard == 0) return -1;
    return std::countr_zero(bitboard);
}

int msb_index(uint64_t bitboard) {
    if (bitboard == 0) return -1;
    return 63 - std::countl_zero(bitboard);
}

int popcount(uint64_t bb) {
    return std::popcount(bb);
}

int bitScanForward(uint64_t bb) {
    return lsb_index(bb);
}

std::vector<int> bitscan(uint64_t bb) {
    std::vector<int> indices;
    while (bb) {
        int idx = std::countr_zero(bb);
        indices.push_back(idx);
        bb &= bb - 1;
    }
    return indices;
}

std::vector<Square> bitboard_to_squares(uint64_t bb) {
    std::vector<Square> squares;
    while (bb) {
        int sq_idx = std::countr_zero(bb);
        squares.push_back(static_cast<Square>(sq_idx));
        bb &= bb - 1;
    }
    return squares;
}
