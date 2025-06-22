#pragma once

#include <vector>
#include <cstdint>
#include "constants.h"

int get_bit(uint64_t bb, int square);
void set_bit(uint64_t* bb, int square);
void clear_bit(uint64_t* bb, int square);
int lsb_index(uint64_t bitboard);
int msb_index(uint64_t bitboard);
int popcount(uint64_t bb);
int bitScanForward(uint64_t bb);
std::vector<int> bitscan(uint64_t bb);
std::vector<Square> bitboard_to_squares(uint64_t bb);
