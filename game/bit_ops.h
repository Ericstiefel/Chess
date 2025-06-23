#pragma once

#include <vector>
#include <cstdint>
#include "constants.h"

uint64_t get_bit(uint64_t bb, uint64_t square);
void set_bit(uint64_t& bb, uint64_t square);
void clear_bit(uint64_t& bb, uint64_t square);
uint64_t lsb_index(uint64_t bitboard);
uint64_t msb_index(uint64_t bitboard);
uint64_t popcount(uint64_t bb);
uint64_t bitScanForward(uint64_t bb);
std::vector<uint64_t> bitscan(uint64_t bb);
std::vector<Square> bitboard_to_squares(uint64_t bb);
