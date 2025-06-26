#pragma once

#include <cstdint>
#include <vector>
#include "constants.h"
#include "bitboard.h"

uint64_t pawn_attacks_from(uint64_t square, Color pawn_color);

uint64_t knight_attacks_from(uint64_t square);

uint64_t king_attacks_from(uint64_t square);

uint64_t get_bishop_attacks(uint64_t square, uint64_t occupancy);

uint64_t get_rook_attacks(uint64_t square, uint64_t occupancy);

bool is_square_attacked(const State& state, uint64_t target_idx);

bool is_in_check(const State& state);

uint64_t attackers_to_square(State& state, uint64_t target_sq);

uint64_t squares_between(uint64_t from_sq, uint64_t to_sq);

bool is_along_ray(uint64_t origin, uint64_t from_sq, uint64_t to_sq);