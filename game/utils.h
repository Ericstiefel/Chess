#pragma once
#include "bit_ops.h"
#include "possible_piece_moves.h"
#include "constants.h"
#include "bitboard.h"

#include <vector>
#include <cstdint>
#include <cmath>


bool is_square_attacked(const State& state, const uint64_t target_sq_idx);
uint64_t attackers_to_square(State& state, uint64_t target_sq);
bool is_along_ray(uint64_t origin, uint64_t from_sq, uint64_t to_sq);

std::vector<uint64_t> knight_attack_indices(uint64_t square);
uint64_t bishop_attacks(uint64_t square, uint64_t occupancy);
uint64_t rook_attacks(uint64_t square, uint64_t occupancy);
std::vector<uint64_t> king_attack_indices(uint64_t square);
uint64_t sliding_attacks(uint64_t square, uint64_t occupancy, const std::vector<std::pair<int, int>>& directions);
uint64_t squares_between(uint64_t from_sq, uint64_t to_sq);
