#pragma once
#include "bitboard.h"
#include "constants.h"
#include "move.h"
#include "bit_ops.h"
#include <vector>
#include <optional>

std::vector<Move> pawnMoves(const State& state, const uint64_t& opp_occupied_bb);
std::vector<Move> knightMoves(const State& state, const uint64_t& own_occupied_bb);
std::vector<Move> bishopMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb);
std::vector<Move> rookMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb);
std::vector<Move> queenMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb);
