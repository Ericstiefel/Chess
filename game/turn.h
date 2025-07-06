#pragma once
#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "move.h"
#include "legal_moves.h"
#include "utils.h"

#include <tuple>
#include <vector>
#include <cstdint>
#include <functional>

void turn(State& state, Move& move);
float game_over(
    State& state,
    std::vector<Move>& moves
);


bool fifty_move_rule(const State& state);
uint8_t num_pieces(const State& state);
float check_or_stale_mate(const State& state, const std::vector<Move>& moves);
std::array<uint8_t, 12> count_pieces(const State& state);
bool is_insufficient_material(const State& state);
bool is_threefold_repetition(State& state); 



