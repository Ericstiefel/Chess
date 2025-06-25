#include "utils.h"
#include "bitboard.h"
#include "bit_ops.h"
#include "constants.h"

#include <iostream>

std::vector<Move> castleMoves(const State& state);
std::vector<Move> kingMoves(State& state, int attacker_ct);