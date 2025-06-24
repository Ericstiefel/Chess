#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "move.h"
#include "legal_moves.h"
#include "utils.h"

#include <unordered_map>

void turn(State& state, Move& move);
float game_over(const State& state);


bool fifty_move_rule(const State& state);
uint8_t num_pieces(const State& state);
float check_or_stale_mate(const State& state, const std::vector<Move>& moves);
std::unordered_map<PieceType, uint8_t> count_pieces(const State& state, Color color);
bool is_insufficient_material(const State& state);