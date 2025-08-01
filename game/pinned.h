#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "utils.h"


bool quick_pin_check(const uint64_t king_sq, const uint64_t piece_sq);
bool pinned(const State& state, const uint64_t piece_sq, const uint64_t king_sq);