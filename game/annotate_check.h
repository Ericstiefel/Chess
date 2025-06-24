#include "utils.h"
#include "constants.h"
#include "bitboard.h"
#include "bit_ops.h"
#include "move.h"

bool is_check(const State& state) const;
void annotate_m_w_check(const State& state, std::vector<Move>& moves);