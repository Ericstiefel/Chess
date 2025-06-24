#include "annotate_check.h"
#include "utils.h"
#include "constants.h"
#include "bitboard.h"
#include "bit_ops.h"
#include "move.h"

#include <vector>


bool is_check(const State& state) const {
    uint64_t king_sq = lsb_index(state.boards[state.toMove]);

    return is_square_attacked(state, king_sq);
}

void annotate_m_w_check(const State& state, std::vector<Move>& moves) {
    for (Move& move: moves) {
        
        if (move.promotion_type != PieceType::NONE) { state.promote(move); }
        else if (move.is_en_passant) { state.en_passant(move); }
        else if (move.is_castle) { state.castle(move); }
        else { state.move_pieces(move); }

        move.is_check = is_check(state);

    }
}