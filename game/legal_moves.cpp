#include "pinned.h"
#include "possible_piece_moves.h"
#include "utils.h"
#include "bitboard.h"
#include "move.h"
#include "bit_ops.h"
#include "legal_king_moves.h"

#include <vector>

void annotate_m_w_check(State& state, std::vector<Move>& moves) { // Time/Memory tradeoff between just making state a copy and not unmaking move
    for (Move& move: moves) {
        
        if (move.promotion_type != PieceType::NONE) { state.promote(move); }
        else if (move.is_en_passant) { state.en_passant(move); }
        else if (move.is_castle) { state.castle(move); }
        else { state.move_pieces(move); }

        move.is_check = is_in_check(state);

        state.unmake_move();

    }
}

std::vector<Move> legal_moves(State& state) {
    std::vector<Move> total_moves;

    uint64_t king_loc = lsb_index(state.boards[state.toMove][5]);
    uint64_t attackers_bb = attackers_to_square(state, king_loc);
    uint8_t attackers_count = popcount(attackers_bb);
    bool in_check = attackers_count > 0;

    // Add king moves
    std::vector<Move> king_moves = kingMoves(state, attackers_count);
    annotate_m_w_check(state, king_moves);
    total_moves.insert(total_moves.end(), king_moves.begin(), king_moves.end());

    if (attackers_count == 2) {
        return total_moves; // Only king moves are legal
    }

    uint64_t valid_targets = 0xFFFFFFFFFFFFFFFF;

    if (attackers_count == 1) {
        uint64_t checker_sq = lsb_index(attackers_bb);
        valid_targets = (1ULL << checker_sq);

        auto checker_piece = state.piece_on_square(static_cast<Square>(checker_sq));
        if (checker_piece.has_value()) {
            PieceType pt = checker_piece.value();
            if (pt == PieceType::QUEEN || pt == PieceType::ROOK || pt == PieceType::BISHOP) {
                valid_targets |= squares_between(king_loc, checker_sq);
            }
        }
    }

    uint64_t own_occ = state.get_occupied_by_color(static_cast<Color>(state.toMove));
    uint64_t opp_occ = state.get_occupied_by_color(static_cast<Color>(state.toMove ^ 1));

    std::vector<Move> candidates;

    // Generate all non-king moves
    {
        auto pawn_moves = pawnMoves(state, opp_occ);
        annotate_m_w_check(state, pawn_moves);
        candidates.insert(candidates.end(), pawn_moves.begin(), pawn_moves.end());
    }
    {
        auto knight_moves = knightMoves(state, own_occ);
        annotate_m_w_check(state, knight_moves);
        candidates.insert(candidates.end(), knight_moves.begin(), knight_moves.end());
    }
    {
        auto bishop_moves = bishopMoves(state, own_occ, opp_occ);
        annotate_m_w_check(state, bishop_moves);
        candidates.insert(candidates.end(), bishop_moves.begin(), bishop_moves.end());
    }
    {
        auto rook_moves = rookMoves(state, own_occ, opp_occ);
        annotate_m_w_check(state, rook_moves);
        candidates.insert(candidates.end(), rook_moves.begin(), rook_moves.end());
    }
    {
        auto queen_moves = queenMoves(state, own_occ, opp_occ);
        annotate_m_w_check(state, queen_moves);
        candidates.insert(candidates.end(), queen_moves.begin(), queen_moves.end());
    }

    for (const Move& move : candidates) {
        bool is_pinned = pinned(state, static_cast<uint64_t>(move.from_sq), king_loc);
        uint64_t to_sq_bit = 1ULL << static_cast<uint64_t>(move.to_sq);

        if (in_check) {
            if ((to_sq_bit & valid_targets) && !is_pinned) {
                total_moves.push_back(move);
            }
            continue;
        }

        if (is_pinned) {
            if (is_along_ray(king_loc, static_cast<uint64_t>(move.from_sq), static_cast<uint64_t>(move.to_sq))) {
                total_moves.push_back(move);
            }
            continue;
        }

        total_moves.push_back(move);
    }

    return total_moves;
}
