#include "turn.h"
#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "move.h"
#include "legal_moves.h"
#include "utils.h"

#include <vector>
#include <unordered_map>


void turn(State& state, Move& move) {
    
    if (move.piece_type == PieceType::PAWN || move.is_capture || move.is_castle) { state.fifty_move = 0; }

    else { state.fifty_move += 1; }

    if (move.promotion_type != PieceType::NONE) {
        state.promote(move);
        return;
    }

    else if (move.is_castle) {
        state.castle(move);
        return;
    }

    else if (move.is_en_passant) {
        state.en_passant(move);
        return;
    }

    state.move_pieces(move);
}

float game_over(const State& state, std::vector<Move>& moves) {
    if (fifty_move_rule(state) || is_insufficient_material(state)) {
        return 0.5f;
    }
    return check_or_stale_mate(state, moves);

}

bool fifty_move_rule(const State& state) {
    return state.fifty_move == 50;
}


uint8_t num_pieces(const State& state) {
    return popcount(state.get_all_occupied_squares());
}

float check_or_stale_mate(const State& state, const std::vector<Move>& moves) {
    if (moves.empty()) {
        if (is_in_check(state)) {
            return (static_cast<Color>(state.toMove) == Color::BLACK) ? 1.0f : -1.0f;
        }
        return 0.5f;  // stalemate
    }
    return 0.0f;
}

std::unordered_map<PieceType, uint8_t> count_pieces(const State& state, Color color) {
    std::unordered_map<PieceType, uint8_t> counts;
    for (PieceType pt : {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN}) {
        counts[pt] = popcount(state.boards[static_cast<uint8_t>(color)][static_cast<uint8_t>(pt)]);
    }
    return counts;
}

bool is_insufficient_material(const State& state) {
    auto get_squares_from_bitboard = [](uint64_t bb) {
        std::vector<uint64_t> squares;
        while (bb) {
            int sq = lsb_index(bb);
            squares.push_back(sq);
            bb &= bb - 1;
        }
        return squares;
    };

    auto is_light_square = [](int sq) {
        return ((sq / 8 + sq % 8) % 2) == 0;
    };

    auto heavy_pieces_or_pawns_exist = [&]() {
        return (state.boards[0][0] |
                state.boards[0][3] |
                state.boards[0][4] |
                state.boards[1][0] |
                state.boards[1][3] |
                state.boards[1][4]) != 0;
    };

    if (heavy_pieces_or_pawns_exist()) {
        return false;
    }

    auto white = count_pieces(state, Color::WHITE);
    auto black = count_pieces(state, Color::BLACK);

    uint8_t white_knights = white[PieceType::KNIGHT];
    uint8_t white_bishops = white[PieceType::BISHOP];
    uint8_t black_knights = black[PieceType::KNIGHT];
    uint8_t black_bishops = black[PieceType::BISHOP];

    int total_minor = white_knights + white_bishops + black_knights + black_bishops;

    if (total_minor <= 1) return true;

    if (total_minor == 2) {
        if (white_knights == 1 && black_knights == 1) return true;
        if ((white_bishops == 1 && black_knights == 1) || (black_bishops == 1 && white_knights == 1)) return true;
        if (white_bishops == 1 && black_bishops == 1) {
            uint64_t white_bishop_sq = get_squares_from_bitboard(state.boards[static_cast<uint8_t>(Color::WHITE)][static_cast<uint8_t>(PieceType::BISHOP)])[0];
            uint64_t black_bishop_sq = get_squares_from_bitboard(state.boards[static_cast<uint8_t>(Color::BLACK)][static_cast<uint8_t>(PieceType::BISHOP)])[0];
            return is_light_square(white_bishop_sq) == is_light_square(black_bishop_sq);
        }
    }

    return false;
}
