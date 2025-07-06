#include "turn.h"

#include <vector>
#include <iostream>
#include <functional>
#include <tuple>
#include <cstdint>


void turn(State& state, Move& move) {
    
    if (move.piece_type == PieceType::PAWN || move.is_capture || move.is_castle) { state.fifty_move = 0; }

    else { if (static_cast<Color>(state.toMove) == Color::WHITE) {state.fifty_move += 1; }}

    if (move.promotion_type != PieceType::NONE) {
        state.promote(move);
        return;
    }

    else if (move.is_castle) {
        state.castle(move);
        if (static_cast<Color>(state.toMove) == Color::WHITE) { state.castling &= 0b1100; }
        else { state.castling &= 0b0011; }
        return;
    }

    else if (move.is_en_passant) {
        state.en_passant(move);
        return;
    }

    if (state.castling && move.piece_type == PieceType::KING) {
        if (static_cast<Color>(state.toMove) == Color::WHITE) { state.castling &= 0b1100; }
        else { state.castling &= 0b0011; }
    }

    state.move_pieces(move);
}

float game_over(
    State& state,
    std::vector<Move>& moves
) {
    if (is_threefold_repetition(state)) {
        return 0.5f;
    }

    if (fifty_move_rule(state)) {
        return 0.5f;
    }

    if (is_insufficient_material(state)) {
        return 0.5f;
    }

    return check_or_stale_mate(state, moves);
}


bool is_threefold_repetition(State& state) {
    std::tuple<uint8_t, std::array<uint64_t, 12>, uint8_t, uint64_t> hash = state.hash();
    state.repetition_table[hash]++;
    return state.repetition_table[hash] >= 3;
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

std::array<uint8_t, 12> count_pieces(const State& state) {
    std::array<uint8_t, 12> counts = {};
    for (int color = 0; color < 2; ++color) {
        for (int pt = 0; pt < 6; ++pt) {
            counts[color * 6 + pt] = popcount(state.boards[color * 6 + pt]);
        }
    }
    return counts;
}


bool is_insufficient_material(const State& state) {
    const auto get_squares_from_bitboard = [](uint64_t bb) -> std::vector<uint64_t> {
        std::vector<uint64_t> squares;
        while (bb) {
            int sq = lsb_index(bb);
            squares.push_back(sq);
            bb &= bb - 1;
        }
        return squares;
    };


    const auto is_light_square = [](int sq) {
        return ((sq / 8 + sq % 8) % 2) == 0;
    };

    const auto heavy_pieces_or_pawns_exist = [&]() {
        return (state.boards[0] | state.boards[3] | state.boards[4] |
                state.boards[6] | state.boards[9] | state.boards[10]) != 0;
    };

    if (heavy_pieces_or_pawns_exist()) {
        return false;
    }

    std::array<uint8_t, 12> counts = count_pieces(state);

    uint8_t white_knights = counts[1];
    uint8_t white_bishops = counts[2];
    uint8_t black_knights = counts[7];
    uint8_t black_bishops = counts[8];

    int total_minor = white_knights + white_bishops + black_knights + black_bishops;

    if (total_minor <= 1) return true;

    if (total_minor == 2) {
        if (white_knights == 1 && black_knights == 1) return true;
        if ((white_bishops == 1 && black_knights == 1) || (black_bishops == 1 && white_knights == 1)) return true;
        if (white_bishops == 1 && black_bishops == 1) {
            std::vector<uint64_t> white_bishop_sq = get_squares_from_bitboard(state.boards[static_cast<uint8_t>(PieceType::BISHOP)]);
            std::vector<uint64_t> black_bishop_sq = get_squares_from_bitboard(state.boards[6 + static_cast<uint8_t>(PieceType::BISHOP)]);

            if (!white_bishop_sq.empty() && !black_bishop_sq.empty()) {
                return is_light_square(white_bishop_sq[0]) == is_light_square(black_bishop_sq[0]);
            }
        }
    }

    return false;
}
