#include "bit_ops.h"
#include "possible_piece_moves.h"
#include "constants.h"
#include "bitboard.h"
#include "utils.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>

bool is_in_check(const State& state) {
    if (state.boards[static_cast<int>(state.toMove)][5] == 0) return false;
    uint64_t king_sq = lsb_index(state.boards[static_cast<int>(state.toMove)][5]);
    Color opponent = static_cast<Color>(state.toMove ^ 1);
    return is_square_attacked(state, king_sq);
}


uint64_t knight_attacks_from(uint64_t square) {
    uint64_t b = 1ULL << square;
    uint64_t attacks = 0;
    attacks |= (b << 17) & ~FILE_A;
    attacks |= (b << 15) & ~FILE_H;
    attacks |= (b << 10) & ~(FILE_A | FILE_B);
    attacks |= (b <<  6) & ~(FILE_H | FILE_G);
    attacks |= (b >>  6) & ~FILE_A;
    attacks |= (b >> 10) & ~FILE_H;
    attacks |= (b >> 15) & ~(FILE_A | FILE_B);
    attacks |= (b >> 17) & ~(FILE_H | FILE_G);
    return attacks;
}


uint64_t king_attacks_from(uint64_t square) {
    uint64_t b = 1ULL << square;
    uint64_t attacks = 0;
    attacks |= (b << 8);
    attacks |= (b >> 8);
    attacks |= (b << 1) & ~FILE_A;
    attacks |= (b >> 1) & ~FILE_H;
    attacks |= (b << 9) & ~FILE_A;
    attacks |= (b << 7) & ~FILE_H;
    attacks |= (b >> 7) & ~FILE_A;
    attacks |= (b >> 9) & ~FILE_H;
    return attacks;
}


uint64_t pawn_attacks_from(uint64_t square, Color pawn_color) {
    uint64_t b = 1ULL << square;
    uint64_t attacks = 0;
    if (pawn_color == Color::WHITE) {
        attacks |= (b << 7) & ~FILE_H; // Attack North-West
        attacks |= (b << 9) & ~FILE_A; // Attack North-East
    } else {
        attacks |= (b >> 9) & ~FILE_H; // Attack South-West
        attacks |= (b >> 7) & ~FILE_A; // Attack South-East
    }
    return attacks;
}


uint64_t sliding_attacks(uint64_t, uint64_t, const std::vector<std::pair<int, int>>&); 

uint64_t get_bishop_attacks(uint64_t square, uint64_t occupancy) {
    return sliding_attacks(square, occupancy, {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});
}

uint64_t get_rook_attacks(uint64_t square, uint64_t occupancy) {
    return sliding_attacks(square, occupancy, {{-1, 0}, {1, 0}, {0, -1}, {0, 1}});
}

// Your original implementation of sliding_attacks
uint64_t sliding_attacks(uint64_t square, uint64_t occupancy, const std::vector<std::pair<int, int>>& directions) {
    uint64_t attacks = 0;
    for (const auto& [dr, df] : directions) {
        int r = (square / 8) + dr;
        int f = (square % 8) + df;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            uint64_t idx = r * 8 + f;
            attacks |= (1ULL << idx);
            if (occupancy & (1ULL << idx)) break;
            r += dr;
            f += df;
        }
    }
    return attacks;
}


bool is_square_attacked(const State& state, uint64_t target_idx) {
    Color attacker_color = static_cast<Color>(state.toMove ^ 1);
    Color defender_color = static_cast<Color>(state.toMove);


    const auto& attacker_boards_vec = state.boards[static_cast<int>(state.toMove ^ 1)];


    if (pawn_attacks_from(target_idx, defender_color) & attacker_boards_vec[0]) {
        return true;
    }

    if (knight_attacks_from(target_idx) & attacker_boards_vec[1]) {
        return true;
    }

    if (king_attacks_from(target_idx) & attacker_boards_vec[5]) {
        return true;
    }


    uint64_t occupancy = state.get_all_occupied_squares();
    uint64_t bishop_queen = attacker_boards_vec[2] | attacker_boards_vec[4];
    uint64_t rook_queen = attacker_boards_vec[3] | attacker_boards_vec[4];

    
    if (get_bishop_attacks(target_idx, occupancy) & bishop_queen) {
        return true;
    }
    if (get_rook_attacks(target_idx, occupancy) & rook_queen) {
        return true;
    }


    return false;
}

uint64_t attackers_to_square(State& state, uint64_t target_sq) {
    uint64_t attackers = 0;


    uint64_t target_bb = 1ULL << target_sq;
    uint64_t all_occ = state.get_all_occupied_squares();

    // PAWN attacks
    uint64_t pawn_bb = state.boards[state.toMove ^ 1][0];
    if (static_cast<Color>(state.toMove ^ 1) == Color::WHITE) {
        if (target_sq >= 9 && target_sq % 8 != 0)
            attackers |= pawn_bb & (1ULL << (target_sq - 9)); // capture from SE
        if (target_sq >= 7 && target_sq % 8 != 7)
            attackers |= pawn_bb & (1ULL << (target_sq - 7)); // capture from SW
    } else {
        if (target_sq <= 55 && target_sq % 8 != 0)
            attackers |= pawn_bb & (1ULL << (target_sq + 7)); // capture from NW
        if (target_sq <= 54 && target_sq % 8 != 7)
            attackers |= pawn_bb & (1ULL << (target_sq + 9)); // capture from NE
    }

    // KNIGHT attacks
    uint64_t knight_bb = state.boards[(state.toMove ^ 1)][1];
    for (uint64_t from_sq : bitscan(knight_bb)) {
        if (knight_attack_mask(from_sq) & target_bb)
            attackers |= (1ULL << from_sq);
    }

    // KING attacks
    uint64_t king_bb = state.boards[(state.toMove ^ 1)][5];
    for (uint64_t from_sq : bitscan(king_bb)) {
        if (king_attack_mask(from_sq) & target_bb)
            attackers |= (1ULL << from_sq);
    }

    // BISHOP + QUEEN (diagonal) attacks
    uint64_t bishop_like = state.boards[(state.toMove ^ 1)][2] | state.boards[(state.toMove ^ 1)][4];
    for (uint64_t from_sq : bitscan(bishop_like)) {
        if (bishop_attack_mask(from_sq, all_occ) & target_bb)
            attackers |= (1ULL << from_sq);
    }

    // ROOK + QUEEN (orthogonal) attacks
    uint64_t rook_like = state.boards[(state.toMove ^ 1)][3] | state.boards[(state.toMove ^ 1)][4];
    for (uint64_t from_sq : bitscan(rook_like)) {
        if (rook_attack_mask(from_sq, all_occ) & target_bb)
            attackers |= (1ULL << from_sq);
    }

    return attackers;
}



uint64_t squares_between(uint64_t from_sq, uint64_t to_sq) {
    uint8_t from_rank = from_sq / 8;
    uint8_t from_file = from_sq % 8;
    uint8_t to_rank = to_sq / 8;
    uint8_t to_file = to_sq % 8;

    int dr = to_rank - from_rank;
    int df = to_file - from_file;

    // Check alignment: same rank, file, or diagonal
    if (dr != 0 && df != 0 && std::abs(dr) != std::abs(df)) {
        return 0ULL;
    }

    if (dr != 0) dr /= std::abs(dr);
    if (df != 0) df /= std::abs(df);

    int direction = dr * 8 + df;
    uint64_t bb = 0ULL;
    int current_sq = from_sq + direction;

    while (current_sq != to_sq) {
        if (current_sq < 0 || current_sq >= 64) break;
        bb |= (1ULL << current_sq);
        current_sq += direction;
    }

    return bb;
}


bool is_along_ray(uint64_t origin, uint64_t from_sq, uint64_t to_sq) {
    int r0 = origin / 8, f0 = origin % 8;
    int r1 = from_sq / 8, f1 = from_sq % 8;
    int r2 = to_sq / 8, f2 = to_sq % 8;

    int dr1 = r1 - r0, df1 = f1 - f0;
    int dr2 = r2 - r1, df2 = f2 - f1;

    // Check if cross product is zero → vectors are collinear
    return dr1 * df2 == df1 * dr2;
}
