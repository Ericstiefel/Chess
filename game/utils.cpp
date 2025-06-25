#include "bit_ops.h"
#include "possible_piece_moves.h"
#include "constants.h"
#include "bitboard.h"
#include "utils.h"

#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>

bool is_in_check(const State& state){
    uint64_t king_sq = lsb_index(state.boards[state.toMove][5]);

    return is_square_attacked(state, king_sq);
}

std::vector<uint64_t> knight_attack_indices(uint64_t square) {
    uint8_t rank = square / 8;
    uint8_t file = square % 8;
    std::vector<uint64_t> attacks;
    const int deltas[8][2] = {
        {-2, -1}, {-2, +1}, {-1, -2}, {-1, +2},
        {+1, -2}, {+1, +2}, {+2, -1}, {+2, +1}
    };

    for (auto [dr, df] : deltas) {
        uint8_t r = rank + dr;
        uint8_t f = file + df;
        if (r >= 0 && r < 8 && f >= 0 && f < 8) {
            attacks.push_back(r * 8 + f);
        }
    }
    return attacks;
}

std::vector<uint64_t> king_attack_indices(uint64_t square) {
    uint8_t rank = square / 8;
    uint8_t file = square % 8;
    std::vector<uint64_t> attacks;
    const int deltas[8][2] = {
        {-1, -1}, {-1, 0}, {-1, +1},
        { 0, -1},          { 0, +1},
        {+1, -1}, {+1, 0}, {+1, +1}
    };

    for (auto [dr, df] : deltas) {
        uint8_t r = rank + dr;
        uint8_t f = file + df;
        if (r >= 0 && r < 8 && f >= 0 && f < 8) {
            attacks.push_back(r * 8 + f);
        }
    }
    return attacks;
}

uint64_t sliding_attacks(uint64_t square, uint64_t occupancy, const std::vector<std::pair<int, int>>& directions) {
    uint64_t attacks = 0;
    uint8_t rank = square / 8;
    uint8_t file = square % 8;

    for (const auto& [dr, df] : directions) {
        uint8_t r = rank + dr;
        uint8_t f = file + df;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            uint8_t idx = r * 8 + f;
            attacks |= (1ULL << idx);
            if (occupancy & (1ULL << idx)) break;
            r += dr;
            f += df;
        }
    }
    return attacks;
}

uint64_t bishop_attacks(uint64_t square, uint64_t occupancy) {
    return sliding_attacks(square, occupancy, {
        {-1, -1}, {-1, +1}, {+1, -1}, {+1, +1}
    });
}

uint64_t rook_attacks(uint64_t square, uint64_t occupancy) {
    return sliding_attacks(square, occupancy, {
        {-1, 0}, {+1, 0}, {0, -1}, {0, +1}
    });
}

bool is_square_attacked(const State& state, uint64_t target_idx) {
    Color by_color = static_cast<Color>(state.toMove ^ 1);

    // 1. Pawn attacks
    uint64_t pawn_bb = state.boards[state.toMove ^ 1][0];
    uint64_t left_attack, right_attack;
    if (by_color == Color::WHITE) {
        left_attack  = (1ULL << target_idx) >> 9 & ~FILE_H;
        right_attack = (1ULL << target_idx) >> 7 & ~FILE_A;
    } else {
        left_attack  = (1ULL << target_idx) << 7 & ~FILE_H;
        right_attack = (1ULL << target_idx) << 9 & ~FILE_A;
    }
    if (pawn_bb & (left_attack | right_attack)) return true;

    // 2. Knight attacks
    for (uint64_t knight_sq : bitscan(state.boards[state.toMove ^ 1][1])) {
        auto attacks = knight_attack_indices(knight_sq);
        if (std::find(attacks.begin(), attacks.end(), target_idx) != attacks.end()) return true;
    }

    // 3. King attacks
    for (uint64_t king_sq : bitscan(state.boards[state.toMove ^ 1][5])) {
        auto attacks = king_attack_indices(king_sq);
        if (std::find(attacks.begin(), attacks.end(), target_idx) != attacks.end()) return true;
    }

    // 4. Sliding piece attacks
    uint64_t occupancy = state.get_all_occupied_squares();

    for (int pt = 2; pt < 5; ++pt) {
        uint64_t attacker_bb = state.boards[state.toMove ^ 1][pt];
        for (uint64_t from_sq : bitscan(attacker_bb)) {
            uint64_t attacks = 0;
            if (pt == 2)
                attacks = bishop_attacks(from_sq, occupancy);
            else if (pt == 3)
                attacks = rook_attacks(from_sq, occupancy);
            else
                attacks = bishop_attacks(from_sq, occupancy) | rook_attacks(from_sq, occupancy);

            if (attacks & (1ULL << target_idx)) return true;
        }
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
