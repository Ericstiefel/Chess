#include "possible_piece_moves.h"
#include "bitboard.h"
#include "constants.h"
#include "move.h"
#include "bit_ops.h"

#include <vector>
#include <iostream>
#include <optional>
#include <cstdlib>
#include <cmath>
#include <array>


std::vector<Move> pawnMoves(const State& state, const uint64_t& opp_occupied_bb){

    std::vector<Move> moves;

    int forward_one, forward_two, attack_left, attack_right;

    uint64_t start_rank, promotion_rank;

    if (static_cast<Color>(state.toMove) == Color::WHITE){
        forward_one = 8;
        forward_two = 16;
        attack_left = 7;
        attack_right = 9;

        start_rank = RANK_2;

        promotion_rank = RANK_8;
    }

    else {
        forward_one = -8;
        forward_two = -16;
        attack_left = -9;
        attack_right = -7;

        start_rank = RANK_7;

        promotion_rank = RANK_1;
    }

    uint64_t all_occupied_bb = state.get_all_occupied_squares();

    uint64_t temp_user_pawns_bb = state.boards[state.toMove][0]; 

    while (temp_user_pawns_bb){
        uint64_t from_sq_idx = lsb_index(temp_user_pawns_bb);
        clear_bit(temp_user_pawns_bb, from_sq_idx);


        // Singular Pawn Push
        uint64_t single_pawn_push_idx = from_sq_idx + forward_one;

        if (!get_bit(all_occupied_bb, single_pawn_push_idx)) {
            if ((1ULL << single_pawn_push_idx) & promotion_rank) {
                for (uint8_t pt = 5; pt > 1; --pt) { // top-down / bottom-up completely arbitrary
                    moves.push_back( Move(
                        PieceType::PAWN,
                        static_cast<Square>(from_sq_idx),
                        static_cast<Square>(single_pawn_push_idx),
                        false,
                        static_cast<PieceType>(pt)
                    ));
                }
            }
            else {
                moves.push_back( Move(
                    PieceType::PAWN,
                    static_cast<Square>(from_sq_idx),
                    static_cast<Square>(single_pawn_push_idx)
                ));

                // Double pawn push
                if ((1ULL << from_sq_idx) & start_rank) {
                    uint64_t push_two_tgt_idx = from_sq_idx + forward_two;
                    if (!get_bit(all_occupied_bb, push_two_tgt_idx)) {
                        moves.push_back( Move(
                            PieceType::PAWN,
                            static_cast<Square>(from_sq_idx),
                            static_cast<Square>(push_two_tgt_idx)                            
                        ));
                    }
                }
            }        
        }
        int offset_arr[2] = {attack_left, attack_right}; // Left & Right capturing
        for (int offset_idx = 0; offset_idx < 2; ++offset_idx){
            uint64_t target_sq_idx = offset_arr[offset_idx] + from_sq_idx;
            // Prevent wraparound
            if ((offset_arr[offset_idx] == attack_left && (1ULL << from_sq_idx) & FILE_A) ||
                (offset_arr[offset_idx] == attack_right && (1ULL << from_sq_idx) & FILE_H)) {
                continue;
            }

            if (((from_sq_idx - target_sq_idx) % 8 == 1) && (get_bit(opp_occupied_bb, target_sq_idx))) {
                if ((1ULL << target_sq_idx) & promotion_rank) {
                    for (uint8_t pt = 5; pt < 1; --pt) { 
                        moves.push_back( Move(
                            PieceType::PAWN,
                            static_cast<Square>(from_sq_idx),
                            static_cast<Square>(target_sq_idx),
                            true,
                            static_cast<PieceType>(pt)
                        ));
                    }

                }

                else {
                    moves.push_back( Move(
                        PieceType::PAWN,
                        static_cast<Square>(from_sq_idx),
                        static_cast<Square>(target_sq_idx)
                    ));
                }
            }
                        
        }
        // En Passant
        if (state.en_passant_sq != Square::NO_SQUARE) {
            uint64_t ep_tgt_sq = static_cast<uint64_t>(state.en_passant_sq);
            uint8_t row = from_sq_idx / 8;
            uint8_t tgt_row;
            if (static_cast<Color>(state.toMove) == Color::WHITE) { tgt_row = 4; }
            else { tgt_row = 3; }

            int offset_arr[2] = {attack_left, attack_right};

            if (row == tgt_row) {
                for (int offset_idx = 0; offset_idx < 2; ++offset_idx){
                    if (from_sq_idx + offset_arr[offset_idx] == ep_tgt_sq) {
                        moves.push_back( Move(
                            PieceType::PAWN, 
                            static_cast<Square>(from_sq_idx),
                            static_cast<Square>(ep_tgt_sq),
                            true,
                            PieceType::NONE,
                            false,
                            false,
                            true
                        ));
                    }
                }
            }
        }

    }

    return moves;

}


std::vector<Move> knightMoves(const State& state, const uint64_t& own_occupied_bb){
    std::vector<Move> moves;

    uint64_t temp_knights_bb = state.boards[state.toMove][static_cast<uint8_t>(PieceType::KNIGHT)];

    while (temp_knights_bb) {
        uint64_t from_sq_idx = lsb_index(temp_knights_bb);

        clear_bit(temp_knights_bb, from_sq_idx);

        uint8_t from_rank = from_sq_idx / 8;
        uint8_t from_file = from_sq_idx % 8;

        for (int i = 0; i < KNIGHT_OFFSETS.size(); ++i) {
            uint64_t target_sq_idx = from_sq_idx + KNIGHT_OFFSETS[i];

            if (target_sq_idx > 63) { continue; } // unsigned anyway (no negative)


            uint8_t to_rank = target_sq_idx / 8;
            uint8_t to_file = target_sq_idx % 8;

            if (std::abs(to_rank - from_rank) + std::abs(to_file - from_file) != 3) {
                continue;
            }
            if (get_bit(own_occupied_bb, target_sq_idx)) { continue; }

            bool is_cap = get_bit(state.get_occupied_by_color(static_cast<Color>(state.toMove ^ 1)), target_sq_idx) != 0; 

            moves.push_back( Move(
                PieceType::KNIGHT, 
                static_cast<Square>(from_sq_idx),
                static_cast<Square>(target_sq_idx),
                is_cap
            ));
        }
    }
    return moves;
}

std::vector<Move> bishopMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb) {
    std::vector<Move> moves;
    uint64_t temp_bishop_bb = state.boards[state.toMove][2];

    int directions[4] = {9, 7, -9, -7};

    while (temp_bishop_bb) {
        uint64_t from_sq_idx = lsb_index(temp_bishop_bb);
        clear_bit(temp_bishop_bb, from_sq_idx);

        for (int i = 0; i < 4; ++i) {

            uint64_t direction = directions[i];

            uint64_t to_sq_idx = from_sq_idx; // Destinations are added shortly

            while (true) {
                uint8_t rank = to_sq_idx / 8;
                uint8_t file = to_sq_idx % 8;

                to_sq_idx += direction;
                
                // Shifting to a square less than 0 is implicitly handled here, as it would become a uint much larger than 64.
                if (to_sq_idx > 63) { break; }

                uint8_t new_rank = to_sq_idx / 8;
                uint8_t new_file = to_sq_idx % 8;

                if (std::abs(new_file - file) != 1) { break; }

                if (get_bit(own_occupied_bb, to_sq_idx)) { break; }

                bool is_cap = get_bit(opp_occupied_bb, to_sq_idx) > 0;

                moves.push_back( Move(
                    PieceType::BISHOP,
                    static_cast<Square>(from_sq_idx),
                    static_cast<Square>(to_sq_idx),
                    is_cap
                ));

                if (is_cap) { break; }


            }

        }
    }

    return moves;
}

std::vector<Move> rookMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb) {
    std::vector<Move> moves;

    uint64_t temp_rook_bb = state.boards[state.toMove][3];

    int directions[4] = {8, 1, -1, -8};

    while (temp_rook_bb) {
        uint64_t from_sq_idx = lsb_index(temp_rook_bb);
        clear_bit(temp_rook_bb, from_sq_idx);

        for (int i = 0; i < 4; ++i) {
            int direction = directions[i];

            uint64_t to_sq_idx = from_sq_idx;

            while (true) {
                uint8_t rank = to_sq_idx / 8;
                to_sq_idx += direction;

                uint8_t new_rank = to_sq_idx / 8;

                if (to_sq_idx > 63) { break; }
                if ((std::abs(direction) == 1) && (rank != new_rank)) { break; }
                if (get_bit(own_occupied_bb, to_sq_idx)) { break; }

                bool is_cap = get_bit(opp_occupied_bb, to_sq_idx) > 0;
                
                moves.push_back( Move(
                    PieceType::ROOK,
                    static_cast<Square>(from_sq_idx),
                    static_cast<Square>(to_sq_idx),
                    is_cap
                ));

                if (is_cap) { break; }

            }
        }
    }
    return moves;

}

std::vector<Move> queenMoves(const State& state, const uint64_t& own_occupied_bb, const uint64_t& opp_occupied_bb) {
    std::vector<Move> moves;

    uint64_t temp_queen_bb = state.boards[state.toMove][4];

    int directions[8] = {8, -8, 1, -1, 9, -9, 7, -7};

    while (temp_queen_bb) {
        uint64_t from_sq_idx = lsb_index(temp_queen_bb);
        clear_bit(temp_queen_bb, from_sq_idx);

        for (int i = 0; i < 8; ++i) {
            int direction = directions[i];
            uint64_t to_sq_idx = from_sq_idx;

            while (true) {
                uint8_t rank = to_sq_idx / 8;
                uint8_t file = to_sq_idx % 8;

                to_sq_idx += direction;
                if (to_sq_idx > 63) break;

                uint8_t new_rank = to_sq_idx / 8;
                uint8_t new_file = to_sq_idx % 8;

                // Prevent wrapping across board edges
                if ((direction == 1 || direction == -1 || direction == 9 || direction == -7) &&
                    new_file == 0 && file == 7)
                    break;
                if ((direction == -1 || direction == 7 || direction == -9) &&
                    new_file == 7 && file == 0)
                    break;

                if (get_bit(own_occupied_bb, to_sq_idx)) break;

                bool is_cap = get_bit(opp_occupied_bb, to_sq_idx) > 0;

                moves.push_back(Move(
                    PieceType::QUEEN,
                    static_cast<Square>(from_sq_idx),
                    static_cast<Square>(to_sq_idx),
                    is_cap
                ));

                if (is_cap) { break; }
            }
        }
    }

    return moves;
}
