#include "legal_king_moves.h"
#include "utils.h"
#include "bitboard.h"
#include "bit_ops.h"
#include "constants.h"


#include <vector>


std::vector<Move> castleMoves(const State& state) {
    std::vector<Move> moves;
    uint8_t color = static_cast<uint8_t>(state.toMove);
    uint64_t occupied = state.get_all_occupied_squares();

    if (color == static_cast<uint8_t>(Color::WHITE)) {
        // White kingside
        if (state.castling & 0b0001) {
            if (!(get_bit(occupied, static_cast<uint8_t>(Square::F1)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::G1)))) {
                if (!is_square_attacked(state, static_cast<uint64_t>(Square::E1)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::F1)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::G1))) {
                    moves.push_back(Move (PieceType::KING,
                        Square::E1, Square::G1, false, PieceType::NONE, true));
                }
            }
        }
        // White queenside
        if (state.castling & 0b0010) {
            if (!(get_bit(occupied, static_cast<uint8_t>(Square::B1)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::C1)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::D1)))) {
                if (!is_square_attacked(state, static_cast<uint64_t>(Square::E1)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::D1)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::C1))) {
                    moves.push_back( Move(PieceType::KING,
                        Square::E1, Square::C1, false, PieceType::NONE, true));
                }
            }
        }
    } else {
        // Black kingside
        if (state.castling & 0b0100) {
            if (!(get_bit(occupied, static_cast<uint8_t>(Square::F8)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::G8)))) {
                if (!is_square_attacked(state, static_cast<uint64_t>(Square::E8)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::F8)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::G8))) {
                    moves.emplace_back( Move(PieceType::KING,
                        Square::E8, Square::G8, false, PieceType::NONE, true));
                }
            }
        }
        // Black queenside
        if (state.castling & 0b1000) {
            if (!(get_bit(occupied, static_cast<uint8_t>(Square::B8)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::C8)) || 
                  get_bit(occupied, static_cast<uint8_t>(Square::D8)))) {
                if (!is_square_attacked(state, static_cast<uint64_t>(Square::E8)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::D8)) &&
                    !is_square_attacked(state, static_cast<uint64_t>(Square::C8))) {
                    moves.emplace_back( Move(PieceType::KING,
                        Square::E8, Square::C8, false, PieceType::NONE, true));
                }
            }
        }
    }

    return moves;
}

std::vector<Move> kingMoves(State& state, int attacker_ct) {
    std::vector<Move> moves;

    uint64_t king_bb = state.boards[state.toMove][static_cast<uint8_t>(PieceType::KING)];
    int from_sq_idx = lsb_index(king_bb);

    uint64_t own_occ = state.get_occupied_by_color(static_cast<Color>(state.toMove));
    uint64_t opp_occ = state.get_occupied_by_color(static_cast<Color>(state.toMove ^ 1));

    const int deltas[8] = {-9, -8, -7, -1, +1, +7, +8, +9};

    int from_file = from_sq_idx % 8;
    int from_rank = from_sq_idx / 8;

    for (int d : deltas) {
        int to_sq_idx = from_sq_idx + d;

        if (to_sq_idx >= 0 && to_sq_idx < 64) {
            int to_file = to_sq_idx % 8;
            int to_rank = to_sq_idx / 8;

            if (std::abs(to_file - from_file) <= 1 &&
                std::abs(to_rank - from_rank) <= 1 &&
                !get_bit(own_occ, static_cast<uint8_t>(to_sq_idx))) {

                bool is_capture = get_bit(opp_occ, static_cast<uint8_t>(to_sq_idx));
                Move move(
                    PieceType::KING,
                    static_cast<Square>(from_sq_idx),
                    static_cast<Square>(to_sq_idx),
                    is_capture
                );

                state.move_pieces(move);
                state.toMove ^= 1;

                if (!is_square_attacked(state, static_cast<uint64_t>(to_sq_idx))) {
                    moves.push_back(move);
                }

                state.toMove ^= 1;
                state.unmake_move();
            }
        }
    }

    if (attacker_ct == 0) {
        std::vector<Move> castles = castleMoves(state);
        moves.insert(moves.end(), castles.begin(), castles.end());
    }

    return moves;
}
