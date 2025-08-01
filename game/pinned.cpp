#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "utils.h"
#include "pinned.h"

#include <cmath>

bool quick_pin_check(const uint64_t king_sq, const uint64_t piece_sq) {
    const int king_rk = king_sq / 8;
    const int king_fl = king_sq % 8;
    const int piece_rk = piece_sq / 8;
    const int piece_fl = piece_sq % 8;

    if (king_rk == piece_rk || king_fl == piece_fl) { return true;}
    return std::abs(king_rk - piece_rk) == std::abs(king_fl == piece_fl);
}

bool pinned(const State& state, const uint64_t piece_sq, const uint64_t king_sq) {
    if (quick_pin_check(king_sq, piece_sq)) { return true;}
    int ray_directions[8] = {-8, -7, 1, 9, 8, 7, -1, -9};

    uint64_t all_occ = state.get_all_occupied_squares();

    for (int i = 0; i < 8; ++i) {
        int direction = ray_directions[i];

        bool found_piece = false;

        uint64_t current_sq_idx = king_sq + direction;
        
        
        while ((current_sq_idx < 64) && (is_along_ray(king_sq, current_sq_idx, piece_sq))) {
            if (get_bit(all_occ, current_sq_idx)) {
                if (!found_piece) {
                    if (current_sq_idx != piece_sq) { break; }
                    found_piece = true;
                }

                else { 
                    PieceType piece = state.piece_on_square(static_cast<Square>(current_sq_idx));
                    Color color = state.color_on_square(static_cast<Square>(current_sq_idx));

                    if ((piece == PieceType::NONE) || (color == Color::NONE)) break;


                    uint8_t abs_dir = std::abs(direction);

                    if (color != static_cast<Color>(state.toMove)) {
                        if ((abs_dir == 1) || (abs_dir == 8)) {
                            if ((piece == PieceType::ROOK) || (piece == PieceType::QUEEN)) { return true; }
                        }

                        if ((abs_dir == 7) || (abs_dir == 9)) {
                            if ((piece == PieceType::BISHOP) || (piece == PieceType::QUEEN)) { return true; }
                        }

                    }
                    break;
                }
            }
            current_sq_idx += direction;
        }
    }
    return false;
}