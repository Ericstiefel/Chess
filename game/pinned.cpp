#include "state.h"
#include "constants.h"
#include "bit_ops.h"
#include "utils.h"
#include "pinned.h"

#include <cmath>

bool pinned(const State& state, const uint64_t piece_sq, const uint64_t king_sq) {
    int ray_directions[8] = {-8, -7, 1, 9, 8, 7, -1, -9};

    uint64_t all_occ = state.get_all_occupied_squares();

    for (int i = 0; i < 8; ++i) {
        int direction = ray_directions[i];

        bool found_piece = false;

        uint64_t current_sq_idx = king_sq + direction;
        
        
        while ((current_sq_idx < 64) && (is_along_ray(king_sq, current_sq, piece_sq))) {
            if (get_bit(all_occ, current_sq_idx)) {
                if (!found_piece) {
                    if (current_sq != piece_sq) { break; }
                    found_piece = true;
                }

                else { 
                    PieceType attacker_piece = state.piece_on_square(static_cast<Square>(current_sq));
                    Color attacker_color = state.color_on_square(static_cast<Square>(current_sq));

                    uint8_t abs_dir = std::abs(direction);

                    if (attacker_color != static_cast<Color>(state.toMove)) {
                        if ((abs_dir == 1) || (abs_dir == 8)) {
                            if ((attacker_piece == PieceType::ROOK) || (attacker_piece == PieceType::QUEEN)) { return true; }
                        }

                        if ((abs_dir == 7) || (abs_dir == 9)) {
                            if ((attacker_piece == PieceType::BISHOP) || (attacker_piece == PieceType::QUEEN)) { return true; }
                        }

                    }
                    break;
                }
            }
            current_sq += direction;
        }
    }
    return false;
}