#include "../../game/move.h"
#include "../../game/bitboard.h"
#include "../../game/possible_piece_moves.h"
#include "../../game/constants.h"
#include "../../game/utils.h"
#include "../../game/legal_moves.h"
#include "../../game/bit_ops.h"

#include <fstream>

void save_index_to_move(const std::string& filename);
void load_index_to_move(const std::string& filename);
void generate_all_moves();