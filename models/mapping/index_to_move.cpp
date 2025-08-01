// The NN output distribution is going to be out of 1972 indicies (the total possible number of moves in any position)
// This file is a way to convert the index it chose to move back to a playable move

#include <iostream> 
#include <fstream>
#include "../../game/move.h"
#include "../../game/bitboard.h"
#include "../../game/possible_piece_moves.h"
#include "../../game/constants.h"
#include "../../game/utils.h"
#include "../../game/bit_ops.h"
#include "../../game/legal_moves.h"
#include "index_to_move.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(index_to_move, m) {
    m.def("generate_all_moves", &generate_all_moves);
    m.def("save_index_to_move", &save_index_to_move);
    m.def("load_index_to_move", &load_index_to_move);
    m.def("move_to_idx", &move_to_idx);
}


Move index_to_move[1972];
void generate_all_moves() {
    int index = 0;

    // 0–3: Castling moves
    index_to_move[index++] = Move(PieceType::KING, Square::E1, Square::G1, false, PieceType::NONE, true);
    index_to_move[index++] = Move(PieceType::KING, Square::E1, Square::C1, false, PieceType::NONE, true);
    index_to_move[index++] = Move(PieceType::KING, Square::E8, Square::G8, false, PieceType::NONE, true);
    index_to_move[index++] = Move(PieceType::KING, Square::E8, Square::C8, false, PieceType::NONE, true);

    std::cout << "Generating queenMoves: " << std::endl;
    // 4–1473: Queen-like moves
    for (int sq = 0; sq < 64; ++sq) {
        
        State state;
        state.boards[0][4] = 1ULL << sq;

        std::vector<Move> moves = queenMoves(state, 0, 0);

        for (const Move& m : moves) {
            index_to_move[index++] = m;

        }
    }
    std::cout << "Index: " << index << std::endl;
    std::cout << "Generating knightMoves: " << std::endl;

    for (int sq = 0; sq < 64; ++sq) {
        
        State state;
        state.boards[0][1] = 1ULL << sq;

        std::vector<Move> moves = knightMoves(state, 0); // white knight at sq

        for (const Move& m : moves) {
            index_to_move[index++] = m;
        }
    }
    std::cout << "Index: " << index << std::endl;


        
    std::cout << "Generating pawnMoves: " << std::endl;
    // 1470–1857: Promotions
    PieceType promos[4] = { PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN };

    // White promotions
    for (int file = 0; file < 8; ++file) {
        
        int from_sq = 6 * 8 + file;
        int forward = from_sq + 8;
        int left = (file > 0) ? from_sq + 7 : -1;
        int right = (file < 7) ? from_sq + 9 : -1;

        for (PieceType p : promos) {
            index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(forward), false, static_cast<PieceType>(p));
            if (left != -1) index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(left), true, static_cast<PieceType>(p));
            if (right != -1) index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(right), true, static_cast<PieceType>(p));
        }
    }
    // Black promotions
    for (int file = 0; file < 8; ++file) {
        
        int from_sq = 1 * 8 + file;
        int forward = from_sq - 8;
        int left = (file > 0) ? from_sq - 9 : -1;
        int right = (file < 7) ? from_sq - 7 : -1;

        for (PieceType p : promos) {
            index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(forward), false, static_cast<PieceType>(p));
            if (left != -1) index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(left), true, static_cast<PieceType>(p));
            if (right != -1) index_to_move[index++] = Move(PieceType::PAWN, static_cast<Square>(from_sq), static_cast<Square>(right), true, static_cast<PieceType>(p));
        }
    }
    std::cout << "Index: " << index << std::endl;

    std::cout << "Total Num Moves Generated: " << index;

}



void save_index_to_move(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    for (int i = 0; i < 1972; ++i) {
        const Move& m = index_to_move[i];
        out << i << " "
            << static_cast<int>(m.piece_type) << " "
            << static_cast<int>(m.from_sq) << " "
            << static_cast<int>(m.to_sq) << " "
            << m.is_capture << " "
            << static_cast<int>(m.promotion_type) << " "
            << m.is_castle << " "
            << m.is_check << " "
            << m.is_en_passant << "\n";
            
    }

    out.close();
}
