#include "bitboard.h"
#include "cases.cpp"
#include "possible_piece_moves.h"
#include "bit_ops.h"
#include "utils.h"

#include <vector>
#include <iostream>

int main() {
    State state;
    state.reset();
    state.move_pieces(Move(PieceType::PAWN, Square::E2, Square::E4));
    std::cout<< "D5: " << is_square_attacked(state, static_cast<uint64_t>(Square::D5)) << std::endl;
    std::cout<< "E5: " << is_square_attacked(state, static_cast<uint64_t>(Square::E5)) << std::endl;
    state.move_pieces(Move(PieceType::PAWN, Square::E7, Square::E5));
    std::cout<< "D4: " << is_square_attacked(state, static_cast<uint64_t>(Square::D4)) << std::endl;
    state.move_pieces(Move(PieceType::PAWN, Square::G2, Square::G4));
    std::cout<< "F3: " << is_square_attacked(state, static_cast<uint64_t>(Square::F3)) << std::endl;
    state.move_pieces(Move(PieceType::PAWN, Square::A7, Square::A5));
    state.printBoard();

    uint64_t own_occupied_bb = state.get_occupied_by_color(static_cast<Color>(state.toMove));
    uint64_t opp_occupied_bb = state.get_occupied_by_color(static_cast<Color>(state.toMove));

    // for (int i = 0; i < EN_PASSANT_SIMPLE.size(); ++i){
    //     Move move = Move(EN_PASSANT_SIMPLE[i]);
    //     std::cout<< move.notation(static_cast<Color>(state.toMove)) << std::endl;
    //     if (move.is_castle){ state.castle(move); }
    //     else if (move.is_en_passant) { state.en_passant(move); }
    //     else if (move.promotion_type != PieceType::NONE) { state.promote(move); }
    //     else{ state.move_pieces(move); }
    //     state.printBoard();
    // }

    // std::cout<<"Now, resetting the board move by move" << std::endl;

    // for (int i = 0; i < EN_PASSANT_SIMPLE.size(); ++i){
    //     Move move = Move(EN_PASSANT_SIMPLE[EN_PASSANT_SIMPLE.size() - i - 1]);
    //     std::cout<< move.notation(static_cast<Color>(state.toMove)) << std::endl;

    //     state.unmake_move();
    //     state.printBoard();

    // std::vector<Move> moves = queenMoves(state, own_occupied_bb, opp_occupied_bb);

    // for (int i = 0; i < moves.size(); ++i){
    //     Move move = moves[i];

    //     state.move_pieces(move);
    //     state.printBoard();

    //     std::cout << move.notation(static_cast<Color>(state.toMove)) << std::endl;

    //     state.unmake_move();
        
    // }


    return 0;
    // return 0;
}
