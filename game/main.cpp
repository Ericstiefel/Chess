#include "bitboard.h"
#include "cases.cpp"
#include "possible_piece_moves.h"

#include <vector>
#include <iostream>

int main() {
    State state;
    state.reset();
    state.move_pieces(Move(PieceType::PAWN, Square::E2, Square::E4));
    state.move_pieces(Move(PieceType::PAWN, Square::E7, Square::E5));
    state.move_pieces(Move(PieceType::PAWN, Square::H2, Square::H4));
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

    std::vector<Move> moves = queenMoves(state, own_occupied_bb, opp_occupied_bb);

    for (int i = 0; i < moves.size(); ++i){
        Move move = moves[i];

        state.move_pieces(move);
        state.printBoard();

        std::cout << move.notation(static_cast<Color>(state.toMove)) << std::endl;

        state.unmake_move();
        
    }
    return 0;
}
