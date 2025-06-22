#include "bitboard.h"
#include "cases.cpp"

#include <vector>
#include <iostream>

int main() {
    State state;
    state.reset();

    for (int i = 0; i < EN_PASSANT_SIMPLE.size(); ++i){
        Move move = Move(EN_PASSANT_SIMPLE[i]);
        std::cout<< move.notation(static_cast<Color>(state.toMove)) << std::endl;
        if (move.is_castle){ state.castle(move); }
        else if (move.is_en_passant) { state.en_passant(move); }
        else if (move.promotion_type != PieceType::NONE) { state.promote(move); }
        else{ state.move_pieces(move); }
        state.printBoard();
    }

    std::cout<<"Now, resetting the board move by move" << std::endl;

    for (int i = 0; i < EN_PASSANT_SIMPLE.size(); ++i){
        Move move = Move(EN_PASSANT_SIMPLE[EN_PASSANT_SIMPLE.size() - i - 1]);
        std::cout<< move.notation(static_cast<Color>(state.toMove)) << std::endl;

        state.unmake_move();
        state.printBoard();
        
    }
    return 0;
}
