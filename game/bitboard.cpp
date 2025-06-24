#include "bitboard.h"
#include "bit_ops.h"

#include <iostream>
#include <optional>

State::State() 
    : toMove(0),
      boards(2, std::vector<uint64_t>(6, 0)),
      castling(0b1111),
      en_passant_sq(Square::NO_SQUARE),
      fifty_move(0)
{}

std::string State::str() const {
    printBoard();
    return ""; 
}

std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t> State::hash() const {
    std::vector<uint64_t> boards_tuple;
    for (int color = 0; color < 2; ++color) {
        for (int piece = 0; piece < 6; ++piece) {
            boards_tuple.push_back(boards[color][piece]);
        }
    }
    return std::make_tuple(toMove, boards_tuple, castling, static_cast<uint64_t>(en_passant_sq));
}

bool State::operator==(const State& other) const {
    return toMove == other.toMove && boards == other.boards && castling == other.castling && en_passant_sq == other.en_passant_sq;
}

void State::reset() {
    boards[0][static_cast<int>(PieceType::PAWN)]   = 0x000000000000FF00ULL;
    boards[1][static_cast<int>(PieceType::PAWN)]   = 0x00FF000000000000ULL;
    boards[0][static_cast<int>(PieceType::KNIGHT)] = 0x0000000000000042ULL;
    boards[1][static_cast<int>(PieceType::KNIGHT)] = 0x4200000000000000ULL;
    boards[0][static_cast<int>(PieceType::BISHOP)] = 0x0000000000000024ULL;
    boards[1][static_cast<int>(PieceType::BISHOP)] = 0x2400000000000000ULL;
    boards[0][static_cast<int>(PieceType::ROOK)]   = 0x0000000000000081ULL;
    boards[1][static_cast<int>(PieceType::ROOK)]   = 0x8100000000000000ULL;
    boards[0][static_cast<int>(PieceType::QUEEN)]  = 0x0000000000000008ULL;
    boards[1][static_cast<int>(PieceType::QUEEN)]  = 0x0800000000000000ULL;
    boards[0][static_cast<int>(PieceType::KING)]   = 0x0000000000000010ULL;
    boards[1][static_cast<int>(PieceType::KING)]   = 0x1000000000000000ULL;

    toMove = 0;
    moves.clear();
    castling = 0b1111;
    en_passant_sq = Square::NO_SQUARE;
    fifty_move = 0;
}

void State::printBoard() const {
    std::vector<std::vector<char>> board(BOARD_SIZE, std::vector<char>(BOARD_SIZE, '.'));
    for (uint64_t color = 0; color < 2; ++color){
        for (uint8_t piece_type = 0; piece_type < 6; ++piece_type){
            uint64_t bb = boards[color][piece_type];
            for (uint64_t sq = 0; sq < 64; ++sq){
                if (get_bit(bb, sq)){
                    uint64_t row = 7 - (sq / 8);
                    uint64_t col = sq % 8;
                    board[row][col] = PIECE_SYMBOLS[color][piece_type];
                }
            }
        }
    }
    std::cout << "\n  +------------------------+\n";
    for (uint64_t i = 0; i < BOARD_SIZE; ++i) {
        std::cout << 8 - i << " | ";
        for (uint64_t j = 0; j < BOARD_SIZE; ++j) {
            std::cout << board[i][j] << "  ";
        }
        std::cout << "|\n";
    }
    std::cout << "  +------------------------+\n";
    std::cout << "    a  b  c  d  e  f  g  h\n";
}

uint64_t State::get_all_occupied_squares() const {
    uint64_t occupied = 0;
    for (int pt = 0; pt < 6; ++pt) {
        occupied |= boards[0][pt] | boards[1][pt];
    }
    return occupied;
}

uint64_t State::get_occupied_by_color(Color color) const {
    uint64_t occupied = 0;
    for (int pt = 0; pt < 6; ++pt) {
        occupied |= boards[static_cast<int>(color)][pt];
    }
    return occupied;
}

std::optional<PieceType> State::piece_on_square(const Square sq) const{
    uint64_t mask = 1ULL << static_cast<int>(sq);
    for (int pt = 0; pt < 6; ++pt) {
        if ((boards[0][pt] | boards[1][pt]) & mask) {
            return static_cast<PieceType>(pt);
        }
    }
    return std::nullopt;
}

std::optional<PieceType> State::piece_on_square_by_color(const Square sq, const Color color) const{
    uint64_t mask = 1ULL << static_cast<int>(sq);
    for (int pt = 0; pt < 6; ++pt) {
        if (boards[static_cast<int>(color)][pt] & mask) {
            return static_cast<PieceType>(pt);
        }
    }
    return std::nullopt;
}

std::optional<Color> State::color_on_square(const Square sq) const{
    uint64_t mask = 1ULL << static_cast<int>(sq);

    for (int color = 0; color < 2; ++color) {
        for (int pt = 0; pt < 6; ++pt) {
            if (boards[color][pt] & mask) {
                return static_cast<Color>(color);
            }
        }
    }
    return std::nullopt;
}

void State::move_pieces(const Move move) {
    uint8_t old_castling = castling;
    Square old_en_passant = en_passant_sq;
    uint8_t old_fifty = fifty_move;
    
    PieceType captured_piece = PieceType::NONE; 
    
    Color opposing_color = static_cast<Color>(toMove ^ 1);
    auto optional_captured_piece = piece_on_square_by_color(move.to_sq, opposing_color);

    if (optional_captured_piece.has_value()) {
        captured_piece = optional_captured_piece.value();
        clear_bit(boards[static_cast<int>(opposing_color)][static_cast<int>(captured_piece)], static_cast<int>(move.to_sq));
    }

    moves.push_back(std::make_tuple(
        move,
        captured_piece,
        old_castling,
        old_en_passant,
        old_fifty
    ));


    uint64_t& bb = boards[toMove][static_cast<int>(move.piece_type)];
    clear_bit(bb, static_cast<int>(move.from_sq));
    set_bit(bb, static_cast<int>(move.to_sq));

    toMove ^= 1;
}

void State::castle(const Move move){
    uint8_t old_castling = castling;
    Square old_en_passant = en_passant_sq;
    uint8_t old_fifty = fifty_move;

    moves.push_back(std::make_tuple(
        move,
        PieceType::NONE,
        old_castling,
        old_en_passant,
        old_fifty
    ));

    Color color = static_cast<Color>(toMove);
    uint64_t& k_b = boards[toMove][5];
    uint64_t& r_b = boards[toMove][3];
    Square to_sq = move.to_sq;

    if (color == Color::WHITE) {
        if (to_sq == Square::G1) { // Kingside
            clear_bit(k_b, static_cast<uint64_t>(Square::E1));
            set_bit(k_b, static_cast<uint64_t>(Square::G1));
            clear_bit(r_b, static_cast<uint64_t>(Square::H1));
            set_bit(r_b, static_cast<uint64_t>(Square::F1));
            castling &= ~0b1100;

        } else if (to_sq == Square::C1) { // Queenside
            clear_bit(k_b, static_cast<uint64_t>(Square::E1));
            set_bit(k_b, static_cast<uint64_t>(Square::C1));
            clear_bit(r_b, static_cast<uint64_t>(Square::A1));
            set_bit(r_b, static_cast<uint64_t>(Square::D1));
            castling &= ~0b1100;
        }
    } else { 
        if (to_sq == Square::G8) { // Kingside
            clear_bit(k_b, static_cast<uint64_t>(Square::E8));
            set_bit(k_b, static_cast<uint64_t>(Square::G8));
            clear_bit(r_b, static_cast<uint64_t>(Square::H8));
            set_bit(r_b, static_cast<uint64_t>(Square::F8));
            castling &= ~0b0011;
            
        } else if (to_sq == Square::C8) { // Queenside
            clear_bit(k_b, static_cast<uint64_t>(Square::E8));
            set_bit(k_b, static_cast<uint64_t>(Square::C8));
            clear_bit(r_b, static_cast<uint64_t>(Square::A8));
            set_bit(r_b, static_cast<uint64_t>(Square::D8));
            castling &= ~0b0011;
        }
    }
    toMove ^= 1;
}

void State::promote(const Move move){
    uint8_t old_castling = castling;
    Square old_en_passant = en_passant_sq;
    uint8_t old_fifty = fifty_move;
    
    PieceType captured_piece = PieceType::NONE; 
    
    Color opposing_color = static_cast<Color>(toMove ^ 1);
    auto optional_captured_piece = piece_on_square_by_color(move.to_sq, opposing_color);

    if (optional_captured_piece.has_value()) {
        captured_piece = optional_captured_piece.value();
        clear_bit(boards[static_cast<int>(opposing_color)][static_cast<int>(captured_piece)], static_cast<int>(move.to_sq));
    }

    moves.push_back(std::make_tuple(
        move,
        captured_piece,
        old_castling,
        old_en_passant,
        old_fifty
    ));

    if (captured_piece != PieceType::NONE){
        uint64_t& opp_bb = boards[toMove ^ 1][static_cast<uint8_t>(captured_piece)];
        clear_bit(opp_bb, static_cast<uint64_t>(move.to_sq));
    }


    clear_bit(boards[toMove][0], static_cast<uint64_t>(move.from_sq));
    set_bit(boards[toMove][static_cast<uint8_t>(move.promotion_type)], static_cast<uint64_t>(move.to_sq));

    toMove ^= 1;
}

void State::en_passant(const Move move){
    uint8_t old_castling = castling;
    Square old_en_passant = en_passant_sq;
    uint8_t old_fifty = fifty_move;
    PieceType captured_piece = PieceType::PAWN;

    moves.push_back(std::make_tuple(
        move,
        PieceType::PAWN,
        old_castling,
        old_en_passant,
        old_fifty
    ));

    uint64_t to_sq_int = static_cast<uint64_t>(move.to_sq);
    uint64_t captured_sq_int = 0;
    if (static_cast<Color>(toMove) == Color::WHITE) { captured_sq_int = to_sq_int - 8; }
    else{ captured_sq_int = to_sq_int + 8; }

    clear_bit(boards[toMove^ 1][0], captured_sq_int);

    clear_bit(boards[toMove][0], static_cast<uint64_t>(move.from_sq));
    set_bit(boards[toMove][0], static_cast<uint64_t>(move.to_sq));

    toMove ^= 1;

}

void State::unmake_move(){
    if (moves.empty()) { return; }


    auto last = moves.back();
    moves.pop_back();

    Move move;
    PieceType captured_piece_;
    uint8_t castle_;
    Square en_pas_;
    uint8_t fifty_;

    std::tie(move, captured_piece_, castle_, en_pas_, fifty_) = last;


    // Restoring old attributes (aside from boards)

    toMove ^= 1;

    castling = castle_;
    en_passant_sq = en_pas_;
    fifty_move = fifty_;
    
    uint64_t from_sq_int = static_cast<uint64_t>(move.from_sq);
    uint64_t to_sq_int = static_cast<uint64_t>(move.to_sq);


    if (move.is_castle){

        uint64_t& k_b = boards[toMove][5];
        uint64_t& r_b = boards[toMove][3];

        if (static_cast<Color>(toMove) == Color::WHITE) {
            if (move.to_sq == Square::G1) { // Kingside
                clear_bit(k_b, static_cast<uint64_t>(Square::G1));
                set_bit(k_b, static_cast<uint64_t>(Square::E1));
                clear_bit(r_b, static_cast<uint64_t>(Square::F1));
                set_bit(r_b, static_cast<uint64_t>(Square::H1));

            } else if (move.to_sq == Square::C1) { // Queenside
                clear_bit(k_b, static_cast<uint64_t>(Square::C1));
                set_bit(k_b, static_cast<uint64_t>(Square::E1));
                clear_bit(r_b, static_cast<uint64_t>(Square::D1));
                set_bit(r_b, static_cast<uint64_t>(Square::A1));
            }
        } else { 
            if (move.to_sq == Square::G8) { // Kingside
                clear_bit(k_b, static_cast<uint64_t>(Square::G8));
                set_bit(k_b, static_cast<uint64_t>(Square::E8));
                clear_bit(r_b, static_cast<uint64_t>(Square::F8));
                set_bit(r_b, static_cast<uint64_t>(Square::H8));
                
            } else if (move.to_sq == Square::C8) { // Queenside
                clear_bit(k_b, static_cast<uint64_t>(Square::C8));
                set_bit(k_b, static_cast<uint64_t>(Square::E8));
                clear_bit(r_b, static_cast<uint64_t>(Square::D8));
                set_bit(r_b, static_cast<uint64_t>(Square::A8));
            }
        }
    }

    else if (move.is_en_passant){
        uint64_t captured_sq_int = 0;
        if (static_cast<Color>(toMove ^ 1) == Color::WHITE) { captured_sq_int = to_sq_int - 8; }
        else{ captured_sq_int = to_sq_int + 8; }

        set_bit(boards[toMove ^ 1][0], captured_sq_int);

        clear_bit(boards[toMove][0], to_sq_int);
        set_bit(boards[toMove][0], from_sq_int);

    }

    else if (move.promotion_type != PieceType::NONE){
        clear_bit(boards[toMove][static_cast<uint8_t>(move.promotion_type)], static_cast<uint64_t>(move.to_sq));
        set_bit(boards[toMove][0], static_cast<uint64_t>(move.from_sq));
    }

    else { // regular non-capture move
        uint8_t piece_int = static_cast<uint8_t>(move.piece_type);
        clear_bit(boards[toMove][piece_int], static_cast<uint64_t>(move.to_sq));

        set_bit(boards[toMove][piece_int], static_cast<uint64_t>(move.from_sq));
    }

    if (move.is_capture && !move.is_en_passant){ // Captured a piece regularly
        set_bit(boards[toMove ^ 1][static_cast<uint8_t>(captured_piece_)], static_cast<uint64_t>(move.to_sq));        
    }

}