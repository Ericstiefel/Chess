#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "bit_ops.h"
#include "constants.h"
#include "utils.h"
#include "move.h"
#include "bitboard.h"
#include "possible_piece_moves.h"
#include "legal_moves.h"
#include "legal_king_moves.h"
#include "pinned.h"
#include "turn.h"

namespace py = pybind11;

PYBIND11_MODULE(game_py, m) {
    m.doc() = "Unified chess engine module from multiple C++ files";

    // ----- bit_ops bindings -----
    m.def("get_bit", &get_bit);
    m.def("set_bit", &set_bit);
    m.def("clear_bit", &clear_bit);
    m.def("lsb_index", &lsb_index);
    m.def("msb_index", &msb_index);
    m.def("popcount", &popcount);
    m.def("bitscan", &bitscan);
    m.def("bitboard_to_squares", &bitboard_to_squares);

    // ----- constants bindings -----
    m.attr("RANK_1") = RANK_1;
    m.attr("RANK_2") = RANK_2;
    m.attr("RANK_7") = RANK_7;
    m.attr("RANK_8") = RANK_8;
    m.attr("FILE_A") = FILE_A;
    m.attr("FILE_B") = FILE_B;
    m.attr("FILE_G") = FILE_G;
    m.attr("FILE_H") = FILE_H;
    m.attr("KNIGHT_OFFSETS") = KNIGHT_OFFSETS;

    m.def("knight_attack_mask", &knight_attack_mask);
    m.def("king_attack_mask", &king_attack_mask);
    m.def("bishop_attack_mask", &bishop_attack_mask);
    m.def("rook_attack_mask", &rook_attack_mask);

    // ----- enums from constants.h -----
    py::enum_<Square>(m, "Square")
        .value("A1", Square::A1).value("B1", Square::B1).value("C1", Square::C1).value("D1", Square::D1)
        .value("E1", Square::E1).value("F1", Square::F1).value("G1", Square::G1).value("H1", Square::H1)
        .value("A2", Square::A2).value("B2", Square::B2).value("C2", Square::C2).value("D2", Square::D2)
        .value("E2", Square::E2).value("F2", Square::F2).value("G2", Square::G2).value("H2", Square::H2)
        .value("A3", Square::A3).value("B3", Square::B3).value("C3", Square::C3).value("D3", Square::D3)
        .value("E3", Square::E3).value("F3", Square::F3).value("G3", Square::G3).value("H3", Square::H3)
        .value("A4", Square::A4).value("B4", Square::B4).value("C4", Square::C4).value("D4", Square::D4)
        .value("E4", Square::E4).value("F4", Square::F4).value("G4", Square::G4).value("H4", Square::H4)
        .value("A5", Square::A5).value("B5", Square::B5).value("C5", Square::C5).value("D5", Square::D5)
        .value("E5", Square::E5).value("F5", Square::F5).value("G5", Square::G5).value("H5", Square::H5)
        .value("A6", Square::A6).value("B6", Square::B6).value("C6", Square::C6).value("D6", Square::D6)
        .value("E6", Square::E6).value("F6", Square::F6).value("G6", Square::G6).value("H6", Square::H6)
        .value("A7", Square::A7).value("B7", Square::B7).value("C7", Square::C7).value("D7", Square::D7)
        .value("E7", Square::E7).value("F7", Square::F7).value("G7", Square::G7).value("H7", Square::H7)
        .value("A8", Square::A8).value("B8", Square::B8).value("C8", Square::C8).value("D8", Square::D8)
        .value("E8", Square::E8).value("F8", Square::F8).value("G8", Square::G8).value("H8", Square::H8)
        .value("NO_SQUARE", Square::NO_SQUARE);

    py::enum_<PieceType>(m, "PieceType")
        .value("PAWN", PieceType::PAWN)
        .value("KNIGHT", PieceType::KNIGHT)
        .value("BISHOP", PieceType::BISHOP)
        .value("ROOK", PieceType::ROOK)
        .value("QUEEN", PieceType::QUEEN)
        .value("KING", PieceType::KING)
        .value("NONE", PieceType::NONE);

    py::enum_<Color>(m, "Color")
        .value("WHITE", Color::WHITE)
        .value("BLACK", Color::BLACK);

    // ----- Move class binding -----
    py::class_<Move>(m, "Move")
    .def(py::init<>())
    .def(py::init<PieceType, Square, Square, bool,
                  PieceType, bool, bool, bool>())  
    .def_readwrite("piece_type", &Move::piece_type)
    .def_readwrite("from_sq", &Move::from_sq)
    .def_readwrite("to_sq", &Move::to_sq)
    .def_readwrite("is_capture", &Move::is_capture)
    .def_readwrite("promotion_type", &Move::promotion_type)
    .def_readwrite("is_castle", &Move::is_castle)
    .def_readwrite("is_check", &Move::is_check)
    .def_readwrite("is_en_passant", &Move::is_en_passant)
    .def("notation", &Move::notation)
    .def("str", &Move::str)
    .def("__eq__", &Move::operator==)
    .def("__hash__", &Move::hash);



    // ----- State class binding (bitboard.h) -----
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("boards", &State::boards)
        .def_readwrite("toMove", &State::toMove)
        .def_readwrite("castling", &State::castling)
        .def_readwrite("en_passant_sq", &State::en_passant_sq)
        .def_readwrite("fifty_move", &State::fifty_move)
        .def_readwrite("repetition_table", &State::repetition_table)
        .def("str", &State::str)
        .def("hash", &State::hash)
        .def("reset", &State::reset)
        .def("printBoard", &State::printBoard)
        .def("get_string_key", &State::get_string_key)
        .def("get_all_occupied_squares", &State::get_all_occupied_squares)
        .def("get_occupied_by_color", &State::get_occupied_by_color)
        .def("piece_on_square", &State::piece_on_square)
        .def("piece_on_square_by_color", &State::piece_on_square_by_color)
        .def("color_on_square", &State::color_on_square)
        .def("move_pieces", &State::move_pieces)
        .def("castle", &State::castle)
        .def("promote", &State::promote)
        .def("en_passant", &State::en_passant)
        .def("unmake_move", &State::unmake_move)
        .def("__eq__", &State::operator==);

    // ----- possible_piece_moves bindings -----
    m.def("pawnMoves", &pawnMoves);
    m.def("knightMoves", &knightMoves);
    m.def("bishopMoves", &bishopMoves);
    m.def("rookMoves", &rookMoves);
    m.def("queenMoves", &queenMoves);

    // ----- legal_moves bindings -----
    m.def("annotate_m_w_check", &annotate_m_w_check);
    m.def("legal_moves", &legal_moves);

    // ----- legal_king_moves bindings -----
    m.def("castleMoves", &castleMoves);
    m.def("kingMoves", &kingMoves);

    // ----- pinned bindings -----
    m.def("pinned", &pinned);

    // ----- utils bindings -----
    m.def("pawn_attacks_from", &pawn_attacks_from);
    m.def("knight_attacks_from", &knight_attacks_from);
    m.def("king_attacks_from", &king_attacks_from);
    m.def("get_bishop_attacks", &get_bishop_attacks);
    m.def("get_rook_attacks", &get_rook_attacks);
    m.def("is_square_attacked", &is_square_attacked);
    m.def("is_in_check", &is_in_check);
    m.def("attackers_to_square", &attackers_to_square);
    m.def("squares_between", &squares_between);
    m.def("is_along_ray", &is_along_ray);

    // ----- turn bindings -----
    m.def("turn", &turn);
    m.def("game_over", &game_over);
    m.def("fifty_move_rule", &fifty_move_rule);
    m.def("num_pieces", &num_pieces);
    m.def("check_or_stale_mate", &check_or_stale_mate);
    m.def("count_pieces", &count_pieces);
    m.def("is_insufficient_material", &is_insufficient_material);
    m.def("is_threefold_repetition", &is_threefold_repetition);
}