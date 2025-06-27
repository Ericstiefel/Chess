#include "bitboard.h"
#include "constants.h"
#include "bit_ops.h"
#include "move.h"
#include "legal_moves.h"
#include "utils.h"

#include <unordered_map>
#pragma once
#include <tuple>
#include <vector>
#include <cstdint>
#include <functional>

void turn(State& state, Move& move);
float game_over(
    const State& state,
    std::vector<Move>& moves,
    std::unordered_map<std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t>, int>& repetition_table
);


bool fifty_move_rule(const State& state);
uint8_t num_pieces(const State& state);
float check_or_stale_mate(const State& state, const std::vector<Move>& moves);
std::unordered_map<PieceType, uint8_t> count_pieces(const State& state, Color color);
bool is_insufficient_material(const State& state);
bool is_threefold_repetition(
    std::unordered_map<std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t>, int>& repetition_table,
    const State& state); 


// Combine hash values
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct std::hash<std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t>> {
    std::size_t operator()(const std::tuple<uint8_t, std::vector<uint64_t>, uint8_t, uint64_t>& tup) const {
        std::size_t seed = 0;

        const auto& [toMove, boards, castling, en_passant] = tup;

        hash_combine(seed, std::hash<uint8_t>{}(toMove));
        for (const auto& b : boards)
            hash_combine(seed, std::hash<uint64_t>{}(b));
        hash_combine(seed, std::hash<uint8_t>{}(castling));
        hash_combine(seed, std::hash<uint64_t>{}(en_passant));

        return seed;
    }
};
