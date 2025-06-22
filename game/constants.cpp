#include "constants.h"

const std::array<std::array<char, 6>, 2> PIECE_SYMBOLS = {{
    {'P', 'N', 'B', 'R', 'Q', 'K'},  // White
    {'p', 'n', 'b', 'r', 'q', 'k'}   // Black
}};


const std::array<std::string, 64> SQUARE_NAMES = []() {
    std::array<std::string, 64> names{};
    const char* files = "abcdefgh";
    for (int i = 0; i < 64; ++i) {
        names[i] = std::string{files[i % 8]} + std::to_string((i / 8) + 1);
    }
    return names;
}();

const uint64_t RANK_1 = 0x00000000000000FFULL;
const uint64_t RANK_2 = 0x000000000000FF00ULL;
const uint64_t RANK_7 = 0x00FF000000000000ULL;
const uint64_t RANK_8 = 0xFF00000000000000ULL;
const uint64_t FILE_A = 0x0101010101010101ULL;
const uint64_t FILE_H = 0x8080808080808080ULL;

const std::array<int, 8> KNIGHT_OFFSETS = {-17, -15, -10, -6, 6, 10, 15, 17};
