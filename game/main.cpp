#include "bitboard.h"
#include "cases.cpp"
#include "possible_piece_moves.h"
#include "bit_ops.h"
#include "utils.h"
#include "legal_moves.h"
#include "turn.h"
#include "move.h"

#include <vector>
#include <iostream>
#include <ctime>
#include <random>
#include <chrono>


Move return_random_move(const State& state, const std::vector<Move>& moves) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, moves.size() - 1);
    int randomNumber = distrib(gen);

    return moves[randomNumber];

}

int main() {
    // Warmup
    int sum = 0;

    for (int i = 0; i < 1000; ++i){
        sum += i;
    }

    using namespace std::chrono;

    int num_games = 2;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_games; ++i){
        State state;
        state.reset();

        int move_num = 0;

        while (true) {
            std::vector<Move> leg_mo = legal_moves(state);
            if (state.toMove == 0) { ++move_num; }

            float result = game_over(state, leg_mo);
            if (result != 0.0) break;

            Move move = return_random_move(state, leg_mo);
            turn(state, move);

        }

    }

    auto end = high_resolution_clock::now();
    duration<double, std::milli> total_time = end - start;
    std::cout << "Average time per game: " << total_time.count() / num_games << " ms\n";

    return 0;
}
