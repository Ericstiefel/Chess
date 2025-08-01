#ifndef MOVE_MAP_H
#define MOVE_MAP_H

#include <string>
#include <map>
#include <tuple>
#include <stdexcept>
#include <pybind11/pybind11.h>

#include "constants.h" 
#include "move.h"     


class MoveMapping {
public:
    bool load_from_file(const std::string& filename);

    int get_index_for_move(const Move& move) const;

    Move get_move_for_index(int index) const;

private:
    std::map<int, Move> index_to_move;
    
    std::map<std::tuple<int, int, bool, int>, int> move_to_index;
};

void load_move_map_from_python(const std::string& filename);
int get_index_for_move_from_python(const Move& move);
Move get_move_for_index_from_python(int index);

#endif // MOVE_MAP_H