from game_py import Move, PieceType, Square, Color


 # Uses Dict for O(1) access

def load_index_to_move(filename):
    index_to_move_map = {}
    move_to_index_map = {}

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0])
            piece_type = int(parts[1])
            from_sq = int(parts[2])
            to_sq = int(parts[3])
            is_capture = bool(int(parts[4]))
            promotion_type = int(parts[5])
            is_castle = bool(int(parts[6]))

            move = Move(
                PieceType(piece_type),
                Square(from_sq),
                Square(to_sq),
                is_capture,
                PieceType(promotion_type),
                is_castle,
                False,  # is_check
                False   # is_en_passant
            )

            index_to_move_map[idx] = move
            move_to_index_map[(move.from_sq, move.to_sq, move.is_castle, move.promotion_type)] = idx

    return index_to_move_map, move_to_index_map


def move_to_idx(move, move_to_index_map):
    key = (move.from_sq, move.to_sq, move.is_castle, move.promotion_type)
    try:
        return move_to_index_map[key]
    except KeyError:
        raise ValueError(f"move_to_idx: Move {move} not found.")
