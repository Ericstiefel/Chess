import torch
import os
import sys
import numpy as np
import copy

from mcts import MCTS
from net import Net, state_to_tensor
from maps import load_index_to_move, move_to_idx
from game_py import State, Move, legal_moves, Square, Color, PieceType, annotate_m_w_check, turn


def main():
    # === Configuration ===
    model_path = "AlphaGoZero.pt"
    index_map_path = "index_to_move.txt"
    in_channels = 18
    num_moves = 1972

    # === Load model ===
    if not os.path.exists(model_path):
        print("Model file does not exist!")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = Net(in_channels, num_moves)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    model.eval()

    index_to_move_map, move_to_index_map = load_index_to_move(index_map_path)

    # === Game Setup ===
    state = State()
    state.reset()

    
    mcts = MCTS(model, device, move_to_index_map)

    print("Welcome to Chess! You are playing as White.")
    state.printBoard()

    while True:
        legal = legal_moves(state)
        if not legal:
            print("Game over! No legal moves.")
            break
        print("Color to Move: ", "White" if Color(state.toMove) == Color.WHITE else "Black")

        move_notations = [move.notation(Color(state.toMove)) for move in legal]
        print(", ".join(move_notations))

        if Color(state.toMove) == Color.WHITE:
            
            # === Human Turn ===
            move_input = input("\nEnter your move : ").strip().lower()
            matched_move = None
            for move in legal:
                if move.notation(Color.WHITE).lower() == move_input:
                    matched_move = move
                    break
            if not matched_move:
                print("Invalid move. Try again.")
                continue
            turn(state, matched_move)
        else:
            # === AI Turn ===
            # === AI Turn ===
            print("\nComputer is thinking brrrrrrr...")
            mcts.search_mini_batches(state, batches=100, model=model)
            policy, _ = mcts.get_policy(state, tau=0.1)

            best_idx = np.argmax(policy)
            best_move = index_to_move_map[int(best_idx)]
            print(best_move.notation(Color.BLACK))

            annotate_m_w_check(state, [best_move])
            if best_move.piece_type == PieceType.PAWN and best_move.to_sq == state.en_passant_sq:
                best_move.is_en_passant = True

            print(f"AI plays: {best_move.notation(Color.BLACK)}")
            turn(state, best_move)

        # Print board after each move
        state.printBoard()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}")
        sys.exit(1)
