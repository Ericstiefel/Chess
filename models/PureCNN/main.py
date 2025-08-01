import torch
import os
import sys
import numpy as np
from net import Net, state_to_tensor
from maps import load_index_to_move, move_to_idx
from game_py import State, Move, legal_moves, Square, Color, PieceType, annotate_m_w_check, turn

def main():
    # === Configuration ===
    model_path = "PureCNN.pt"
    index_map_path = "index_to_move.txt"
    in_channels = 18
    num_moves = 1972

    # === Load model ===
    if not os.path.exists(model_path):
        print("Model file does not exist!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(
        in_channels=18,
        num_moves=1972,
        hidden_channels=256,     
        num_res_blocks=8,        
        value_fc_dims=(256, 128) 
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # === Load move index map ===
    index_to_move, move_to_index_map = load_index_to_move(index_map_path)

    # === Game Setup ===
    state = State()
    state.reset()

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
            print("\nComputer is thinking...")
            state_tensor = state_to_tensor(state, device)
            with torch.no_grad():
                policy_logits, _ = model(state_tensor)
                policy = torch.nn.functional.softmax(policy_logits, dim=1)[0].cpu().numpy()

            legal_move_indices = [move_to_idx(m, move_to_index_map) for m in legal]
            legal_policy = np.array([policy[i] for i in legal_move_indices])
            policy_sum = np.sum(legal_policy)

            if policy_sum > 0:
                normalized_legal_policy = legal_policy / policy_sum
                move_idx = np.argmax(normalized_legal_policy)
            else:
                move_idx = np.random.randint(len(legal))

            best_move = legal[move_idx]
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
