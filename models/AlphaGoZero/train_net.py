import torch
import random
import numpy as np
from collections import deque
import time
import os

from net import Net, state_to_tensor
from maps import move_to_idx, load_index_to_move
from game_py import State, legal_moves, game_over, turn, Color
from mse_loss import MSELoss
from mcts import MCTS


def run_self_play_game(model: Net, move_to_index_map: dict, device: torch.device):
    model.eval()
    game_replay_buffer = []
    state = State()
    state.reset()
    game_data = []
    
    mcts = MCTS(model, device, move_to_index_map, c_puct=1.0)

    while True:
        legal = legal_moves(state)

        result = game_over(state, legal)
        if result != 0:
            value = 0 if result == 0.5 else (1 if Color(state.toMove) == Color.BLACK else -1)
            for s_tensor, policy_vec, turn_color in game_data:
                final_value = value if turn_color == Color.BLACK else -value
                game_replay_buffer.append((s_tensor, policy_vec, final_value))
            break

        if not legal:
            break

        mcts.search_mini_batches(state, batches=10, model=model)
        policy, _ = mcts.get_policy(state, tau=1.0)

        legal_move_indices = [move_to_idx(m, move_to_index_map) for m in legal]
        legal_policy = [policy[idx] for idx in legal_move_indices]
        policy_sum = sum(legal_policy)

        if policy_sum > 0:
            normalized_legal_policy = [p / policy_sum for p in legal_policy]
            move_idx = np.random.choice(len(legal), p=normalized_legal_policy)
        else:
            move_idx = random.randint(0, len(legal) - 1)

        chosen_move = legal[move_idx]
        state_tensor = state_to_tensor(state, device)

        full_policy = np.zeros_like(policy)
        for idx, legal_idx in enumerate(legal_move_indices):
            full_policy[legal_idx] = normalized_legal_policy[idx] if policy_sum > 0 else 1.0 / len(legal)
        game_data.append((state_tensor.cpu().numpy(), full_policy, Color(state.toMove)))

        turn(state, chosen_move)
    return game_replay_buffer


class ChessTrainer:
    def __init__(self, model: Net, index_to_move: dict, device, lr=0.001, batch_size=1000, buffer_size=50000):
        self.device = device
        self.model = model.to(self.device)
        self.index_to_move = index_to_move
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamesPlayed = 0
        self.mseLoss = MSELoss()

    def train(self, epochs=2):
        if len(self.replay_buffer) < self.batch_size:
            print("Replay buffer is smaller than batch size. Skipping training.")
            return

        t_train_start = time.time()

        self.model.train()
        for epoch in range(epochs):
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, policies, values = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).squeeze(1).to(self.device)
            policies = torch.tensor(np.array(policies), dtype=torch.float32).squeeze(1).to(self.device)
            values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            pred_policies, pred_values = self.model(states)
            log_probs = torch.nn.functional.log_softmax(pred_policies, dim=1)
            policy_loss = -torch.sum(policies * log_probs) / policies.size(0)
            value_loss = self.mseLoss(pred_values, values)

            total_loss = policy_loss + value_loss
            total_loss.backward()
            self.optimizer.step()

        print_timing("Training phase", time.time() - t_train_start)
        print(f"→ Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
        self.model.eval()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Net(in_channels=18, num_moves=1972).to(device)

    model_path = "AlphaGoZero.pt"
    if os.path.exists(model_path):
        print("Loading existing model checkpoint...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model Loaded Successfully")
        except Exception as e:
            print(f"Could not load model: {e}. Starting with a new model.")
    else:
        print("No checkpoint found, starting with a new model.")

    model.eval()

    index_to_move_path = "index_to_move.txt"
    index_to_move_map, move_to_index_map = load_index_to_move(index_to_move_path)
    print("Index to Move Loaded")

    trainer = ChessTrainer(model, move_to_index_map, device)

    GAMES_PER_TRAINING_CYCLE = 1750
    TRAINING_DURATION_SECONDS =  72000 # 20 hours

    start_time = time.time()
    print(f"Starting training loop for {TRAINING_DURATION_SECONDS} seconds...")

    while time.time() - start_time < TRAINING_DURATION_SECONDS:
        cycle_start_time = time.time()

        new_data_count = 0
        for i in range(GAMES_PER_TRAINING_CYCLE):
            game_data = run_self_play_game(model, move_to_index_map, device)
            trainer.replay_buffer.extend(game_data)
            new_data_count += len(game_data)
            trainer.gamesPlayed += 1

        cycle_duration = time.time() - cycle_start_time
        print_timing(f"Self-play cycle ({GAMES_PER_TRAINING_CYCLE} games)", cycle_duration)
        print(f"Generated {new_data_count} new training samples.")
        print(f"Total Games Played: {trainer.gamesPlayed}, Buffer Size: {len(trainer.replay_buffer)}")

        if len(trainer.replay_buffer) >= trainer.batch_size:
            print("Starting training phase...")
            trainer.train()
        else:
            print("Not enough data to train yet. Continuing self-play.")

        if trainer.gamesPlayed > 0 and trainer.gamesPlayed % 50 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint at {trainer.gamesPlayed} games played.")

        print("-" * 50)

    print(f"Finished training. Total Games Played: {trainer.gamesPlayed}")
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")
