import torch
import random
import numpy as np
from collections import deque
import time
import os
import torch.nn.functional as F

# Assume these are in your project directory
from net import Net, state_to_tensor
from game_py import State, legal_moves, game_over, turn, Color
from game_py import load_move_map, move_to_idx, idx_to_move
from mse_loss import MSELoss


class ChessTrainer:
    def __init__(self, model: Net, device, lr=0.001, training_sample_size=50000, mini_batch_size=2048, buffer_size=600000):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.training_sample_size = training_sample_size
        self.mini_batch_size = mini_batch_size
        
        self.gamesPlayed = 0
        self.mseLoss = MSELoss()

    def train(self, epochs=3):
        """
        Trains the model using mini-batches.
        """
        if len(self.replay_buffer) < self.training_sample_size:
            print(f"Replay buffer needs {self.training_sample_size} samples, but has {len(self.replay_buffer)}. Skipping training.")
            return

        self.model.train()

        training_samples = random.sample(self.replay_buffer, self.training_sample_size)
        
        print(f"Starting training with {self.training_sample_size} samples, using mini-batch size of {self.mini_batch_size}.")

        for epoch in range(epochs):
            random.shuffle(training_samples)  # Shuffle samples for each epoch
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_mini_batches = 0

            # --- THE CORE CHANGE: Loop over mini-batches ---
            for i in range(0, len(training_samples), self.mini_batch_size):
                mini_batch = training_samples[i:i + self.mini_batch_size]
                if not mini_batch:
                    continue

                states, policies, values = zip(*mini_batch)

                states = torch.stack(states).to(device=self.device, dtype=torch.half).squeeze(1)
                policies = torch.tensor(np.stack(policies), dtype=torch.float32).to(self.device)
                values = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()
                pred_policies, pred_values = self.model(states)
                
                log_probs = torch.nn.functional.log_softmax(pred_policies.float(), dim=1)
                policy_loss = -torch.sum(policies * log_probs) / policies.size(0)
                value_loss = self.mseLoss(pred_values.float(), values)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_mini_batches += 1

            avg_policy_loss = total_policy_loss / num_mini_batches if num_mini_batches > 0 else 0
            avg_value_loss = total_value_loss / num_mini_batches if num_mini_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs} completed. Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")

        print("Training instance completed.")
        self.model.eval()


def run_single_game(model: Net, device: torch.device):
    """
    Simulates a single game of chess, one move at a time.
    """
    # 1. Initialization
    state = State()
    state.reset()
    game_data_buffer = []
    final_replay_data = []
    model.eval()

    while True:
        # 2. Check for game over
        legal = legal_moves(state)
        result = game_over(state, legal)

        # If the game is over (win/loss/draw) or there are no legal moves
        if result != 0 or not legal:
            white_score = 0.0
            if result != 0.5:  # Not a draw
                winner = Color(1 - state.toMove)
                white_score = 1.0 if winner == Color.WHITE else -1.0

            # Backpropagate the final result
            for s_tensor, policy_vec, turn_color in game_data_buffer:
                final_value = white_score if turn_color == Color.WHITE else -white_score
                final_replay_data.append((s_tensor, policy_vec, final_value))
            break  # Exit the game loop


        state_tensor = state_to_tensor(state, device)

        with torch.no_grad():
            # CORRECTED: Pass the 4D tensor directly to the model.
            policy_logits_batch, _ = model(state_tensor)
        
        policy_logits = policy_logits_batch.squeeze(0)  # Shape becomes [num_moves]

        # 4. Select a move from the legal options
        legal_move_indices = []

    
        for m in legal:
            try:
                legal_move_indices.append(move_to_idx(m))
            except RuntimeError:
                continue

        legal_move_indices = [move_to_idx(m) for m in legal]
        legal_indices_gpu = torch.tensor(legal_move_indices, device=device, dtype=torch.long)

        legal_logits = torch.gather(policy_logits, 0, legal_indices_gpu)
        legal_policy = F.softmax(legal_logits, dim=0)

        if legal_policy.sum() > 0:
            chosen_idx_in_legal = torch.multinomial(legal_policy, 1).item()
        else:
            chosen_idx_in_legal = random.randint(0, len(legal) - 1)

        move = legal[chosen_idx_in_legal]

        # 5. Store data and apply the move
        full_policy_vector = torch.zeros_like(policy_logits)
        full_policy_vector[legal_move_indices] = legal_policy
        
        state_tensor_cpu = state_tensor.cpu()
        current_turn_color = Color(state.toMove)
        game_data_buffer.append((state_tensor_cpu, full_policy_vector.cpu().numpy(), current_turn_color))

        turn(state, move)

    return final_replay_data


def run_parallel_games(model: Net, device: torch.device, num_games: int):
    # 1. Initialization
    states = [State() for _ in range(num_games)]
    for s in states:
        s.reset()

    game_data_buffers = [[] for _ in range(num_games)]
    final_replay_data = []
    active_game_indices = list(range(num_games))
    model.eval()

    while active_game_indices:
        # 2. Collect states and check for game over
        batch_states_for_nn = []
        batch_map = {}
        finished_games_info = []
        next_active_indices = []

        for game_idx in active_game_indices:
            state = states[game_idx]
            legal = legal_moves(state)
            result = game_over(state, legal)

            # FIX: Check for no legal moves (stalemate/checkmate) to end the game.
            if result != 0 or not legal:
                finished_games_info.append((game_idx, result, state))
            else:
                next_active_indices.append(game_idx)
                batch_map[len(batch_states_for_nn)] = (game_idx, legal)
                batch_states_for_nn.append(state_to_tensor(state, device).half())

        # 3. Process finished games
        for game_idx, result, final_state in finished_games_info:
            white_score = 0.0
            if result != 0.5:
                # The winner is the player who just moved, so the turn is opposite the current state
                winner = Color(1 - final_state.toMove)
                white_score = 1.0 if winner == Color.WHITE else -1.0
            
            for s_tensor, policy_vec, turn_color in game_data_buffers[game_idx]:
                final_value = white_score if turn_color == Color.WHITE else -white_score
                final_replay_data.append((s_tensor, policy_vec, final_value))

        active_game_indices = next_active_indices
        if not active_game_indices:
            break

        # 4. Batched Neural Network Inference on the GPU
        state_batch_tensor = torch.stack(batch_states_for_nn)
        state_batch_tensor = state_batch_tensor.squeeze(1)
        
        with torch.no_grad():
            policy_logits_batch, _ = model(state_batch_tensor)

        # 5. Process NN output and play moves (CPU)
        for i in range(policy_logits_batch.size(0)):
            game_idx, legal_moves_for_game = batch_map[i]
            policy_logits = policy_logits_batch[i]
            
            legal_move_indices = []
            for m in legal_moves_for_game:
                try:
                    legal_move_indices.append(move_to_idx(m))
                except RuntimeError:
                    continue
            

            if not legal_move_indices:
                continue
            
            legal_indices_gpu = torch.tensor(legal_move_indices, device=device, dtype=torch.long)
            
            legal_logits = torch.gather(policy_logits, 0, legal_indices_gpu)
            legal_policy = F.softmax(legal_logits, dim=0)
            
            if legal_policy.sum() > 0:
                chosen_idx_in_legal = torch.multinomial(legal_policy, 1).item()
            else:
                chosen_idx_in_legal = random.randint(0, len(legal_move_indices) - 1)
            
            # FIX 3: Robust way to select the move.
            chosen_move_idx = legal_move_indices[chosen_idx_in_legal]
            move = idx_to_move(chosen_move_idx)
            
            full_policy_vector = torch.zeros_like(policy_logits)
            full_policy_vector[legal_move_indices] = legal_policy
            
            state_tensor_cpu = state_batch_tensor[i].cpu().unsqueeze(0)
            current_turn_color = Color(states[game_idx].toMove)
            game_data_buffers[game_idx].append((state_tensor_cpu, full_policy_vector.cpu().numpy(), current_turn_color))
            
            turn(states[game_idx], move)
    
    return final_replay_data


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Net(
        in_channels=18,
        num_moves=1972,
        hidden_channels=256,
        num_res_blocks=8,
        value_fc_dims=(256, 128)
    ).to(device)

    model.half()
    model_path = "PureCNN.pt"
    if os.path.exists(model_path):
        print("Loading existing model checkpoint...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model Loaded Successfully")
        except Exception as e:
            print(f"Could not load model: {e}. Starting with a new model.")
    else:
        print("No checkpoint found, starting with a new model.")

    model = torch.jit.script(model)
    model.eval()

    load_move_map("index_to_move.txt")
    print("Index to Move Loaded")

    trainer = ChessTrainer(model, device)
    
    NUM_PARALLEL_GAMES = 1500
    TRAINING_DURATION_SECONDS = 30000 

    print(f"Training for Hours: {int(TRAINING_DURATION_SECONDS / 3600)} / Minutes: {int(TRAINING_DURATION_SECONDS / 60)}")

    print(f"Starting training loop. Playing {NUM_PARALLEL_GAMES} games in parallel per cycle.")
    start_time = time.time()
    
    cycle_num = 0
    while time.time() - start_time < TRAINING_DURATION_SECONDS:
        cycle_num += 1
        print(f"\n--- Cycle {cycle_num} ---")
        cycle_start_time = time.time()

        print("Starting self-play phase...")
        game_data = run_parallel_games(model, device, NUM_PARALLEL_GAMES)
        trainer.replay_buffer.extend(game_data)
        trainer.gamesPlayed += NUM_PARALLEL_GAMES
        
        data_gen_duration = time.time() - cycle_start_time
        print(f"Self-play for {NUM_PARALLEL_GAMES} games finished in {data_gen_duration:.2f}s.")
        print(f"Average game time: {data_gen_duration / NUM_PARALLEL_GAMES:.4f}s.")
        print(f"Total Games Played: {trainer.gamesPlayed}, Buffer Size: {len(trainer.replay_buffer)}")

        if len(trainer.replay_buffer) >= trainer.mini_batch_size:
            print("Starting training phase...")
            train_start_time = time.time()
            trainer.train()
            print(f"Training phase finished in {time.time() - train_start_time:.2f}s.")

        torch.save(model.state_dict(), model_path)
        print(f"Saved model checkpoint at {trainer.gamesPlayed} games played.")
        
        elapsed_time = time.time() - start_time
        print(f"Estimated time remaining: {(TRAINING_DURATION_SECONDS - elapsed_time) / 3600:.2f} hours")

    print(f"\nFinished training. Total Games Played: {trainer.gamesPlayed}")
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")


def timed_run_parallel_games(model: Net, device: torch.device, num_games: int):
    """
    Simulates a batch of games in parallel and records performance timings.
    """
    # 1. Initialization
    states = [State() for _ in range(num_games)]
    for s in states:
        s.reset()

    game_data_buffers = [[] for _ in range(num_games)]
    final_replay_data = []
    active_game_indices = list(range(num_games))
    model.eval()

    total_moves_played = 0
    timings = {
        "batch_prep": 0.0,
        "inference": 0.0,
        "post_processing": 0.0,
    }
    run_start_time = time.time()

    while active_game_indices:
        # --- 2. Collect states and check for game over (CPU) ---
        t0 = time.time()
        batch_states_for_nn = []
        batch_map = {}
        finished_games_info = []
        next_active_indices = []

        for game_idx in active_game_indices:
            state = states[game_idx]
            legal = legal_moves(state)
            result = game_over(state, legal)

            if result != 0 or not legal:
                finished_games_info.append((game_idx, result, state))
            else:
                next_active_indices.append(game_idx)
                batch_map[len(batch_states_for_nn)] = (game_idx, legal)
                batch_states_for_nn.append(state_to_tensor(state, device))
        timings["batch_prep"] += time.time() - t0

        # --- 3. Process finished games ---
        for game_idx, result, final_state in finished_games_info:
            white_score = 0.0
            if result != 0.5:
                winner = Color(1 - final_state.toMove)
                white_score = 1.0 if winner == Color.WHITE else -1.0
            for s_tensor, policy_vec, turn_color in game_data_buffers[game_idx]:
                final_value = white_score if turn_color == Color.WHITE else -white_score
                final_replay_data.append((s_tensor, policy_vec, final_value))

        active_game_indices = next_active_indices
        if not active_game_indices:
            break
            
        # --- 4. Batched Neural Network Inference (GPU) ---
        t1 = time.time()
        state_batch_tensor = torch.stack(batch_states_for_nn)
        state_batch_tensor = state_batch_tensor.squeeze(1)

        with torch.no_grad():
            policy_logits_batch, _ = model(state_batch_tensor)
        timings["inference"] += time.time() - t1

        # --- 5. Process NN output and play moves (CPU) ---
        t2 = time.time()
        total_moves_played += policy_logits_batch.size(0)
        for i in range(policy_logits_batch.size(0)):
            game_idx, legal_moves_for_game = batch_map[i]
            policy_logits = policy_logits_batch[i]

            legal_move_indices = []
            for m in legal_moves_for_game:
                try:
                    legal_move_indices.append(move_to_idx(m))
                except RuntimeError:

                    continue

            if not legal_move_indices:
                continue

            legal_indices_gpu = torch.tensor(legal_move_indices, device=device, dtype=torch.long)
            
            legal_logits = torch.gather(policy_logits, 0, legal_indices_gpu)
            legal_policy = F.softmax(legal_logits, dim=0)
            
            if legal_policy.sum() > 0:
                chosen_idx_in_legal = torch.multinomial(legal_policy, 1).item()
            else:
                chosen_idx_in_legal = random.randint(0, len(legal_move_indices) - 1)


            chosen_move_idx = legal_move_indices[chosen_idx_in_legal]
            move_obj = idx_to_move(chosen_move_idx) 
            
            full_policy_vector = torch.zeros_like(policy_logits)
            full_policy_vector[legal_move_indices] = legal_policy
            
            state_tensor_cpu = state_batch_tensor[i].cpu().unsqueeze(0)
            current_turn_color = Color(states[game_idx].toMove)
            game_data_buffers[game_idx].append((state_tensor_cpu, full_policy_vector.cpu().numpy(), current_turn_color))
            
            turn(states[game_idx], move_obj)
        timings["post_processing"] += time.time() - t2
    
    total_run_time = time.time() - run_start_time
    return final_replay_data, timings, total_run_time, total_moves_played
    

def profile_parallel():
    """
    Runs the parallel simulation and prints a detailed performance report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    model = Net(
        in_channels=18,
        num_moves=1972,
        hidden_channels=256,
        num_res_blocks=8,
        value_fc_dims=(256, 128)
    ).to(device)
    model.half()

    model_path = "PureCNN.pt"
    if os.path.exists(model_path):
        print("Loading existing model for profiling...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No model found, profiling with a new, untrained model.")

    model = torch.jit.script(model)
    model.eval()
    
    try:
        load_move_map("index_to_move.txt")
        print("Successfully loaded moves from index_to_move.txt")
    except Exception as e:
        print(f"Failed to load move map: {e}")
        return

    NUM_PROFILING_RUNS = 3
    GAMES_PER_RUN = 1500

    total_timings = {"batch_prep": 0.0, "inference": 0.0, "post_processing": 0.0}
    total_run_time = 0.0
    total_moves = 0
    total_games = 0

    print(f"Starting profiler: {NUM_PROFILING_RUNS} runs of {GAMES_PER_RUN} parallel games each.")

    for i in range(NUM_PROFILING_RUNS):
        print(f"  Running profile run {i+1}/{NUM_PROFILING_RUNS}...")
        _, timings, run_time, moves = timed_run_parallel_games(model, device, GAMES_PER_RUN)
        
        for k in timings:
            total_timings[k] += timings[k]
        
        total_run_time += run_time
        total_moves += moves
        total_games += GAMES_PER_RUN

    print("\n--- Parallel Profiling Report ---")
    avg_run_time = total_run_time / NUM_PROFILING_RUNS
    print(f"Avg time per batch of {GAMES_PER_RUN} games: {avg_run_time:.4f} seconds")
    print(f"Time per game: {avg_run_time / GAMES_PER_RUN:.4f} seconds/game")
    print(f"Avg game time: {total_run_time / total_games:.6f} seconds")
    print(f"Total moves processed: {total_moves}")
    print(f"Moves per second: {total_moves / total_run_time:.2f}")

    print("\n--- Avg Time Per Move ---")
    if total_moves > 0:
        prep_time_ms = (total_timings['batch_prep'] / total_moves) * 1000
        infer_time_ms = (total_timings['inference'] / total_moves) * 1000
        post_time_ms = (total_timings['post_processing'] / total_moves) * 1000
        print(f"CPU Batch Prep       (legal moves, etc.): {prep_time_ms:.4f} ms/move")
        print(f"GPU NN Inference     (model evaluation):  {infer_time_ms:.4f} ms/move")
        print(f"CPU Post-Processing  (sampling, play):    {post_time_ms:.4f} ms/move")
    print("---------------------------------")


def profile_single():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling single-game simulation on device: {device}")

    # Initialize the model
    model = Net(
        in_channels=18, num_moves=1972, hidden_channels=256,
        num_res_blocks=8, value_fc_dims=(256, 128)
    ).to(device)

    model_path = "PureCNN.pt"
    if os.path.exists(model_path):
        print("Loading existing model for profiling...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No model found, profiling with a new, untrained model.")

    model = torch.jit.script(model)
    model.eval()
    
    try:
        load_move_map("index_to_move.txt")
        print("Successfully loaded moves from index_to_move.txt")
    except Exception as e:
        print(f"Failed to load move map: {e}")
        return

    NUM_PROFILING_GAMES = 25
    total_time = 0.0
    total_moves = 0

    print(f"\nStarting profiler: Running {NUM_PROFILING_GAMES} games sequentially.")
    start_time = time.time()
    for i in range(NUM_PROFILING_GAMES):
        print(f"  Running game {i+1}/{NUM_PROFILING_GAMES}...")
        game_data = run_single_game(model, device)
        total_moves += len(game_data)
    total_time = time.time() - start_time

    print("\n--- Single-Game Profiling Report ---")
    print(f"Total time for {NUM_PROFILING_GAMES} games: {total_time:.4f} seconds")
    if NUM_PROFILING_GAMES > 0:
        print(f"Average time per game: {total_time / NUM_PROFILING_GAMES:.4f} seconds")
    if total_time > 0:
        print(f"Moves per second: {total_moves / total_time:.2f}")
    print("------------------------------------")

if __name__ == "__main__":
    run()
