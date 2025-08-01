import torch
import random
import math
import numpy as np
from collections import defaultdict
import copy

from game_py import State, Color, legal_moves, turn, game_over, Square, PieceType
from maps import move_to_idx

from net import Net, state_to_tensor
from softmax import Softmax

class MCTS:
    def __init__(self, model: Net, device, move_to_idx_map, c_puct: float = 1.0):
        self.model = model
        self.model.eval()
        self.c_puct = c_puct
        self.rng = random.Random()
        self.device = device

        self.num_moves = 1972  # Global fixed action space
        self.visit_count = defaultdict(lambda: [0] * self.num_moves)
        self.value = defaultdict(lambda: [0.0] * self.num_moves)
        self.avg_value = defaultdict(lambda: [0.0] * self.num_moves)
        self.probs = defaultdict(lambda: [0.0] * self.num_moves)

        self.move_to_idx_map = move_to_idx_map

        self.softmax = Softmax()

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.avg_value.clear()
        self.probs.clear()

    def board_key(self, state):
        to_move, boards_tuple, castling, en_passant = state.hash()
        key = f"{to_move}" + "_".join(map(str, boards_tuple))
        return key

    def find_leaf(self, root_state):
        visited_states = []
        actions_taken_idx = []
        player = Color(root_state.toMove)
        reward = 0.0

        while True:
            key = self.board_key(root_state)

            legal = legal_moves(root_state)

            if key not in self.visit_count:
                size = len(legal)
                self.visit_count[key] = [0] * size
                self.value[key] = [0.0] * size
                self.avg_value[key] = [0.0] * size
                self.probs[key] = [1.0 / size] * size if size > 0 else []
                break

            g_o = game_over(root_state, legal)
            if g_o != 0:
                
                if g_o == 0.5:
                    reward = 0.5
                elif (player == Color.WHITE and g_o == 1.0) or (player == Color.BLACK and g_o == -1.0):
                    reward = 1.0
                else:
                    reward = -1.0
                break

            counts = self.visit_count[key]
            Q = self.avg_value[key]
            P = self.probs[key]

            sqrt_total = math.sqrt(sum(counts) + 1.0)

            best_idx_legal = -1
            best_score = float("-inf")
            for i in range(len(legal)):
                u = Q[i] + self.c_puct * P[i] * sqrt_total / (1 + counts[i])
                if u > best_score:
                    best_score = u
                    best_idx_legal = i


            best_move = legal[best_idx_legal]

            best_idx = move_to_idx(best_move, self.move_to_idx_map)

            visited_states.append(key)
            actions_taken_idx.append(best_idx)
            

            turn(root_state, best_move)
            

        return reward, root_state, visited_states, actions_taken_idx

    def search_mini_batches(self, root, batches, model):
        states_to_eval = []
        leaf_infos = []  # to store (leaf_state, visited, moves, leaf_key)

        for _ in range(batches):
            copy_state = State()
            copy_state.toMove = root.toMove
            copy_state.boards = root.boards
            copy_state.castling = root.castling
            copy_state.en_passant_sq = root.en_passant_sq
            copy_state.fifty_move = root.fifty_move
            copy_state.repetition_table = copy.deepcopy(root.repetition_table)

            reward, leaf_state, visited, moves = self.find_leaf(copy_state)
            leaf_key = self.board_key(leaf_state)

            states_to_eval.append(state_to_tensor(leaf_state, self.device))  # [1, 18, 8, 8]
            leaf_infos.append((leaf_state, visited, moves, leaf_key))

        # Batch inference
        input_tensor = torch.cat(states_to_eval, dim=0)  # [B, 18, 8, 8]
        with torch.no_grad():
            policy_logits_batch, value_batch = model(input_tensor)

        for i, (leaf_state, visited, moves, leaf_key) in enumerate(leaf_infos):
            policy_logits = policy_logits_batch[i]  # [1972]
            value_tensor = value_batch[i]  # scalar

            legal = legal_moves(leaf_state)
            legal_indices = [move_to_idx(m, self.move_to_idx_map) for m in legal]
            mask = torch.zeros_like(policy_logits)
            mask[legal_indices] = 1

            masked_logits = policy_logits.masked_fill(mask == 0, float('-inf'))
            policy_probs = self.softmax(masked_logits)

            move_probs = [0.0] * self.num_moves
            for idx in legal_indices:
                move_probs[idx] = policy_probs[idx].item()

            self.probs[leaf_key] = move_probs
            self.value[leaf_key] = [value_tensor.item()] * self.num_moves
            self.visit_count[leaf_key] = [0] * self.num_moves
            self.avg_value[leaf_key] = [0.0] * self.num_moves

            v = value_tensor.item()
            for key, a in zip(visited, moves):
                self.visit_count[key][a] += 1
                n = self.visit_count[key][a]
                q = self.avg_value[key][a]
                self.avg_value[key][a] = (q * (n - 1) + v) / n




    def get_policy(self, state, tau=1.0):
        key = self.board_key(state)
        counts = self.visit_count[key]
        policy = np.power(counts, 1.0 / tau)
        total = np.sum(policy) + 1e-8
        policy /= total
        return policy, self.avg_value[key]

