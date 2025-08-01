import torch
import torch.nn as nn
import torch.nn.functional as F
from game_py import State, Color, Square
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 =  nn.BatchNorm2d(channels)
        self.relu_2d = nn.ReLU()
        self.relu_4d = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu_4d(x)
        x = self.bn2(self.conv2(x))
        return self.relu_4d(x + residual)


class Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_moves: int,
        hidden_channels: int = 256,
        num_res_blocks: int = 6,
        value_fc_dims: tuple = (128, 64)
    ):
        super().__init__()
        self.relu_2d = nn.ReLU()
        self.relu_4d = nn.ReLU()

        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # Dynamically create residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        fc1_dim, fc2_dim = value_fc_dims
        self.value_fc1 = nn.Linear(8 * 8, fc1_dim)
        self.value_fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.value_out = nn.Linear(fc2_dim, 1)

    def forward(self, x):
        x = self.relu_4d(self.input_conv(x))
        x = self.res_blocks(x)

        # Policy
        p = self.relu_4d(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = self.relu_4d(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = self.relu_2d(self.value_fc1(v))
        v = self.relu_2d(self.value_fc2(v))
        value = torch.tanh(self.value_out(v))

        return policy_logits, value


# Convert State object to tensor
def state_to_tensor(state, device=None):
    piece_bitboards = np.array(state.boards[:12], dtype=np.uint64)
    s = np.unpackbits(piece_bitboards.view(np.uint8)).reshape(-1, 8, 8).astype(np.float32)

    tensor = torch.zeros((18, 8, 8), dtype=torch.float32, device=device)
    tensor[:12] = torch.from_numpy(s)

    tensor[12].fill_(1.0 if state.toMove == 0 else 0.0) 
    if state.castling & 0b0001: tensor[13].fill_(1.0) # White K-side
    if state.castling & 0b0010: tensor[14].fill_(1.0) # White Q-side
    if state.castling & 0b0100: tensor[15].fill_(1.0) # Black K-side
    if state.castling & 0b1000: tensor[16].fill_(1.0) # Black Q-side

    if state.en_passant_sq != Square.NO_SQUARE:
        file_index = int(state.en_passant_sq) % 8
        tensor[17, :, file_index] = 1.0

    return tensor.unsqueeze(0)
    
if __name__ == "__main__":
    in_channels = 18
    num_moves = 1972
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(
        in_channels=18,
        num_moves=1972,
        hidden_channels=256,     
        num_res_blocks=8,        
        value_fc_dims=(256, 128) 
    ).to(device)

    torch.save(model.state_dict(), "PureCNN.pt")
    print("Saved model with randomly initialized weights.")
