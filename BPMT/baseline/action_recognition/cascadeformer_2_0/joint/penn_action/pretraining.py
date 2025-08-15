import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

POSITIONAL_UPPER_BOUND = 1024

class SpatialAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=1
        )

    def forward(self, x):  # (B, T, J, D)
        B, T, J, D = x.shape
        x = x.reshape(B * T, J, D)  # Use reshape instead of view
        x = self.encoder(x)
        return x.reshape(B, T, J, D)  # Use reshape here too for consistency


class TemporalAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=1
        )

    def forward(self, x):  # (B, T, J, D)
        B, T, J, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, D)
        x = self.encoder(x)
        return x.reshape(B, J, T, D).permute(0, 2, 1, 3)


class BaseT1(nn.Module):
    def __init__(self, num_joints: int, three_d: bool, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = 3 if three_d else 2
        self.d_model = d_model

        self.three_d = three_d
        self.num_joints = num_joints

        self.joint_embedding = nn.Linear(self.input_dim, d_model // num_joints)    

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Interleaved spatial and temporal attention
        self.spatial_blocks = nn.ModuleList([
            SpatialAttention(d_model // num_joints, nhead=nhead) for _ in range(num_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalAttention(d_model // num_joints, nhead=nhead) for _ in range(num_layers)
        ])

        # Reconstruction head
        self.reconstruction_head = nn.Linear(d_model, num_joints * self.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        assert D == self.input_dim
        assert T <= POSITIONAL_UPPER_BOUND

        x = self.joint_embedding(x)                   # (B, T, J, d_model // J)
        x = x.reshape(B, T, -1)                       # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]          # ✅ Now shapes match
        x = x.reshape(B, T, J, self.d_model // J)     # Restore for attention: D' = d_model // J

        # Interleaved spatial and temporal attention
        for sa, ta in zip(self.spatial_blocks, self.temporal_blocks):
            x = sa(x)
            x = ta(x)

        # Flatten back to (B, T, d_model)
        x = x.reshape(B, T, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        assert J == self.num_joints and D == self.input_dim

        encoded = self.encode(x)                         # (B, T, d_model)
        decoded = self.reconstruction_head(encoded)      # (B, T, J*D)
        return decoded.view(B, T, J, D)                  # (B, T, J, D)
    
PAD_IDX = 0.0

def mask_random_global_joints(inputs: torch.Tensor, mask_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask joint *coordinates* globally per sample across (T, J),
    and apply mask over all D dimensions of a joint.

    Args:
        inputs: Tensor of shape (B, T, J, D)
        mask_ratio: Fraction of total (T*J) locations to mask

    Returns:
        masked_inputs: Tensor of same shape as inputs with masked values
        mask: Boolean mask of shape (B, T, J) indicating masked joints
    """
    B, T, J, D = inputs.shape
    masked_inputs = inputs.clone()
    mask = torch.zeros(B, T, J, dtype=torch.bool, device=inputs.device)

    total_slots = T * J
    num_to_mask = max(1, int(mask_ratio * total_slots))

    for i in range(B):
        indices = torch.randperm(total_slots, device=inputs.device)[:num_to_mask]
        t_indices = indices // J
        j_indices = indices % J
        mask[i, t_indices, j_indices] = 1

    # Expand to (B, T, J, D)
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, D)
    masked_inputs[expanded_mask] = PAD_IDX
    return masked_inputs, mask

def train_T1(masking_strategy, train_dataset, val_dataset, model: BaseT1, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):
    from penn_utils import collate_fn_batch_padding

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batch_padding)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batch_padding)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for sequences, _ in train_loader:
            sequences = sequences.float().to(device)  # (B, T, J, D)
            B, T, J, D = sequences.shape

            if masking_strategy == "global_joint":
                masked_inputs, mask = mask_random_global_joints(sequences, mask_ratio=mask_ratio)
                # Expand to (B, T, J, D)
                mask_broadcasted = mask.unsqueeze(-1).expand(B, T, J, D)
            else:
                raise ValueError(f"Unknown masking strategy: {masking_strategy}")

            recons = model(masked_inputs)  # output: (B, T, J, D)
            loss_matrix = criterion(recons, sequences)  # shape: (B, T, J, D)

            masked_loss = loss_matrix * mask_broadcasted
            num_masked = mask_broadcasted.sum()
            loss = masked_loss.sum() / (num_masked + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.float().to(device)
                B, T, J, D = sequences.shape

                masked_inputs, mask = mask_random_global_joints(sequences, mask_ratio=mask_ratio)
                mask_broadcasted = mask.unsqueeze(-1).expand(B, T, J, D)

                recons = model(masked_inputs)
                loss_matrix = criterion(recons, sequences)

                masked_loss = loss_matrix * mask_broadcasted
                num_masked = mask_broadcasted.sum()
                loss = masked_loss.sum() / (num_masked + 1e-8)
                val_loss += loss.item() * sequences.size(0)

        val_loss /= len(val_dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model