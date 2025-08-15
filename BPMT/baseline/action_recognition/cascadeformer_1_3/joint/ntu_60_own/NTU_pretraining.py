import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from typing import Tuple

POSITIONAL_UPPER_BOUND = 64

NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (3, 4),         # right arm
    (20, 5), (5, 6), (6, 7), (7, 8),                  # left arm
    (0, 9), (9, 10), (10, 11), (11, 12),              # right leg
    (0, 13), (13, 14), (14, 15), (15, 16),            # left leg
    (0, 17), (17, 18), (18, 19), (19, 21), (21, 22),  # spine + head
    (19, 23), (12, 24)                                # hands/feet (optional)
]

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.register_buffer('A', A) # (J, J) adjacency matrix
        self.linear = nn.Linear(in_features, out_features)
        self.residual = (in_features == out_features)

        # Initialization
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        if self.linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        Ax = torch.einsum('ij,btjd->btid', self.A, x)
        out = self.linear(Ax)
        if self.residual:
            out = out + x
        return out

class SpatialGCN(nn.Module):
    def __init__(self, num_joints, input_dim, hidden_dim, output_dim):
        super().__init__()
        A = self.build_adjacency(num_joints, NTU_BONES) # (J, J)
        self.gcn1 = GCNLayer(input_dim, hidden_dim, A)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, A)
        self.gcn3 = GCNLayer(hidden_dim, output_dim, A)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):  # x: (B, T, J, D)
        x = self.relu(self.dropout(self.gcn1(x)))
        x = self.relu(self.dropout(self.gcn2(x)))
        x = self.gcn3(x)
        return x

    @staticmethod
    def build_adjacency(num_joints, bones, self_loops=True, normalize=True):
        A = torch.zeros((num_joints, num_joints), dtype=torch.float32)
        for i, j in bones:
            A[i, j] = A[j, i] = 1
        if self_loops:
            A += torch.eye(num_joints)
        if normalize:
            D_inv_sqrt = torch.diag(1.0 / A.sum(dim=1).clamp(min=1e-5).sqrt())
            A = D_inv_sqrt @ A @ D_inv_sqrt
        return A

    
class BaseT1(nn.Module):
    def __init__(self, num_joints, three_d, d_model=400, nhead=4, num_layers=2):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = 3 if three_d else 2
        self.d_model = d_model
        self.d_model_per_joint = d_model // num_joints
        assert d_model % num_joints == 0, "d_model must be divisible by num_joints"

        # Spatial GCN: D -> d_model / J
        self.spatial_gcn = SpatialGCN(
            num_joints, 
            input_dim=self.input_dim, # D
            hidden_dim=self.d_model_per_joint,       # d_model / J
            output_dim=self.d_model_per_joint        # d_model / J
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=0.1, # dropout added!
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(d_model, num_joints * self.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        x = self.spatial_gcn(x)  # (B, T, J, d_model / J)
        x = x.view(B, T, -1)  # (B, T, J * d_model / J) -> (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]
        return self.transformer_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, D = x.shape
        encoded = self.encode(x)  # (B, T, d_model)
        decoded = self.reconstruction_head(encoded)
        return decoded.view(B, T, J, D)


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


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
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
