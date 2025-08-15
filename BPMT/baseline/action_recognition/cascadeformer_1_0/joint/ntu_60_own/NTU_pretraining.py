import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import copy
from transformers import get_cosine_schedule_with_warmup
import math
from math import ceil

POSITIONAL_UPPER_BOUND = 64
BONE_PAIRS = [
    # Spine
    (0, 1), (1, 2), (2, 3), (2, 20),                        

    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7),(7, 21), (7, 22),

    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11),(11, 23), (11, 24),

    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),

    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19)
]

class BiomechanicsReparameterization(nn.Module):
    def __init__(self, bone_pairs, d_model):
        super().__init__()
        self.bone_pairs = bone_pairs
        self.num_bones = len(bone_pairs)
        self.d_model = d_model
        # feature vector: sin(theta), cos(theta), sin(phi), cos(phi), log(length)
        # theta is (90 - altitude), phi is longitude, length is log-normalized
        self.input_dim = 5 * self.num_bones
        self.proj = nn.Linear(self.input_dim, d_model)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints: Tensor of shape (B, T, J, D), joint positions (e.g., 25 joints, 3D)
        Returns:
            Tensor of shape (B, T, d_model) features for transformer
        """
        B, T, J, D = joints.shape

        thetas, phis, lengths = [], [], []

        for parent, child in self.bone_pairs:
            # bone vector from parent joint to child joint
            v = joints[..., child, :] - joints[..., parent, :]  # (B, T, 3)
            length = torch.norm(v, dim=-1) + 1e-6
            u = v / length.unsqueeze(-1)  # normalize it

            x, y, z = u[..., 0], u[..., 1], u[..., 2]
            theta = torch.acos(z.clamp(-1.0, 1.0))
            phi = torch.atan2(y, x) % (2 * math.pi)

            thetas.append(theta)
            phis.append(phi)
            lengths.append(torch.log(length))


        theta = torch.stack(thetas, dim=-1)
        phi = torch.stack(phis, dim=-1)
        length = torch.stack(lengths, dim=-1)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        features = torch.stack([sin_theta, cos_theta, sin_phi, cos_phi, length], dim=-1)
        features = features.view(B, T, -1)  # Flatten to (B, T, 5K)
        return self.proj(features)  # (B, T, d_model)


class BaseT1(nn.Module):
    """
        A simple baseline transformer model for reconstructing masked keypoints.
        The model consists of:
        - Keypoint embedding layer
        - Positional embedding layer
        - Transformer encoder
        - Reconstruction head
        The model is designed to take in sequences of 2D keypoints and reconstruct the masked frames.
    """
    
    def __init__(self, num_joints: int, three_d: bool, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super(BaseT1, self).__init__()
        self.num_joints = num_joints
        self.input_dim = 3 if three_d else 2
        self.d_model = d_model

        # Final linear projection to Transformer dimension
        self.joint_embedding = nn.Linear(num_joints * self.input_dim, d_model)
        #self.joint_embedding = BiomechanicsReparameterization(BONE_PAIRS, d_model)

        # Learnable positional encoding over time
        self.pos_embedding = nn.Parameter(torch.zeros(1, POSITIONAL_UPPER_BOUND, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction head to go from d_model → J*D
        self.reconstruction_head = nn.Linear(d_model, num_joints * self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D)
        Returns:
            output: (B, T, J, D)
        """
        B, T, J, D = x.shape
        assert D == self.input_dim
        assert T <= POSITIONAL_UPPER_BOUND, "T exceeds positional embedding size"

        # JointConv over (B*T, D, J)
        # x = x.reshape(B * T, J, D).permute(0, 2, 1)     # → (B*T, D, J)
        # x = self.joint_conv(x)                          # → (B*T, D, J)
        x = x.reshape(B, T, J * D)      # → (B, T, J*D)

        # frame embedding and positional encoding
        x = self.joint_embedding(x)               # → (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]            # → (B, T, d_model)

        encoded = self.transformer_encoder(x)           # → (B, T, d_model)

        # Reconstruct and reshape
        decoded = self.reconstruction_head(encoded)     # → (B, T, J*D)
        return decoded.view(B, T, J, D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes input into latent representations without reconstruction.
        Args:
            x: (B, T, J, D)
        Returns:
            latent: (B, T, d_model)
        """
        B, T, J, D = x.shape
        assert D == self.input_dim
        assert T <= POSITIONAL_UPPER_BOUND

        # x = x.reshape(B * T, J, D).permute(0, 2, 1)     # → (B*T, D, J)
        # x = self.joint_conv(x)                          # → (B*T, D, J)
        x = x.reshape(B, T, J * D)                      # → (B, T, J*D)
        x = self.joint_embedding(x)                     # → (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]            # → (B, T, d_model)
        return self.transformer_encoder(x)              # → (B, T, d_model)
    
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

    total_steps = num_epochs * ceil(len(train_dataset) / batch_size)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    best_model_state_dict = None
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for i, (sequences, _, _) in enumerate(train_loader):

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = sequences.shape
            sequences = sequences.permute(0, 2, 3, 1, 4)

            # Select most active person (M=1)
            motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
            main_person_idx = motion.argmax(dim=-1)       # (B,)

            indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
            sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
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
            scheduler.step()

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, _, _ in val_loader:
                # Preprocessing sequences from CTR-GCN-style input
                B, C, T, V, M = sequences.shape
                sequences = sequences.permute(0, 2, 3, 1, 4)

                # Select most active person (M=1)
                motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
                main_person_idx = motion.argmax(dim=-1)       # (B,)

                indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
                sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
                sequences = sequences.float().to(device)  # (B, T, J, D)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            tqdm.write(f"[Epoch {epoch+1}] New best validation loss: {val_loss:.4f}")

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return model


def train_T1_both(masking_strategy, train_dataset, val_dataset, model: BaseT1, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_steps = num_epochs * ceil(len(train_dataset) / batch_size)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    best_model_state_dict = None
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        for i, (sequences, _, _) in enumerate(train_loader):

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = sequences.shape
            sequences = sequences.permute(0, 2, 3, 1, 4)

            # Step 1: Permute to (B, M, V, C, T)
            sequences = sequences.permute(0, 4, 3, 1, 2)  # (B, M, V, C, T)

            # Step 2: Flatten batch and person
            sequences = sequences.reshape(B * M, C, T, V).permute(0, 2, 3, 1)  # (B*M, C, T, V) → (B*M, T, V, C)
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
            scheduler.step()

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, _, _ in val_loader:
                # Preprocessing sequences from CTR-GCN-style input
                B, C, T, V, M = sequences.shape
                sequences = sequences.permute(0, 2, 3, 1, 4)

                # Step 1: Permute to (B, M, V, C, T)
                sequences = sequences.permute(0, 4, 3, 1, 2)  # (B, M, V, C, T)

                # Step 2: Flatten batch and person
                sequences = sequences.reshape(B * M, C, T, V).permute(0, 2, 3, 1)  # (B*M, C, T, V) → (B*M, T, V, C)
                sequences = sequences.float().to(device)  # (B, T, J, D)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            tqdm.write(f"[Epoch {epoch+1}] New best validation loss: {val_loss:.4f}")

        tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    return model