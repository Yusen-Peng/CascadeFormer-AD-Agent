import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from typing import List
from NTU_pretraining import BaseT1

def load_T1(model_path: str, num_joints: int = 13, three_d: bool = False, d_model: int = 128, nhead: int = 4, num_layers: int = 2, freeze: bool = True,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT1:
    """
        loads a BaseT1 model from a checkpoint
    """

    model = BaseT1(num_joints=num_joints, three_d=three_d, d_model=d_model, nhead=nhead, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # optionally freeze the model parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # move model to device and return the model
    return model.to(device)

class SimpleCrossAttention(nn.Module):
    """
    Simple cross-attention block that applies multi-head attention between two feature sequences.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super(SimpleCrossAttention, self).__init__()
        # cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # layer normalization
        self.norm = nn.LayerNorm(d_model)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:        
        # cross attention
        attn_out, _ = self.cross_attention(Q, K, V)

        # apply layer normalization and dropout, and residual connection
        out = self.norm(Q + self.dropout(attn_out))
        
        return out

class CrossAttentionWithFFN(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.2):
        super(CrossAttentionWithFFN, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,  # dropout inside attention weights
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN block (standard in Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        attn_out, _ = self.cross_attention(Q, K, V)
        x = self.norm1(Q + self.dropout1(attn_out))  # residual + norm

        # Feedforward + residual
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.dropout2(ffn_out))  # residual + norm
        return out


class BaseT2(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(BaseT2, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
    
    def encode(self, x):
        """
            Encodes the input sequence using the transformer encoder.
            Returns the encoded features.
        """
        encoded = self.encoder(x)
        return encoded

class SimpleGaitRecognitionHead(nn.Module):
    """
        A simple linear head for gait recognition.
        The model consists of a linear layer that maps the output of the transformer to the number of classes.
    """
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class GaitRecognitionHeadWithDropout(nn.Module):
    """
        A simple linear head for gait recognition.
        The model consists of a linear layer that maps the output of the transformer to the number of classes.
    """
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)  # apply dropout
        return self.fc(x)

class GaitRecognitionHeadMLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    t1: BaseT1,
    gait_head: nn.Module,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_epochs: int = 200,
    lr: float = 1e-5,
    wd: float = 1e-2,
    freezeT1: bool = True,
    t2_dropout: float = 0.1,
    cross_attn_dropout: float = 0.1,
    unfreeze_layers: List[int] = None,
    lr_lower_bound: float = 3e-6,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[BaseT1, nn.Module, nn.Module]:
    
    # freeze T1 parameters
    if freezeT1:
        for param in t1.parameters():
            param.requires_grad = False

        # unfreeze specific layers if specified
        if unfreeze_layers is not None:
            for layer in unfreeze_layers:
                for param in t1.transformer_encoder.layers[layer].parameters():
                    param.requires_grad = True
    else:
        for param in t1.parameters():
            param.requires_grad = True
    t1.to(device)
    gait_head.to(device)

    # intialize T2 transformer and cross-attention
    t2 = BaseT2(d_model, nhead, num_layers, dropout=t2_dropout).to(device)
    #cross_attn = SimpleCrossAttention(d_model, nhead).to(device)
    cross_attn = CrossAttentionWithFFN(
        d_model=d_model,
        nhead=nhead,
        dropout=cross_attn_dropout
    ).to(device)

    # optimizer and loss
    params = list(filter(lambda p: p.requires_grad, t1.parameters())) + \
         list(t2.parameters()) + \
         list(cross_attn.parameters()) + \
         list(gait_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=lr_lower_bound
    )

    # add label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(num_epochs)):
        gait_head.train()
        t2.train()
        cross_attn.train()

        t1_trainable = any(p.requires_grad for p in t1.parameters())
        t1.train(mode=t1_trainable)

        total_loss, correct, total = 0.0, 0, 0
        for i, (skeletons, labels, _) in enumerate(train_loader):
            skeletons, labels = skeletons.to(device), labels.to(device)

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = skeletons.shape
            sequences = skeletons.permute(0, 2, 3, 1, 4)

            # Select most active person (M=1)
            motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
            main_person_idx = motion.argmax(dim=-1)       # (B,)

            indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
            sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
            sequences = sequences.float().to(device)  # (B, T, J, D)

            if t1_trainable:
                x1 = t1.encode(sequences)  # grads will flow
            else:
                with torch.no_grad():
                    x1 = t1.encode(sequences)  # frozen model, no grads

            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)

            # we need to do pooling
            # other than mean pooling, we can do Adaptive Pooling 
            pooled = fused.mean(dim=1)
            logits = gait_head(pooled)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)


        train_acc = correct / total
        avg_loss = total_loss / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        # Validation
        t1.eval()
        gait_head.eval()
        t2.eval()
        cross_attn.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for skeletons, labels, _ in val_loader:
                skeletons, labels = skeletons.to(device), labels.to(device)

                # Preprocessing sequences from CTR-GCN-style input
                B, C, T, V, M = skeletons.shape
                sequences = skeletons.permute(0, 2, 3, 1, 4)

                # Select most active person (M=1)
                motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
                main_person_idx = motion.argmax(dim=-1)       # (B,)

                indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
                sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
                sequences = sequences.float().to(device)  # (B, T, J, D)

                x1 = t1.encode(sequences)  # (B, T, J, D)

                # # add a CLS token
                # B = x1.size(0)
                # cls_token = torch.zeros(B, 1, d_model).to(x1.device)
                # x1 = torch.cat([cls_token, x1], dim=1)  # prepend CLS

                x2 = t2.encode(x1)
                fused = cross_attn(x1, x2, x2)

                # we need to do pooling
                pooled = fused.mean(dim=1)
                # CLS token pooling
                #pooled = fused[:, 0]  # take the [CLS] token
                logits = gait_head(pooled)


                val_loss = criterion(logits, labels)
                val_total_loss += val_loss.item() * labels.size(0)
                
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_avg_loss = val_total_loss / val_total

        val_losses.append(val_avg_loss)
        val_accuracies.append(val_acc)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        #tqdm.write(f"Epoch {epoch+1}/{num_epochs}: LR = {current_lr:.6f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}: LR = {current_lr:.6f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}", flush=True)

    # return T2, cross_attn, and gait_head
    return t2, cross_attn, gait_head



def adjust_learning_rate(
        epoch: int, 
        optimizer: torch.optim.SGD, 
        warm_up_epoch: int = 5, # match official!
        base_lr: float = 0.025, # match official! 
        lr_decay_rate: float = 0.1, # match official! 
        step: list = [110, 120] # match official!
    ):
    """ Custom learning rate warm-up and decay function."""
    if epoch < warm_up_epoch:
        lr = base_lr * (epoch + 1) / warm_up_epoch
    else:
        lr = base_lr * (
                lr_decay_rate ** np.sum(epoch >= np.array(step)))
        
    # adjust the learning rate for the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # return it for sanity check
    return lr


def finetuning_both(
    train_loader: DataLoader,
    val_loader: DataLoader,
    t1: BaseT1,
    gait_head: nn.Module,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_epochs: int = 200,
    lr: float = 1e-5,
    wd: float = 1e-2,
    freezeT1: bool = True,
    t2_dropout: float = 0.1,
    cross_attn_dropout: float = 0.1,
    unfreeze_layers: List[int] = None,
    lr_lower_bound: float = 3e-6,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[BaseT1, nn.Module, nn.Module]:
    
    # freeze T1 parameters
    if freezeT1:
        for param in t1.parameters():
            param.requires_grad = False

        # unfreeze specific layers if specified
        if unfreeze_layers is not None:
            for layer in unfreeze_layers:
                for param in t1.transformer_encoder.layers[layer].parameters():
                    param.requires_grad = True
    else:
        for param in t1.parameters():
            param.requires_grad = True
    t1.to(device)
    gait_head.to(device)

    # intialize T2 transformer and cross-attention
    t2 = BaseT2(d_model, nhead, num_layers, dropout=t2_dropout).to(device)
    #cross_attn = SimpleCrossAttention(d_model, nhead).to(device)
    cross_attn = CrossAttentionWithFFN(
        d_model=d_model,
        nhead=nhead,
        dropout=cross_attn_dropout
    ).to(device)

    # optimizer and loss
    params = list(filter(lambda p: p.requires_grad, t1.parameters())) + \
         list(t2.parameters()) + \
         list(cross_attn.parameters()) + \
         list(gait_head.parameters())

    #optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=0.9,
        nesterov=True, # match!
        weight_decay=wd
    )


    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=num_epochs,
    #     eta_min=lr_lower_bound
    # )

    # add label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(num_epochs)):
        # adjust learning rate at the BEGINNING of each epoch
        lr = adjust_learning_rate(epoch, optimizer, warm_up_epoch=5, base_lr=lr, lr_decay_rate=0.1, step=[110, 120])

        gait_head.train()
        t2.train()
        cross_attn.train()

        t1_trainable = any(p.requires_grad for p in t1.parameters())
        t1.train(mode=t1_trainable)

        total_loss, correct, total = 0.0, 0, 0
        for i, (skeletons, labels, _) in enumerate(train_loader):
            skeletons = skeletons.to(device)
            labels = labels.to(device)
            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = skeletons.shape
            sequences = skeletons.permute(0, 2, 3, 1, 4)

            # Step 1: Permute to (B, M, V, C, T)
            sequences = sequences.permute(0, 4, 3, 1, 2)  # (B, M, V, C, T)

            # Step 2: Flatten batch and person
            sequences = sequences.reshape(B * M, C, T, V).permute(0, 2, 3, 1)  # (B*M, C, T, V) → (B*M, T, V, C)
            sequences = sequences.float().to(device)  # (B, T, J, D)

            if t1_trainable:
                x1 = t1.encode(sequences)  # grads will flow
            else:
                with torch.no_grad():
                    x1 = t1.encode(sequences)  # frozen model, no grads

            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)

            # we need to do pooling
            # other than mean pooling, we can do Adaptive Pooling 
            pooled = fused.mean(dim=1)
            logits = gait_head(pooled)

            if sequences.shape[0] != labels.shape[0]:
                # You flattened the person dimension (B*M), so replicate labels accordingly
                M = sequences.shape[0] // labels.shape[0]
                labels = labels.unsqueeze(1).repeat(1, M).view(-1)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)


        train_acc = correct / total
        avg_loss = total_loss / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        # Validation
        t1.eval()
        gait_head.eval()
        t2.eval()
        cross_attn.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for skeletons, labels, _ in val_loader:
                skeletons = skeletons.to(device)
                labels = labels.to(device)
                # Preprocessing sequences from CTR-GCN-style input
                B, C, T, V, M = skeletons.shape
                sequences = skeletons.permute(0, 2, 3, 1, 4)

                # Step 1: Permute to (B, M, V, C, T)
                sequences = sequences.permute(0, 4, 3, 1, 2)  # (B, M, V, C, T)

                # Step 2: Flatten batch and person
                sequences = sequences.reshape(B * M, C, T, V).permute(0, 2, 3, 1)  # (B*M, C, T, V) → (B*M, T, V, C)
                sequences = sequences.float().to(device)  # (B, T, J, D)

                x1 = t1.encode(sequences)  # (B, T, J, D)

                # # add a CLS token
                # B = x1.size(0)
                # cls_token = torch.zeros(B, 1, d_model).to(x1.device)
                # x1 = torch.cat([cls_token, x1], dim=1)  # prepend CLS

                x2 = t2.encode(x1)
                fused = cross_attn(x1, x2, x2)

                # we need to do pooling
                pooled = fused.mean(dim=1)
                # CLS token pooling
                #pooled = fused[:, 0]  # take the [CLS] token
                logits = gait_head(pooled)
                if sequences.shape[0] != labels.shape[0]:
                    # You flattened the person dimension (B*M), so replicate labels accordingly
                    M = sequences.shape[0] // labels.shape[0]
                    labels = labels.unsqueeze(1).repeat(1, M).view(-1)

                val_loss = criterion(logits, labels)
                val_total_loss += val_loss.item() * labels.size(0)
                
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_avg_loss = val_total_loss / val_total

        val_losses.append(val_avg_loss)
        val_accuracies.append(val_acc)
        #scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        #tqdm.write(f"Epoch {epoch+1}/{num_epochs}: LR = {current_lr:.6f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}: LR = {current_lr:.6f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}", flush=True)

    # return T2, cross_attn, and gait_head
    return t2, cross_attn, gait_head





def load_T2(model_path: str,d_model: int = 128, nhead: int = 4, num_layers: int = 2, t2_dropout: float = 0.1,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BaseT2:
    """
        loads a BaseT2 model from a checkpoint
    """
    model = BaseT2(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=t2_dropout)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for param in model.parameters():
        param.requires_grad = False
        
    # move model to device and return the model
    return model.to(device)

def load_simple_cross_attn(path: str,
                    d_model: int = 128,
                    nhead: int = 4,
                    cross_attn_dropout: float = 0.1,
                    device: str = "cuda") -> SimpleCrossAttention:
    """
        loads a CrossAttention model from a checkpoint
    """
    layer = SimpleCrossAttention(d_model=d_model, nhead=nhead, dropout=cross_attn_dropout)
    layer.load_state_dict(torch.load(path, map_location="cpu"))
    return layer.to(device)

def load_cross_attn_with_ffn(path: str,
                    d_model: int = 128,
                    nhead: int = 4,
                    dropout: float = 0.1,
                    device: str = "cuda") -> CrossAttentionWithFFN:
    """
        loads a CrossAttentionWithFFN model from a checkpoint
    """
    layer = CrossAttentionWithFFN(d_model=d_model, nhead=nhead, dropout=dropout)
    layer.load_state_dict(torch.load(path, map_location="cpu"))
    return layer.to(device)
