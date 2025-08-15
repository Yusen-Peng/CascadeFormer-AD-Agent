
import numpy as np
import torch
import argparse
from torch import nn
from penn_utils import set_seed
from NTU_utils import NUM_JOINTS_NTU
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead
from torch_lr_finder import LRFinder
from typing import List
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ActionRecognitionDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        self.seqs   = sequences
        self.labels = labels
        self.num_classes = len(set(labels))

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.float32)
        label_raw = self.labels[idx]

        # Convert one-hot or list to scalar index
        if isinstance(label_raw, (list, np.ndarray)) and not np.isscalar(label_raw):
            label = int(np.argmax(label_raw))
        else:
            label = int(label_raw)

        label = torch.tensor(label, dtype=torch.long)
        return seq, label


class CascadeWrapper(nn.Module):
    def __init__(self, T1, T2, cross_attn, head):
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.cross_attn = cross_attn
        self.head = head

    def forward(self, x):
        feat1 = self.T1.encode(x)               
        feat2 = self.T2.encode(feat1)               
        fused = self.cross_attn(feat1, feat2, feat2)
        pooled = fused.mean(dim=1)
        out = self.head(pooled)
        return out


def load_cached_data(path="ntu_cache_train_sub.npz"):
    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    labels = list(data["labels"])
    
    # Convert from one-hot to index labels if needed
    labels = [np.argmax(label) if isinstance(label, (np.ndarray, list)) and not np.isscalar(label) else label for label in labels]

    return sequences, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Inference")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    device = args.device

    # Set the device

    hidden_size = 512 # 256, 512, 768, 1024
    n_heads = 8
    num_layers = 8    # 4, 8, 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting NTU dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    test_seq, test_lbl = load_cached_data('ntu_cache_test_sub_64_10.npz')
    print("[DEBUG] Label example:", test_lbl[0], type(test_lbl[0]), np.array(test_lbl[0]).shape)
    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)
    # get the number of classes
    num_classes = len(set(test_lbl))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1("action_checkpoints/NTU_GCN/NTU_pretrained.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
    else:
        t1 = load_T1("action_checkpoints/NTU_GCN/NTU_finetuned_T1.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    t2 = load_T2("action_checkpoints/NTU_GCN/NTU_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn("action_checkpoints/NTU_GCN/NTU_finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("action_checkpoints/NTU_GCN/NTU_finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting LR search...")
    print("=" * 50)

    # Initialize LR Finder
    model = CascadeWrapper(t1, t2, cross_attn, gait_head).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # safe start
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(test_loader, end_lr=1e-1, num_iter=100, step_mode="exp")

    # Access data
    lrs = lr_finder.history['lr']
    losses = lr_finder.history['loss']

    # Compute the steepest descent (same logic as LRFinder's default suggestion)
    grads = np.gradient(losses, np.log(lrs))
    min_grad_idx = np.argmin(grads)
    suggested_lr = lrs[min_grad_idx]

    # Plot as usual
    lr_finder.plot()
    plt.scatter(suggested_lr, losses[min_grad_idx], s=20, c='green', label=f"Suggested LR: {suggested_lr:.2e}")
    plt.legend()
    plt.title("Learning Rate Finder")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")

    # Save
    plt.savefig("lr_finder_plot_annotated.png")

if __name__ == "__main__":
    main()