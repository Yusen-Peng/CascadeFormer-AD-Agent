import numpy as np
import torch
from sklearn.metrics import accuracy_score
import argparse
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from NTU_feeder import Feeder
from penn_utils import set_seed
from NTU_utils import NUM_JOINTS_NTU
from NTU_pretraining import BaseT1
from finetuning import load_T1, load_T2, BaseT2, load_cross_attn_with_ffn, GaitRecognitionHeadMLP

def load_cached_data(path="ntu_cache_train_sub.npz"):
    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    labels = list(data["labels"])
    return sequences, labels

def evaluate(
    data_loader: DataLoader,
    t1: BaseT1,
    t2: BaseT2,
    cross_attn: nn.Module,
    gait_head: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Performs inference and computes accuracy over the given dataset.

    Args:
        data_loader: DataLoader for evaluation
        t1: pretrained (frozen or finetuned) T1 transformer
        t2: trained T2 transformer
        cross_attn: trained CrossAttention module
        gait_head: trained classification head
        device: device to run inference on
        pooling: pooling strategy - 'mean' or 'attention'
        attention_pool: optional attention pooling module (required if pooling == 'attention')

    Returns:
        accuracy: float
        all_preds: tensor of predictions
        all_labels: tensor of ground-truth labels
    """
    t1.eval()
    t2.eval()
    cross_attn.eval()
    gait_head.eval()
   

    all_preds, all_labels = [], []

    with torch.no_grad():
        for skeletons, labels, _ in data_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = skeletons.shape
            sequences = skeletons.permute(0, 2, 3, 1, 4)

            # Select most active person (M=1)
            motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
            main_person_idx = motion.argmax(dim=-1)       # (B,)

            indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
            sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
            skeletons = sequences.float().to(device)  # (B, T, J, D)

            x1 = t1.encode(skeletons)
            x2 = t2.encode(x1)
            fused = cross_attn(x1, x2, x2)
            pooled = fused.mean(dim=1)

            logits = gait_head(pooled)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy, all_preds, all_labels

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
    WINDOW_SIZE = 64
    num_classes = 60  # NTU has 60 classes
    T2_DROPOUT = 0.2
    CROSS_ATTN_DROPOUT = 0.2
    HEAD_DROPOUT = 0.3  # used to be 0.2
    SPLIT = "CV" # "CS" for cross subject, "CV" for cross view

    if SPLIT == "CS":
        DATA_PATH = "NTU60_CS.npz" # for cross subject
    elif SPLIT == "CV":
        DATA_PATH = "NTU60_CV.npz" # for cross view
    else:
        raise ValueError("Invalid split type. Choose either 'CS' or 'CV'.")

    # Set the device

    # transformer parameters
    hidden_size = 512 # 768 for CS, 512 for CV
    n_heads = 8 # 16 for CS, 8 for CV
    num_layers = 12 # 16 for CS, 12 for CV
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting NTU dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
        window_size=WINDOW_SIZE,
        p_interval=[0.95],
        vel=False,
        bone=False,
        debug=False
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1(f"action_checkpoints/NTU_{SPLIT}/NTU_pretrained.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
    else:
        t1 = load_T1(f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_T1.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    # load T2 model
    t2 = load_T2(f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, t2_dropout=T2_DROPOUT, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn_with_ffn(f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_cross_attn.pt", d_model=hidden_size, device=device, nhead=n_heads, dropout=CROSS_ATTN_DROPOUT)

    # load the gait recognition head
    gait_head = GaitRecognitionHeadMLP(input_dim=hidden_size, num_classes=num_classes, dropout=HEAD_DROPOUT)
    gait_head.load_state_dict(torch.load(f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting evaluation...")
    print("=" * 50)
    accuracy, all_preds, all_labels = evaluate(
        test_loader,
        t1,
        t2,
        cross_attn,
        gait_head,
        device=device
    )


    conf_mat = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=np.arange(num_classes))

    # Normalize confusion matrix by true class (row-wise normalization)
    conf_mat_normalized = conf_mat.astype(np.float32) / conf_mat.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat_normalized, cmap="viridis", square=True,
                cbar_kws={'label': 'Normalized Frequency'},
                xticklabels=False, yticklabels=False)

    plt.title("Normalized Confusion Matrix (Covariance-like Visualization)", fontsize=14)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("ntu_confusion_matrix.png", dpi=300)

    # Define the 11 two-person interaction class indices (from NTU60)
    two_person_class_indices = list(range(49, 60))

    # Per-class accuracy
    class_accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
    two_person_acc = class_accuracies[two_person_class_indices]
    non_two_person_indices = [i for i in range(60) if i not in two_person_class_indices]
    non_two_person_acc = class_accuracies[non_two_person_indices]

    print("=" * 50)
    print("[INFO] Evaluation completed!")
    print(f"two-person interaction accuracy: {two_person_acc.mean():.4f}")
    print(f"non-two-person interaction accuracy: {non_two_person_acc.mean():.4f}")
    print(f"🥶Final Accuracy🥶: {accuracy:.4f}🥶")
    print("=" * 50)


if __name__ == "__main__":
    main()