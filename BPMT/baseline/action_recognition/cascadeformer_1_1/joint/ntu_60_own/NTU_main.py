import numpy as np
import torch
import argparse
from NTU_feeder import Feeder
from NTU_pretraining import train_T1, BaseT1
from finetuning import load_T1, finetuning, GaitRecognitionHeadMLP
from penn_utils import set_seed
from NTU_utils import NUM_JOINTS_NTU

def load_cached_data(path="ntu_cache_train_sub.npz"):
    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    labels = list(data["labels"])
    return sequences, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--pretrain", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    device = args.device
    pretrain = args.pretrain
    WINDOW_SIZE = 64
    T2_DROPOUT = 0.2
    CROSS_ATTN_DROPOUT = 0.2
    HEAD_DROPOUT = 0.3  # used to be 0.2
    LR_LOWER_BOUND = 1e-6 # tune the lower bound for the learning rate
    SPLIT = "CV" # "CS" for cross subject, "CV" for cross view (136263)

    if SPLIT == "CS":
        DATA_PATH = "NTU60_CS.npz" # for cross subject
    elif SPLIT == "CV":
        DATA_PATH = "NTU60_CV.npz" # for cross view
    else:
        raise ValueError("Invalid split type. Choose either 'CS' or 'CV'.")

    mask_strategy = "global_joint"
    num_classes = 60 # NTU has 60 classes
    mask_ratio = 0.3

    # transformer parameters
    hidden_size = 512 # 768 for CS, 512 for CV
    n_heads = 8 # 16 for CS, 8 for CV
    num_layers = 12 # 16 for CS, 12 for CV

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the dataset
    import time
    t_start = time.time()
    
    # train_feeder_args:
    #   data_path: data/ntu/NTU60_CS.npz
    #   split: train
    #   debug: False
    #   random_choose: False
    #   random_shift: False
    #   random_move: False
    #   window_size: 64
    #   normalization: False
    #   random_rot: True
    #   p_interval: [0.5, 1]
    #   vel: False
    #   bone: False
    train_dataset = Feeder(
        data_path=DATA_PATH,
        split='train',
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=True,
        p_interval=[0.5, 1],
        vel=False,
        bone=False
    )

    # test_feeder_args:
    #   data_path: data/ntu/NTU60_CS.npz
    #   split: test
    #   window_size: 64
    #   p_interval: [0.95]
    #   vel: False
    #   bone: False
    #   debug: False
    val_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
        window_size=WINDOW_SIZE,
        p_interval=[0.95],
        vel=False,
        bone=False,
        debug=False
    )
    t_end = time.time()
    print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds", flush=True)

    if pretrain: 
        """
            pretraining on the whole dataset
        """

        print("\n==========================", flush=True)
        print("Starting Pretraining...", flush=True)
        print("==========================", flush=True)

        # instantiate the model
        three_d = True

        model = BaseT1(
            num_joints=NUM_JOINTS_NTU,
            three_d=three_d,
            d_model=hidden_size,
            nhead=n_heads,
            num_layers=num_layers,
        ).to(device)
        
        # training
        # dataset, model, num_epochs=50, batch_size=16, lr=1e-4, mask_ratio=0.15, device='cuda'):

        lr = 1e-4
        print(f"[INFO] Mask ratio: {mask_ratio * 100}%", flush=True)
        train_T1(
            masking_strategy=mask_strategy,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            mask_ratio=mask_ratio,
            device=device
        )

        # save pretrained model 
        torch.save(model.state_dict(), f"action_checkpoints/NTU_{SPLIT}/NTU_pretrained.pt")

        print("Aha! pretraining is done!", flush=True)
        print("=" * 100, flush=True)


    print("=" * 100, flush=True)
    print("=" * 100, flush=True)
    print("=" * 100, flush=True)

    # load T1 models
    three_d = True
    t1 = load_T1(
        model_path=f"action_checkpoints/NTU_{SPLIT}/NTU_pretrained.pt",
        num_joints=NUM_JOINTS_NTU,
        three_d=three_d,
        d_model=hidden_size,
        nhead=n_heads,
        num_layers=num_layers,
        freeze=True,
        device=device
    )

    print("pretrained model loaded successfully!", flush=True)


    train_finetuning_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_finetuning_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    gait_head_template = GaitRecognitionHeadMLP(
        input_dim=hidden_size, 
        num_classes=num_classes,
        dropout=HEAD_DROPOUT
        ).to(device)

    freezeT1 = False
    unfreeze_layers = None # freeze all layers

    ft_lr = 1e-4 # 3e-5 for CS, 1e-4 for CV
    wd = 1e-2
    trained_T2, train_cross_attn, train_head = finetuning(
        train_loader=train_finetuning_dataloader,
        val_loader=val_finetuning_dataloader,
        t1=t1,
        gait_head=gait_head_template,
        d_model=hidden_size,
        nhead=n_heads,
        num_layers=num_layers,
        num_epochs=num_epochs,
        lr=ft_lr,
        wd=wd,
        freezeT1=freezeT1,
        t2_dropout=T2_DROPOUT,
        cross_attn_dropout=CROSS_ATTN_DROPOUT,
        unfreeze_layers=unfreeze_layers,
        lr_lower_bound=LR_LOWER_BOUND,
        device=device
    )

    print("Aha! Finetuning completed successfully!", flush=True)

    # save the finetuned models
    torch.save(trained_T2.state_dict(), f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_T2.pt")
    torch.save(train_cross_attn.state_dict(), f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_cross_attn.pt")
    torch.save(train_head.state_dict(), f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_head.pt")

    if any(param.requires_grad for param in t1.parameters()):
        torch.save(t1.state_dict(), f"action_checkpoints/NTU_{SPLIT}/NTU_finetuned_T1.pt")

    print("Aha! finetuned models saved successfully!", flush=True)


if __name__ == "__main__":
    main()
