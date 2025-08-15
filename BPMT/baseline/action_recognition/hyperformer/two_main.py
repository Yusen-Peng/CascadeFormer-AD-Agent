import torch
import argparse
from HyperFormer import Model as HyperFormer
from NTU_feeder import Feeder
from penn_utils import set_seed
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm


def adjust_learning_rate(
        epoch: int, 
        optimizer: optim.SGD, 
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


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--train", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()

def main():
    set_seed(42)
    NUM_EPOCHS = 140  # HyperFormer uses 140 epochs
    BATCH_SIZE = 64  # HyperFormer uses 64 batch size for both train and val
    NUM_CLASSES = 60  # NTU has 60 classes
    NUM_JOINTS_NTU = 25  # NTU skeleton has 25 joints
    NUM_PERSON = 2  # it has to be 2 here!!
    WINDOW_SIZE = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    JOINT_LABELS = [0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1]
    args = parse_args()
    TRAIN = args.train

    # load the dataset
    import time
    t_start = time.time()
    
    train_dataset = Feeder(
        data_path="NTU60_CS.npz",
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
    ) # matches official HyperFormer configs

    val_dataset = Feeder(
        data_path="NTU60_CS.npz",
        split='test',
        window_size=WINDOW_SIZE,
        p_interval=[0.95],
        vel=False,
        bone=False,
        debug=False
    ) # matches official HyperFormer configs
    t_end = time.time()
    print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds")

    if TRAIN:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = HyperFormer(
            num_class=NUM_CLASSES,
            num_point=NUM_JOINTS_NTU,
            num_person=NUM_PERSON, # this time, we are enforcing two persons!
            graph='graph.ntu_rgb_d.Graph', # match!
            graph_args={'labeling_mode': 'spatial'}, # match!
            joint_label=JOINT_LABELS,
            in_channels=3, # match the default!
            drop_out=0.0, # original is actually 0
            num_of_heads=9 # match the default!
        ).to(device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.025, # match!
            momentum=0.9, # match the default!
            nesterov=True, # match!
            weight_decay=0.0004 # match!
        )
        
        for epoch in trange(NUM_EPOCHS, desc="Training Progress"):

            # adjust learning rate at the BEGINNING of each epoch
            lr = adjust_learning_rate(epoch, optimizer)
            tqdm.write(f"[Epoch {epoch+1}] Learning Rate: {lr:.6f}")

            model.train()
            total_loss, correct, total = 0, 0, 0

            for data, label, _ in tqdm(train_loader, desc=f"Train [{epoch+1}]"):
                with torch.no_grad():
                    data = data.float().to(device)
                    label = label.to(device)

                output, _ = model(data, label) 
                loss = F.cross_entropy(output, label)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * label.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

            train_acc = correct / total
            tqdm.write(f"[Epoch {epoch+1}] Train Loss: {total_loss / total:.4f} | Acc: {train_acc:.4f}")

            # validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for data, label, _ in tqdm(val_loader, desc=f"Val [{epoch+1}]"):
                    data = data.float().to(device)
                    label = label.to(device)

                    output, _ = model(data, label)
                    loss = F.cross_entropy(output, label)

                    val_loss += loss.item() * label.size(0)
                    pred = output.argmax(dim=1)
                    correct += (pred == label).sum().item()
                    total += label.size(0)

            val_acc = correct / total
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss / total:.4f} | Acc: {val_acc:.4f}")


        print("Training completed successfully!")
        # Save the model
        torch.save(model.state_dict(), "action_checkpoints/HYPER_DOUBLE/NTU_hyperformer.pt")
        print("Model saved successfully!")

    else:
        print("no training, just testing...")

    # test set evaluation
    print("=" * 50)
    print(f"[INFO] Starting NTU dataset testing on {device}...")
    print("=" * 50)
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Load the model
    model = HyperFormer(
        num_class=NUM_CLASSES,
        num_point=NUM_JOINTS_NTU,
        num_person=NUM_PERSON,
        graph='graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        joint_label=JOINT_LABELS,
        in_channels=3,
        drop_out=0.0,
        num_of_heads=9
    ).to(device)
    model.load_state_dict(torch.load("action_checkpoints/HYPER_DOUBLE/NTU_hyperformer.pt"))
    print("[INFO] Model loaded successfully!")

    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, label, _ in tqdm(test_loader, desc="Testing"):
            data = data.float().to(device)
            label = label.to(device)

            output, _ = model(data, label)
            loss = F.cross_entropy(output, label)

            test_loss += loss.item() * label.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    test_acc = correct / total
    print(f"[Test] Loss: {test_loss / total:.4f} | Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

