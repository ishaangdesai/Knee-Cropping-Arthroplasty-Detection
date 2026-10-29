"""
Train the knee metal-vs-regular classifier.

    conda activate kneelabel
    python train.py

Trains until EARLY_STOPPING_PATIENCE epochs pass with no validation ROC-AUC
improvement (or EPOCHS is reached), then saves the LAST epoch's weights to
best_model/model.pt - that's what evaluate_folder.py/app.py load. The
best-by-val-auc epoch along the way is kept separately at
best_model/model_best_val_auc.pt for comparison. Also writes metrics.json
and a training curve plot.
"""

import gpu_env  # noqa: F401 - must be imported before torch, see gpu_env.py

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLASS_NAMES, KneeCropDataset, load_all_samples, split_samples
from model import ARCHITECTURE, build_model
from preprocessing import IMAGE_SIZE

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 12

# Same reasoning as dataset/train.py: bfloat16 has float32's exponent range, so it
# doesn't need loss scaling and won't overflow to NaN the way float16 can.
AMP_DTYPE = torch.bfloat16
ENABLE_MIXED_PRECISION = True

OUTPUT_DIR = Path(__file__).resolve().parent / "best_model"


def run_epoch(model, loader, criterion, optimizer, device, train: bool, amp_enabled: bool, desc: str):
    model.train(train)
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            with autocast(device.type, dtype=AMP_DTYPE, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.detach().cpu().numpy().ravel().tolist())
            all_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy().ravel().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    acc = sum((p >= 0.5) == (y >= 0.5) for p, y in zip(all_probs, all_labels)) / len(all_labels)
    return avg_loss, auc, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_samples = load_all_samples()
    train_samples, val_samples, test_samples = split_samples(all_samples)
    print(f"train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    train_ds = KneeCropDataset(train_samples, train=True)
    val_ds = KneeCropDataset(val_samples, train=False)

    # persistent_workers=True matches dataset/train.py - without it, DataLoader
    # respawns every worker process from scratch (re-importing torch/torchvision/cv2
    # in each) at the start of EVERY epoch. On Windows that spawn cost is large, and
    # shows up as the GPU sitting idle at 0% util while workers restart, even though
    # CUDA memory is already reserved from the previous epoch.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                               persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=NUM_WORKERS > 0)

    n_regular = sum(s.label == 0 for s in train_samples)
    n_metal = sum(s.label == 1 for s in train_samples)
    pos_weight = torch.tensor([n_regular / max(n_metal, 1)], device=device)
    print(f"Class balance in train set -> regular={n_regular} metal={n_metal}  "
          f"(pos_weight={pos_weight.item():.3f})")

    model = build_model(pretrained=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    OUTPUT_DIR.mkdir(exist_ok=True)

    amp_enabled = ENABLE_MIXED_PRECISION and device.type == "cuda"

    best_val_auc = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_auc, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            amp_enabled=amp_enabled, desc=f"Epoch {epoch}/{EPOCHS} (train)")
        val_loss, val_auc, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
            amp_enabled=amp_enabled, desc=f"Epoch {epoch}/{EPOCHS} (val)")
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_val_auc
        marker = "  <- best so far, saved" if improved else ""
        print(f"epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}  "
              f"({time.time() - t0:.1f}s){marker}")

        if improved:
            best_val_auc = val_auc
            epochs_without_improvement = 0
            # Kept only for reference/comparison - model.pt (what inference.py loads)
            # is overwritten with the LAST epoch's weights once training finishes,
            # not this best-by-val-auc snapshot. See the end of this loop.
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": ARCHITECTURE,
                "image_size": IMAGE_SIZE,
                "class_names": CLASS_NAMES,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "epoch": epoch,
            }, OUTPUT_DIR / "model_best_val_auc.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"No val_auc improvement for {EARLY_STOPPING_PATIENCE} epochs, stopping early.")
                break

    # model.pt (what inference.py/evaluate_folder.py/app.py load) is always the LAST
    # trained epoch's weights, not the best-by-val-auc one - the `model` object in
    # memory already reflects that epoch's weights since training never reloads a
    # checkpoint mid-run. model_best_val_auc.pt above is kept alongside it for
    # comparison.
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": ARCHITECTURE,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "val_auc": val_auc,
        "val_acc": val_acc,
        "epoch": epoch,
    }, OUTPUT_DIR / "model.pt")
    print(f"\nSaved LAST epoch ({epoch}, val_auc={val_auc:.4f}) to "
          f"{OUTPUT_DIR / 'model.pt'} - this is what evaluate_folder.py/app.py will use.")
    print(f"Best val_auc during training was {best_val_auc:.4f}, kept separately at "
          f"{OUTPUT_DIR / 'model_best_val_auc.pt'} for comparison.")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_auc": best_val_auc}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_auc"], label="train")
    axes[1].plot(history["val_auc"], label="val")
    axes[1].set_title("ROC-AUC")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    print("Run: python evaluate_folder.py --folder <test folder> for held-out statistics.")


if __name__ == "__main__":
    main()
