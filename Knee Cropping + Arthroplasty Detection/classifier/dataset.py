"""
Dataset assembly for the knee metal/regular classifier.

Ground truth comes from two manually-sorted folders at the project root:

    Metal/     - knee crops the user hand-labeled as a metal replacement
    Regular/   - knee crops the user hand-labeled as a native/regular knee

(Sorted with image-sorter.html, which copies files straight to disk.)

Earlier this project inferred labels from a "_metal" substring already
present in the source filenames (dataset/crops/left, dataset/crops/right).
That turned out to be unreliable - cross-checking it against the manually
sorted folders showed dozens of mismatches (e.g. files with "_metal" in the
name that the user placed in Regular) - so it is NOT used for labels
anymore. It's mentioned here only because file *paths* still route through
dataset/crops/{left,right}/<filename>: the Metal/Regular folders just
record which filename belongs to which class.

Every file in Metal/ and Regular/ is checked to be a distinct scan (no scan
appears in both crops/left and crops/right, and none is shared between the
two label folders), so this is a small, clean, 1-crop-per-patient dataset -
174 images total as of the last count, with only ~21 positive (metal)
examples. That's thin for a CNN; splits below are stratified by class and a
random seed is fixed so results are at least reproducible, but treat any
metrics from this dataset size as a rough first pass, not a reliable
estimate.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from preprocessing import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SIZE,
    standardize_exposure,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_DIRS = {1: PROJECT_ROOT / "Metal", 0: PROJECT_ROOT / "Regular"}

CLASS_NAMES = ["regular", "metal"]  # index 0 / 1

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42


@dataclass
class Sample:
    path: Path
    label: int  # 0 = regular, 1 = metal
    scan_id: str  # patient/scan this crop came from, used for grouping


def _scan_id_from_crop(path: Path) -> str:
    """'EEC2111D_Lcutoffmetal_knee1' -> 'EEC2111D' (everything before the first underscore)."""
    return path.stem.split("_")[0]


def load_all_samples() -> List[Sample]:
    samples = []
    for label, label_dir in LABEL_DIRS.items():
        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected labeled folder at {label_dir} - it should contain the "
                f"hand-sorted '{CLASS_NAMES[label]}' knee crops."
            )
        for path in sorted(label_dir.iterdir()):
            if path.is_file():
                samples.append(Sample(path=path, label=label, scan_id=_scan_id_from_crop(path)))
    return samples


def split_samples(samples: List[Sample]):
    """Stratified split by class, grouped by scan_id so a patient's crops
    never straddle two splits. Returns (train, val, test)."""
    rng = random.Random(RANDOM_SEED)
    train_set, val_set, test_set = [], [], []

    for label in (0, 1):
        class_samples = [s for s in samples if s.label == label]
        scan_ids = sorted({s.scan_id for s in class_samples})
        rng.shuffle(scan_ids)

        n = len(scan_ids)
        n_test = max(1, round(n * TEST_FRACTION))
        n_val = max(1, round(n * VAL_FRACTION))
        n_test = min(n_test, n - 2) if n > 2 else min(n_test, n - 1) if n > 1 else 0
        n_val = min(n_val, max(n - n_test - 1, 0))

        test_ids = set(scan_ids[:n_test])
        val_ids = set(scan_ids[n_test:n_test + n_val])

        for s in class_samples:
            if s.scan_id in test_ids:
                test_set.append(s)
            elif s.scan_id in val_ids:
                val_set.append(s)
            else:
                train_set.append(s)

    return train_set, val_set, test_set


def build_transform(train: bool) -> transforms.Compose:
    ops = [standardize_exposure]
    if train:
        ops += [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ]
    else:
        ops += [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class KneeCropDataset(Dataset):
    def __init__(self, samples: List[Sample], train: bool):
        self.samples = samples
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        img = self.transform(img)
        return img, float(sample.label)


if __name__ == "__main__":
    all_samples = load_all_samples()
    train_set, val_set, test_set = split_samples(all_samples)

    def counts(name, s):
        n_metal = sum(x.label == 1 for x in s)
        print(f"{name:12s} total={len(s):5d}  metal={n_metal:5d}  regular={len(s) - n_metal:5d}")

    counts("all", all_samples)
    counts("train", train_set)
    counts("val", val_set)
    counts("test", test_set)
