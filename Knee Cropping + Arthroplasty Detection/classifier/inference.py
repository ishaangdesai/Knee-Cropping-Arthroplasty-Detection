"""Shared model-loading/prediction wrapper used by evaluate_folder.py and app.py."""

from pathlib import Path

import torch
from PIL import Image

from dataset import build_transform
from model import build_model

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best_model" / "model.pt"

# Deliberately conservative: only call something "metal" if the model is >=90%
# sure. Anything under that is classified "regular" - i.e. false positives
# (calling a regular knee "metal") are penalized much more than false
# negatives here.
METAL_THRESHOLD = 0.90


class KneeClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.class_names = checkpoint["class_names"]  # ["regular", "metal"]
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transform(train=False)
        self.checkpoint_info = {
            "epoch": checkpoint.get("epoch"),
            "val_auc": checkpoint.get("val_auc"),
            "val_acc": checkpoint.get("val_acc"),
        }

    @torch.no_grad()
    def predict_image(self, img: Image.Image):
        """Returns (predicted_label: str, probability_metal: float)."""
        tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        logit = self.model(tensor)
        prob_metal = torch.sigmoid(logit).item()
        label = self.class_names[1] if prob_metal >= METAL_THRESHOLD else self.class_names[0]
        return label, prob_metal

    def predict_path(self, path: Path):
        with Image.open(path) as img:
            return self.predict_image(img)
