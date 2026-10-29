import os
import torch
import torchvision
import numpy as np
import cv2  # <--- Using OpenCV for speed
import csv
from typing import List, Tuple, Dict
from tqdm import tqdm
from PIL import Image

# Check for Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

from torch.utils.data import Dataset, DataLoader

# Import components for EfficientNet-FPN model
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import FasterRCNN

# ==============================================================================
# 							USER CUSTOMIZATIONS
# ==============================================================================

# Directory containing the model files
MODEL_DIR = 'model_fasterrcnn_efficientnet_b1'
# Directory of test images
TEST_DIR = 'all_images'
# Path to the trained model weights
PTH_FILE_PATH = os.path.join(MODEL_DIR, 'checkpoint_epoch_55.pth')

# Visualization type: 'box' for bounding boxes, 'center' for center points
SHOW_TYPE = 'box'
# Confidence threshold for saving detections
CONFIDENCE_THRESHOLD = 0.50 

# Enable or disable Grad-CAM visualization
SHOW_GRAD_CAM = False
GRAD_CAM_TARGET_LAYER = '8'

# INFERENCE OPTIMIZATION
MAX_IMAGE_SIZE = 1024  
BATCH_SIZE = 1 
# Set to 4 to prevent Windows freezing, but keep it parallel
NUM_WORKERS = 4 

# ==============================================================================
# 							HELPER CLASSES
# ==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANNOTATED_DIR = 'images_test_output_base_efficientnet_FPN'
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Individual knee-box crops, sorted by patient-side (left/right) and metal/regular
CROPS_DIR = 'crops'
for _side in ('left', 'right'):
    for _kind in ('metal', 'regular'):
        os.makedirs(os.path.join(CROPS_DIR, _side, _kind), exist_ok=True)

class InferenceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        # Standard PyTorch transform
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, filename
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return torch.zeros((3, 800, 800)), "ERROR"

def collate_fn(batch):
    return tuple(zip(*batch))

def get_object_detection_model(num_classes: int, backbone_name: str) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    efficientnet_b1 = torchvision.models.efficientnet_b1(weights=weights)

    return_nodes = {"2": "0", "4": "1", "6": "2", "8": "3"}
    backbone_body = efficientnet_b1.features
    in_channels_list = [24, 80, 192, 1280]
    out_channels = 256

    backbone = BackboneWithFPN(backbone_body, return_nodes, in_channels_list, out_channels)
    
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # STRICT SIZE ENFORCEMENT
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=MAX_IMAGE_SIZE,
        max_size=MAX_IMAGE_SIZE
    )
    return model

def save_knee_crops(
    file_name: str,
    img_uint8: np.ndarray,
    predictions_to_show: List[Tuple[str, List[float], float]]
):
    """
    Crops every 'knee' detection out of img_uint8 (BGR, un-annotated) and saves
    it as its own JPG under crops/<left|right>/<metal|regular>/.

    Side is the patient's actual knee, not image position: the X-ray is taken
    facing the patient, so it's mirrored -- a box on the left half of the
    image is the patient's RIGHT knee, and a box on the right half is the
    patient's LEFT knee.
    """
    img_h, img_w = img_uint8.shape[:2]
    mid_x = img_w / 2
    kind = 'metal' if 'metal' in file_name.lower() else 'regular'
    base_name = os.path.splitext(file_name)[0]

    knee_idx = 0
    for label, box, score in predictions_to_show:
        if label.lower() != 'knee':
            continue

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_uint8[y1:y2, x1:x2]

        box_center_x = (x1 + x2) / 2
        side = 'right' if box_center_x < mid_x else 'left'

        knee_idx += 1
        crop_filename = f"{base_name}_knee{knee_idx}.jpg"
        save_path = os.path.join(CROPS_DIR, side, kind, crop_filename)
        cv2.imwrite(save_path, crop)

def save_annotated_image_cv2(
    file_name: str, 
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    predictions: Tuple[List[str], List[List[float]], List[float]], 
    classes: List[str]
):
    """
    High-performance image saving using OpenCV.
    """
    # 1. Convert Tensor (C, H, W) RGB 0-1  ->  Numpy (H, W, C) BGR 0-255
    img_np = image_tensor.permute(1, 2, 0).numpy()
    
    labels, boxes, scores = predictions
    
    # Filter predictions
    predictions_to_show = []
    if CONFIDENCE_THRESHOLD == 0:
        best_preds_by_class = {}
        for l, b, s in zip(labels, boxes, scores):
            if l not in best_preds_by_class or s > best_preds_by_class[l][2]:
                best_preds_by_class[l] = (l, b, s)
        predictions_to_show = list(best_preds_by_class.values())
    else:
        for l, b, s in zip(labels, boxes, scores):
            if s >= CONFIDENCE_THRESHOLD:
                predictions_to_show.append((l, b, s))

    # --- Grad-CAM Logic (Optional) ---
    if SHOW_GRAD_CAM and predictions_to_show and GRAD_CAM_AVAILABLE:
        try:
            target_layers = [model.backbone.body[GRAD_CAM_TARGET_LAYER]]
            cam = GradCAM(model=model, target_layers=target_layers)
            
            labels_for_cam = [classes.index(p[0]) for p in predictions_to_show]
            boxes_for_cam = torch.tensor([p[1] for p in predictions_to_show], device=device)
            
            input_tensor = image_tensor.unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[FasterRCNNBoxScoreTarget(labels=labels_for_cam, bounding_boxes=boxes_for_cam)])[0, :]
            
            # Apply CAM overlay (Output is RGB float 0-1)
            img_np = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        except Exception as e:
            print(f"CAM Error for {file_name}: {e}")

    # 2. Finalize conversion to OpenCV BGR format
    # If Grad-CAM ran, img_np is RGB float. If not, it's RGB float.
    # We need BGR uint8.
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_uint8 = (img_bgr * 255).astype(np.uint8)

    # 3. Save individual knee crops (before annotations are drawn on img_uint8)
    save_knee_crops(file_name, img_uint8, predictions_to_show)

    # 4. Draw Annotations
    for label, box, score in predictions_to_show:
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{label}: {score:.2f}"

        if SHOW_TYPE == 'box':
            # Draw Rectangle (Green)
            cv2.rectangle(img_uint8, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Text Background (Green filled)
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_uint8, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            # Draw Text (Black)
            cv2.putText(img_uint8, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        elif SHOW_TYPE == 'center':
            # Draw Center Dot (Red)
            cx, cy = int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)
            cv2.circle(img_uint8, (cx, cy), 5, (0, 0, 255), -1)

            # Draw Text nearby
            cv2.putText(img_uint8, label_text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 5. Save to Disk (Fast)
    save_path = os.path.join(ANNOTATED_DIR, file_name)
    cv2.imwrite(save_path, img_uint8)

def main():
    print(f"Using device: {device}")
    
    # --- Load Model ---
    try:
        checkpoint = torch.load(PTH_FILE_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict']
        CLASSES = checkpoint['classes']
        num_classes = len(CLASSES)
        backbone_name = checkpoint.get('backbone_name', 'efficientnet_b1')
        
        print(f"Loading model... Backbone: {backbone_name}, Classes: {num_classes}")
        model = get_object_detection_model(num_classes, backbone_name)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded from '{PTH_FILE_PATH}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Data Loader ---
    dataset = InferenceDataset(TEST_DIR)
    if len(dataset) == 0:
        print(f"No images found in {TEST_DIR}")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    csv_rows = []
    print(f"Starting inference on {len(dataset)} images with Batch Size {BATCH_SIZE}...")
    print(f"Saving method: OpenCV (High Performance)")

    # --- Inference Loop ---
    with torch.no_grad():
        for batch_images, batch_filenames in tqdm(dataloader, desc="Processing"):
            valid_indices = [i for i, f in enumerate(batch_filenames) if f != "ERROR"]
            if not valid_indices:
                continue
            
            images = [batch_images[i].to(device) for i in valid_indices]
            filenames = [batch_filenames[i] for i in valid_indices]
            
            # GPU Prediction
            predictions = model(images)
            
            # Process Results
            for i, pred in enumerate(predictions):
                filename = filenames[i]
                img_tensor = batch_images[valid_indices[i]] # Keep on CPU
                
                pred_labels = [CLASSES[l] for l in pred['labels'].tolist()]
                pred_boxes = pred['boxes'].tolist()
                pred_scores = pred['scores'].tolist()
                
                # Append to CSV
                for l, b, s in zip(pred_labels, pred_boxes, pred_scores):
                    csv_rows.append([filename, l, b[0], b[1], b[2], b[3], s])
                
                # Save Image (if detections exist)
                if len(pred_scores) > 0:
                    save_annotated_image_cv2(
                        filename, 
                        img_tensor, 
                        model, 
                        (pred_labels, pred_boxes, pred_scores), 
                        CLASSES
                    )

    # --- Save CSV ---
    csv_path = 'predictions.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "class", "xmin", "ymin", "xmax", "ymax", "confidence"])
        writer.writerows(csv_rows)
        
    print(f"\nDone! Predictions saved to '{csv_path}'")
    print(f"Annotated images saved to '{ANNOTATED_DIR}'")

if __name__ == "__main__":
    main()