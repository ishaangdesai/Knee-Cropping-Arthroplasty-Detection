import os
import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import math

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torch.amp import autocast

# ==============================================================================
# 							USER CUSTOMIZATIONS
# ==============================================================================

# -----------------
# Project and File Paths
# -----------------
# Annotations file containing all bounding box and class data
ANNOTATIONS_FILE = 'merged_annotations.csv'
# Directory containing all training images referenced in the annotations file
DATA_DIR = 'images_train_cropped'

# -----------------
# Validation Set Configuration
# -----------------
VALIDATION_SPLIT = 0.05 # Percentage of data to use for validation (0-1)
RANDOM_SEED = 42

# -----------------
# Training Hyperparameters (Optimized for NVIDIA RTX 5090 w/ 32GB VRAM - see notes below)
# -----------------
# Set a high number of epochs; early stopping will terminate training when appropriate.
EPOCHS = 100
# REDUCED for RTX 5070 (12GB VRAM). BATCH_SIZE=8 at MAX_IMAGE_SIZE=1024 overflowed
# dedicated VRAM into slow shared/system memory (100% GPU util but ~30s/iteration).
BATCH_SIZE = 4
LEARNING_RATE = 0.0003
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
# REDUCED to match physical core count (8c/16t Ryzen 9 270) - 16 workers left no
# headroom for the main training process and oversubscribed the physical cores.
NUM_WORKERS = 8
# Max image size for training (0 = no constraint)
MAX_IMAGE_SIZE = 1024  
# Enable mixed precision training (faster on newer GPUs)
ENABLE_MIXED_PRECISION = True

# -----------------
# Early Stopping Configuration
# -----------------
# Stop training if validation mAP does not improve for this many consecutive epochs.
EARLY_STOPPING_PATIENCE = 50

# -----------------
# Checkpoint Configuration
# -----------------
# Save a checkpoint every N epochs. Set to 0 or less to disable.
CHECKPOINT_SAVE_INTERVAL = 5

# -----------------
# Model Backbone Configuration
# -----------------
# Backbone: "resnet50_fpn" | "resnet101_fpn" | "efficientnet_b0" | "mobilenet_v2" | "mobilenet_v3_large"
BACKBONE_NAME = 'efficientnet_b1' # FasterRCNN with EfficientNet-B1 backbone

# -----------------
# Data Augmentation Configuration (all disabled by default)
# -----------------
# Basic augmentations
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_COLOR_JITTER = False
# Advanced augmentations (disabled by default)
AUGMENT_ROTATION = False  # Random rotation of images
AUGMENT_ROTATION_DEGREES = 10  # Max rotation degrees
AUGMENT_SCALE = True  # Random scale of images
AUGMENT_SCALE_FACTOR = 0.2  # Scale factor range (1±factor)
AUGMENT_TRANSLATE = True  # Random translation of images
AUGMENT_TRANSLATE_FACTOR = 0.1  # Max translation as fraction of image size


# ==============================================================================
# 							HELPER FUNCTIONS AND CLASSES
# ==============================================================================

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, root_dir: str, annotations_df: pd.DataFrame, classes: list, transforms: Any = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.image_filenames = sorted(annotations_df['filename'].unique())
        
        # More efficient pre-processing of annotations
        self.processed_annotations = {}
        for filename, group in annotations_df.groupby('filename'):
            boxes_list = []
            labels_list = []
            
            for _, row in group.iterrows():
                class_name = row['class']
                if class_name in self.classes:
                    label_id = self.class_to_idx[class_name]
                    img_width = int(row['width'])
                    img_height = int(row['height'])
                    # Clip to image bounds defensively: some annotation sources include
                    # coordinates slightly outside the frame (e.g. objects marked as
                    # cut off at the edge with a negative xmin/ymin).
                    xmin = max(0, min(int(row['xmin']), img_width))
                    ymin = max(0, min(int(row['ymin']), img_height))
                    xmax = max(0, min(int(row['xmax']), img_width))
                    ymax = max(0, min(int(row['ymax']), img_height))

                    if xmax > xmin and ymax > ymin:
                        boxes_list.append([xmin, ymin, xmax, ymax])
                        labels_list.append(label_id)
                    else:
                        print(f"Warning: Invalid box coords {class_name} ({xmin},{ymin},{xmax},{ymax}) in {filename}. Skipping.")
            
            # Store pre-processed data for each image
            self.processed_annotations[filename] = {
                'boxes': torch.as_tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.as_tensor(labels_list, dtype=torch.int64) if labels_list else torch.zeros((0,), dtype=torch.int64)
            }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # Use pre-processed annotations (optimized)
        annotation_data = self.processed_annotations[img_name]
        boxes = annotation_data['boxes']
        labels = annotation_data['labels']

        # Calculate areas once
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
            
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target: Dict[str, torch.Tensor] = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.image_filenames)

class Compose:
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, torch.Tensor]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = F.to_tensor(image)
        return image, target

def _clip_and_filter_boxes(
    boxes: torch.Tensor, labels: torch.Tensor, width: int, height: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Clamp boxes into [0, width] x [0, height] and drop any that become degenerate.

    Returns (boxes, labels, area, iscrowd) for the surviving boxes, or None if none survive.
    """
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=height)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid.any():
        return None

    boxes = boxes[valid]
    labels = labels[valid]
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    return boxes, labels, area, iscrowd


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        if torch.rand(1).item() < self.prob:
            image_width = image.width
            boxes = target["boxes"]
            if boxes.numel() > 0:
                flipped_boxes = boxes.clone()
                flipped_boxes[:, [0, 2]] = image_width - flipped_boxes[:, [2, 0]]

                result = _clip_and_filter_boxes(flipped_boxes, target["labels"], image_width, image.height)
                if result is None:
                    # Flip would remove every box; skip this augmentation for this sample.
                    return image, target

                boxes, labels, area, iscrowd = result
                target = dict(target)
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
            return F.hflip(image), target
        return image, target

class ColorJitter:
    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0, hue: float = 0):
        self.color_jitter_transform = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        image = self.color_jitter_transform(image)
        return image, target

class RandomRotation:
    def __init__(self, degrees: float = 10.0):
        self.degrees = degrees

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        angle = torch.FloatTensor(1).uniform_(-self.degrees, self.degrees).item()
        
        # Get image center
        width, height = image.size
        center = (width // 2, height // 2)
        
        # Rotate image
        rotated_image = F.rotate(image, angle, expand=False)
        
        # Rotate boxes if they exist
        boxes = target["boxes"]
        if boxes.numel() > 0:
            # Convert angle to radians
            angle_rad = -math.radians(angle)
            cos_val = math.cos(angle_rad)
            sin_val = math.sin(angle_rad)
            
            # Adjust box coordinates (simplified - works best for small angles)
            new_boxes = []
            keep_mask = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                
                # Box corners
                corners = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]
                ]
                
                # Rotate each corner
                rotated_corners = []
                for x, y in corners:
                    # Translate to origin
                    x -= center[0]
                    y -= center[1]
                    
                    # Rotate
                    new_x = cos_val * x - sin_val * y
                    new_y = sin_val * x + cos_val * y
                    
                    # Translate back
                    new_x += center[0]
                    new_y += center[1]
                    
                    rotated_corners.append([new_x, new_y])
                
                # Get bounding box of rotated corners
                xs = [c[0] for c in rotated_corners]
                ys = [c[1] for c in rotated_corners]
                new_xmin = max(0, min(xs))
                new_ymin = max(0, min(ys))
                new_xmax = min(width, max(xs))
                new_ymax = min(height, max(ys))

                # Only add valid boxes, tracking which original box each one came from
                # so labels/area/iscrowd stay aligned with boxes.
                keep_mask.append(new_xmax > new_xmin and new_ymax > new_ymin)
                if keep_mask[-1]:
                    new_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

            if not new_boxes:
                # Rotation would remove every box; skip this augmentation for this sample.
                return image, target

            keep_mask_t = torch.tensor(keep_mask, dtype=torch.bool)
            new_boxes_t = torch.tensor(new_boxes, dtype=boxes.dtype)
            new_labels = target["labels"][keep_mask_t]
            area = (new_boxes_t[:, 3] - new_boxes_t[:, 1]) * (new_boxes_t[:, 2] - new_boxes_t[:, 0])
            iscrowd = torch.zeros((new_boxes_t.shape[0],), dtype=torch.int64)

            target = dict(target)
            target["boxes"] = new_boxes_t
            target["labels"] = new_labels
            target["area"] = area
            target["iscrowd"] = iscrowd

        return rotated_image, target

class RandomScale:
    def __init__(self, scale_factor: float = 0.2):
        self.scale_factor = scale_factor

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        scale = torch.FloatTensor(1).uniform_(1-self.scale_factor, 1+self.scale_factor).item()

        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Adjust box coordinates
        boxes = target["boxes"]
        if boxes.numel() > 0:
            scaled_boxes = boxes * scale

            # new_width/new_height are floor()'d while box coords use the exact float
            # scale, so a box that touched the original edge can land fractionally
            # outside the resized frame - clip it back in.
            result = _clip_and_filter_boxes(scaled_boxes, target["labels"], new_width, new_height)
            if result is None:
                # Scaling would remove every box; skip this augmentation for this sample.
                return image, target

            boxes, labels, area, iscrowd = result
            target = dict(target)
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd

        resized_image = F.resize(image, [new_height, new_width])
        return resized_image, target

class RandomTranslate:
    def __init__(self, translate_factor: float = 0.1):
        self.translate_factor = translate_factor

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        width, height = image.size
        max_dx = int(width * self.translate_factor)
        max_dy = int(height * self.translate_factor)

        # random_(from, to) requires from < to, which fails for very narrow/short
        # images where int(size * translate_factor) rounds down to 0.
        dx = torch.IntTensor(1).random_(-max_dx, max_dx).item() if max_dx > 0 else 0
        dy = torch.IntTensor(1).random_(-max_dy, max_dy).item() if max_dy > 0 else 0

        # Adjust box coordinates
        boxes = target["boxes"]
        if boxes.numel() > 0:
            translated_boxes = boxes.clone()
            translated_boxes[:, [0, 2]] += dx
            translated_boxes[:, [1, 3]] += dy

            # Boxes near the edge (e.g. objects annotated as cut off by the frame) can
            # get pushed fully outside the image and clamped to zero size. If this
            # translation would remove every box in the image, skip it entirely rather
            # than silently handing the model a background-only (empty target) sample.
            result = _clip_and_filter_boxes(translated_boxes, target["labels"], width, height)
            if result is None:
                return image, target

            boxes, labels, area, iscrowd = result
            target = dict(target)
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd

        # Create translated image with padding
        translated_image = Image.new(image.mode, (width, height), (0, 0, 0))
        translated_image.paste(image, (dx, dy))
        return translated_image, target

class ResizeWithMaxSize:
    def __init__(self, max_size: int):
        self.max_size = max_size
    
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        width, height = image.size
        
        # No resizing needed if max_size is 0 or image is smaller
        if self.max_size == 0 or (width <= self.max_size and height <= self.max_size):
            return image, target
        
        # Calculate scale factor to maintain aspect ratio
        scale = self.max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Scale bounding boxes
        boxes = target["boxes"]
        if boxes.numel() > 0:
            scaled_boxes = boxes * scale

            # new_width/new_height are floor()'d while box coords use the exact float
            # scale, so a box that touched the original edge can land fractionally
            # outside the resized frame - clip it back in. This runs on every image
            # (train and val), so an empty result here would mean the source
            # annotation had no valid box to begin with; keep the (now-empty) result
            # rather than silently reintroducing an invalid box.
            result = _clip_and_filter_boxes(scaled_boxes, target["labels"], new_width, new_height)
            if result is not None:
                boxes, labels, area, iscrowd = result
                target = dict(target)
                target["boxes"] = boxes
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
            else:
                target = dict(target)
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64)
                target["area"] = torch.zeros((0,), dtype=torch.float32)
                target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        resized_image = F.resize(image, [new_height, new_width])
        return resized_image, target

def get_transform(train: bool):
    transforms_list = []
    
    # Add max size constraint (always applied if configured)
    if MAX_IMAGE_SIZE > 0:
        transforms_list.append(ResizeWithMaxSize(MAX_IMAGE_SIZE))
    
    if train:
        # Basic augmentations (controlled by parameters)
        if AUGMENT_COLOR_JITTER:
            transforms_list.append(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
        if AUGMENT_HORIZONTAL_FLIP:
            transforms_list.append(RandomHorizontalFlip(0.5))
            
        # Advanced augmentations (off by default)
        if AUGMENT_ROTATION:
            transforms_list.append(RandomRotation(AUGMENT_ROTATION_DEGREES))
        if AUGMENT_SCALE:
            transforms_list.append(RandomScale(AUGMENT_SCALE_FACTOR))
        if AUGMENT_TRANSLATE:
            transforms_list.append(RandomTranslate(AUGMENT_TRANSLATE_FACTOR))
    
    # Always convert to tensor at the end
    transforms_list.append(ToTensor())
    return Compose(transforms_list)

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    return tuple(zip(*batch))

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.feature_extraction import create_feature_extractor

# ==============================================================================
# 							MODEL DEFINITION
# ==============================================================================

def get_object_detection_model(num_classes: int, backbone_name: str) -> torch.nn.Module:
    """
    Creates a Faster R-CNN model with an EfficientNet-B1 backbone and a Feature Pyramid Network (FPN).
    This allows the model to detect objects at multiple scales, improving sensitivity for small objects.
    """
    # 1. Load a pre-trained EfficientNet-B1 model
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    efficientnet_b1 = torchvision.models.efficientnet_b1(weights=weights)

    # 2. Define the layers from which to extract features for the FPN.
    # The keys ("2", "4", "6", "8") correspond to the end of major stages in the
    # `features` block of the EfficientNet model, providing a stable set of
    # multi-scale feature maps.
    return_nodes = {
        "2": "0",  # End of stage 3
        "4": "1",  # End of stage 4
        "6": "2",  # End of stage 6
        "8": "3",  # Final feature layer
    }
    
    # The body of the backbone is the 'features' section of the EfficientNet model.
    backbone_body = efficientnet_b1.features

    # 4. Define the input channel depths for each of the selected feature maps.
    # These values have been programmatically verified.
    # For EfficientNet-B1, these are the output channels of features[2], [4], [6], and [8].
    in_channels_list = [24, 80, 192, 1280]
    
    # The FPN will transform these feature maps to have a consistent depth of 256 channels.
    out_channels = 256

    # 5. Create the backbone with FPN.
    # We pass the main 'features' block of the model and the return_nodes dictionary.
    # The BackboneWithFPN utility will internally use IntermediateLayerGetter to
    # extract the feature maps from the specified nodes.
    backbone = BackboneWithFPN(backbone_body, return_nodes, in_channels_list, out_channels)

    # 6. Define a new AnchorGenerator with an additional smaller anchor size (16)
    # to improve sensitivity for small keypoints. The sizes correspond to the 5 feature
    # maps generated by the FPN (P2, P3, P4, P5, P6).
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # 7. The RoI Pooler will be automatically created by the FasterRCNN constructor
    # based on the FPN output. We don't need to define it manually.

    # 8. Create the final Faster R-CNN model.
    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # CUSTOM MODIFICATION: Force model to respect input sizes
        # This prevents internal upscaling/resizing which wastes VRAM and GPU cycles
        min_size=MAX_IMAGE_SIZE, 
        max_size=MAX_IMAGE_SIZE
    )
    
    return model


# ==============================================================================
# 							SCRIPT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Read Annotations and Determine Classes ---
    try:
        annotations_df = pd.read_csv(ANNOTATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at '{ANNOTATIONS_FILE}'. Please check the path.")
        exit()

    unique_classes = sorted(annotations_df['class'].unique())
    # Ensure '__background__' is the first class for torchvision models
    CLASSES = ['__background__'] + unique_classes
    num_classes = len(CLASSES)

    # --- Construct Model Directory Name ---
    ARCHITECTURE_NAME = "fasterrcnn"
    MODEL_DIR = f"model_{ARCHITECTURE_NAME}_{BACKBONE_NAME}"

    # --- Create output directory ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Output directory '{MODEL_DIR}' created or already exists.")

    # --- Display Configuration ---
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"{ 'Classes:':<30} {num_classes} - {', '.join(CLASSES)}")
    print(f"{ 'Annotations File:':<30} {ANNOTATIONS_FILE}")
    print(f"{ 'Image Directory:':<30} {DATA_DIR}")
    print(f"{ 'Output Model Directory:':<30} {MODEL_DIR}")
    print(f"{ 'Validation Split:':<30} {VALIDATION_SPLIT}")
    print(f"{ 'Epochs:':<30} {EPOCHS}")
    print(f"{ 'Batch Size:':<30} {BATCH_SIZE}")
    print(f"{ 'Learning Rate:':<30} {LEARNING_RATE}")
    print(f"{ 'Num Workers:':<30} {NUM_WORKERS}")
    print(f"{ 'Max Image Size:':<30} {MAX_IMAGE_SIZE if MAX_IMAGE_SIZE > 0 else 'No constraint'}")
    print(f"{ 'Mixed Precision Training:':<30} {'Enabled' if ENABLE_MIXED_PRECISION else 'Disabled'}")
    print(f"{ 'Early Stopping Patience:':<30} {EARLY_STOPPING_PATIENCE} epochs")
    print(f"{ 'Checkpoint Interval:':<30} {CHECKPOINT_SAVE_INTERVAL if CHECKPOINT_SAVE_INTERVAL > 0 else 'Disabled'}")
    print(f"{ 'Architecture:':<30} {ARCHITECTURE_NAME.upper()}")
    print(f"{ 'Backbone:':<30} {BACKBONE_NAME}")
    print("Data Augmentation:")
    print(f"  { 'Horizontal Flip:':<30} {'Enabled' if AUGMENT_HORIZONTAL_FLIP else 'Disabled'}")
    print(f"  { 'Color Jitter:':<30} {'Enabled' if AUGMENT_COLOR_JITTER else 'Disabled'}")
    print(f"  { 'Rotation:':<30} {'Enabled' if AUGMENT_ROTATION else 'Disabled'} ({AUGMENT_ROTATION_DEGREES}°)")
    print(f"  { 'Scale:':<30} {'Enabled' if AUGMENT_SCALE else 'Disabled'} (±{AUGMENT_SCALE_FACTOR})")
    print(f"  { 'Translate:':<30} {'Enabled' if AUGMENT_TRANSLATE else 'Disabled'} (±{AUGMENT_TRANSLATE_FACTOR})")
    print("="*60)
    print(f"Please ensure 'torchmetrics' is installed (pip install torchmetrics).\n")

    if not 0.0 <= VALIDATION_SPLIT < 1.0:
        raise ValueError("VALIDATION_SPLIT must be between 0.0 and 1.0.")

    # Note: The code block that wrote classes.txt has been removed from here

    # --- Data Loading and Splitting ---
    all_image_filenames = sorted(annotations_df['filename'].unique())

    if not all_image_filenames:
        print(f"Error: No image filenames found in '{ANNOTATIONS_FILE}'.")
        exit()
        
    # Validate that all images exist before training
    missing_files = [f for f in all_image_filenames if not os.path.exists(os.path.join(DATA_DIR, f))]
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} image file(s) referenced in annotations but not found in '{DATA_DIR}':")
        for f in missing_files[:10]:
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        proceed = input("\nDo you want to continue training ONLY with the available images? [y/N]: ").lower().strip() == 'y'
        if proceed:
            print("Proceeding with training, ignoring missing files...")
            # Filter out missing files from the list and the dataframe
            all_image_filenames = [f for f in all_image_filenames if f not in missing_files]
            annotations_df = annotations_df[annotations_df['filename'].isin(all_image_filenames)].copy()
        else:
            print("Training cancelled by user.")
            exit()

    dataset_size = len(all_image_filenames)
    if dataset_size == 0:
        print("Error: No valid image files remaining after filtering. Cannot proceed with training.")
        exit()

    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_filenames = [all_image_filenames[i] for i in train_indices]
    val_filenames = [all_image_filenames[i] for i in val_indices]

    train_df = annotations_df[annotations_df['filename'].isin(train_filenames)]
    val_df = annotations_df[annotations_df['filename'].isin(val_filenames)]

    print(f"\nDataset contains {dataset_size} unique images.")
    if val_size > 0:
        print(f"Splitting into {len(train_filenames)} training and {len(val_filenames)} validation images.")
    else:
        print(f"Using all {len(train_filenames)} images for training (no validation split).")

    train_dataset = CustomObjectDetectionDataset(DATA_DIR, train_df, CLASSES, get_transform(train=True))
    # OPTIMIZATION: persistent_workers=True keeps workers alive, pin_memory=True speeds up transfer
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=True, pin_memory=True
    )

    val_loader = None
    if val_size > 0:
        val_dataset = CustomObjectDetectionDataset(DATA_DIR, val_df, CLASSES, get_transform(train=False))
        # OPTIMIZATION: persistent_workers=True keeps workers alive, pin_memory=True speeds up transfer
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            persistent_workers=True, pin_memory=True
        )

    # --- Model Initialization ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = get_object_detection_model(num_classes, backbone_name=BACKBONE_NAME)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # NOTE: previously used float16 autocast + GradScaler. float16's narrow exponent
    # range is prone to overflow (-> inf -> NaN loss) on a freshly-initialized detection
    # head early in training, which is the most likely cause of majority-NaN losses.
    # bfloat16 has the same exponent range as float32, so it doesn't need loss scaling
    # and is far more numerically stable for this use case.
    AMP_DTYPE = torch.bfloat16
    amp_enabled = ENABLE_MIXED_PRECISION and device != 'cpu'

    # --- Training ---
    print("\nStarting model training...")
    history = {'train_loss': [], 'val_loss': [], 'val_map': [], 'val_map_50': [], 'val_map_75': []}
    best_val_map_50 = -1.0
    epochs_no_improve = 0
    final_epoch = 0 # To track the actual last epoch if early stopping occurs 

    if val_loader:
        map_metric = MeanAveragePrecision(iou_type="bbox").to(device)

    for epoch in range(EPOCHS):
        final_epoch = epoch
        model.train()
        train_loss_accumulator = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)", leave=False)

        num_valid_batches = 0
        for images, targets in train_pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Mixed precision training
            with autocast('cuda', dtype=AMP_DTYPE, enabled=amp_enabled):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                # Log which images and which loss component triggered this so the
                # underlying data/config issue can be tracked down, then skip the
                # batch entirely rather than letting a NaN poison the epoch average.
                bad_filenames = [
                    train_dataset.image_filenames[t['image_id'].item()] for t in targets
                ]
                bad_losses = {k: round(v.item(), 4) for k, v in loss_dict.items()}
                tqdm.write(
                    f"[WARN] Non-finite training loss {bad_losses} for images "
                    f"{bad_filenames} - skipping batch."
                )
                continue

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = losses.item()
            train_loss_accumulator += current_loss
            num_valid_batches += 1
            train_pbar.set_postfix(loss=f"{current_loss:.4f}")

        avg_train_loss = train_loss_accumulator / max(num_valid_batches, 1)
        history['train_loss'].append(avg_train_loss)
        epoch_summary = f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}"

        current_val_map = float('nan')
        avg_val_loss = float('nan')

        if val_loader:
            model.eval()
            val_loss_accumulator = 0.0
            map_metric.reset()
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)", leave=False)

            num_valid_val_batches = 0
            with torch.no_grad():
                for images_val, targets_val in val_pbar:
                    images_val = list(image.to(device) for image in images_val)
                    targets_for_metric = [{k: v.to(device) for k, v in t.items()} for t in targets_val]

                    predictions = model(images_val)
                    map_metric.update(predictions, targets_for_metric)

                    # Get validation loss
                    model.train() # Switch to train mode to get loss
                    loss_dict_val = model(images_val, targets_for_metric)
                    model.eval() # Switch back to eval mode
                    losses_val = sum(loss for loss in loss_dict_val.values())

                    if not torch.isfinite(losses_val):
                        bad_filenames = [
                            val_dataset.image_filenames[t['image_id'].item()] for t in targets_for_metric
                        ]
                        tqdm.write(f"[WARN] Non-finite validation loss for images {bad_filenames} - skipping.")
                        continue

                    current_val_loss_item = losses_val.item()
                    val_loss_accumulator += current_val_loss_item
                    num_valid_val_batches += 1
                    val_pbar.set_postfix(loss=f"{current_val_loss_item:.4f}")

            avg_val_loss = val_loss_accumulator / max(num_valid_val_batches, 1)
            map_results = map_metric.compute()
            
            # Extract all mAP metrics
            current_val_map = map_results['map'].item()
            current_val_map_50 = map_results['map_50'].item()
            current_val_map_75 = map_results['map_75'].item()
            
            history['val_loss'].append(avg_val_loss)
            history['val_map'].append(current_val_map)
            history['val_map_50'].append(current_val_map_50)
            history['val_map_75'].append(current_val_map_75)
            
            epoch_summary += f", Val Loss: {avg_val_loss:.4f}, Val mAP: {current_val_map:.4f}"
            epoch_summary += f", mAP@50: {current_val_map_50:.4f}, mAP@75: {current_val_map_75:.4f}"

            # --- Best Model Saving ---
            if current_val_map_50 > best_val_map_50:
                best_val_map_50 = current_val_map_50
                epochs_no_improve = 0
                
                # --- Save Inference-Only Model ---
                best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'backbone_name': BACKBONE_NAME,
                    'classes': CLASSES,
                    'num_classes': num_classes
                }, best_model_path)

                epoch_summary += " -> New Best Model Saved!"
            else:
                epochs_no_improve += 1
        else:
            history['val_loss'].append(float('nan'))
            history['val_map'].append(float('nan'))
            history['val_map_50'].append(float('nan'))
            history['val_map_75'].append(float('nan'))
        
        print(epoch_summary)
        lr_scheduler.step()

        # --- Checkpoint Saving ---
        if CHECKPOINT_SAVE_INTERVAL > 0 and (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path = os.path.join(MODEL_DIR, checkpoint_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'backbone_name': BACKBONE_NAME,
                'classes': CLASSES,
                'num_classes': num_classes
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # --- Early Stopping ---
        if val_loader and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print("\nTraining complete.")
    if val_size > 0:
        print(f"Highest validation mAP@50 was {best_val_map_50:.4f}.")
        print(f"The best performing model was saved to '{os.path.join(MODEL_DIR, 'best_model.pth')}'")

    final_model_filename = f'final_model_epoch_{final_epoch+1}.pth'
    final_model_path = os.path.join(MODEL_DIR, final_model_filename)
    
    # Save the complete final model
    torch.save({
        'epoch': final_epoch + 1,
        'model_state_dict': model.state_dict(),
        'backbone_name': BACKBONE_NAME,
        'classes': CLASSES,
        'num_classes': num_classes
    }, final_model_path)
    
    print(f"\nFinal model from last epoch ({final_epoch+1}) saved to: {final_model_path}")

    history_filename = 'training_history.npy'
    history_path = os.path.join(MODEL_DIR, history_filename)
    np.save(history_path, history)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if val_loader:
            plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation mAP metrics
        if val_loader:
            plt.subplot(2, 1, 2)
            plt.plot(history['val_map'], label='mAP')
            plt.plot(history['val_map_50'], label='mAP@50')
            plt.plot(history['val_map_75'], label='mAP@75')
            plt.title('Validation mAP Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(MODEL_DIR, 'training_curves.png')
        plt.savefig(plot_path)
        print(f"Training curves saved to: {plot_path}")
        
    except ImportError:
        print("Note: matplotlib not available. Install it to generate training curve plots.")