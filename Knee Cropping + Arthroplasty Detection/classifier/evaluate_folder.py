"""
Run the trained classifier over images and report statistics.

    python evaluate_folder.py

With no arguments, this runs immediately against classifier/test_set/ (the
held-out test split materialized to disk by build_test_folder.py - run that
first, or whenever the Metal/Regular folders change and you want a fresh
test set). Falls back to the in-memory --split test if test_set/ doesn't
exist yet.

Three modes:

0. --split test (recommended) - evaluates on the exact held-out test set
   dataset.py/train.py carved out (same RANDOM_SEED, so it's deterministic
   and reproducible). This is the only mode that's a true held-out
   evaluation - the model never saw these images during training.

       python evaluate_folder.py --split test
       python evaluate_folder.py --split val    # or val/train

1. --folder pointing at a labeled folder - the target folder directly
   contains "Metal"/"metal" and "Regular"/"regular" subfolders (the same
   convention image-sorter.html produces). Ground truth is taken from which
   subfolder a file is in. Prints accuracy/precision/recall/F1/confusion
   matrix/ROC-AUC. NOTE: if this folder is (or overlaps with) the Metal/
   Regular folders used for training, these numbers are NOT a fair test -
   the model has seen most of these images already. Use --split test for
   an honest number.

       python evaluate_folder.py --folder ".."

2. --folder pointing at an unlabeled folder - a flat folder of images with
   no such subfolders. Just runs predictions, no accuracy stats (there's no
   reliable ground truth to compare against - filenames are NOT trusted as
   labels here, since that convention already turned out to be unreliable
   once for this project, see dataset.py).

       python evaluate_folder.py --folder "some/unlabeled/folder"

Either way, writes per-image predictions to a CSV next to this script.
"""

import gpu_env  # noqa: F401 - must be imported before torch, see gpu_env.py

import argparse
import csv
import html
import webbrowser
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import dataset as dataset_module
from inference import KneeClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_SUBFOLDER_NAMES = {"metal": 1, "regular": 0}


def find_label_subfolders(folder: Path):
    """Returns {label_idx: subfolder_path} if the folder has metal/regular
    subfolders (case-insensitive), else None."""
    found = {}
    for child in folder.iterdir():
        if child.is_dir() and child.name.lower() in LABEL_SUBFOLDER_NAMES:
            found[LABEL_SUBFOLDER_NAMES[child.name.lower()]] = child
    return found if len(found) == 2 else None


def collect_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def build_misclassified_gallery(out_name: str, wrong_rows: list) -> Path:
    """Writes a self-contained dark-themed HTML gallery of misclassified images
    (using file:// links to the originals, so no copying/embedding needed) and
    returns its path."""
    items_html = []
    for r in wrong_rows:
        file_uri = Path(r["abs_path"]).resolve().as_uri()
        items_html.append(f"""
        <div class="item">
          <img src="{file_uri}" loading="lazy">
          <div class="fname">{html.escape(r["filename"])}</div>
          <div class="line">actual: <b class="{html.escape(r['ground_truth_label'])}">{html.escape(r["ground_truth_label"])}</b>
            &nbsp;predicted: <b class="{html.escape(r['predicted_label'])}">{html.escape(r["predicted_label"])}</b></div>
          <div class="line">probability metal: {html.escape(r["probability_metal"])}</div>
        </div>""")

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Misclassified - {html.escape(out_name)}</title>
<style>
  body {{ margin:0; background:#14161a; color:#eef0f3; font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; padding:32px 24px; }}
  h1 {{ font-size:18px; margin:0 0 20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:18px; }}
  .item {{ background:#1d2026; border:1px solid #2c303a; border-radius:12px; padding:12px; }}
  .item img {{ width:100%; height:180px; object-fit:contain; background:#000; border-radius:8px; }}
  .fname {{ font-size:12px; color:#9aa2b1; margin-top:8px; word-break:break-all; }}
  .line {{ font-size:12px; margin-top:4px; }}
  b.metal {{ color:#4f8fff; }}
  b.regular {{ color:#ff9f43; }}
</style></head>
<body>
  <h1>{len(wrong_rows)} misclassified - {html.escape(out_name)}</h1>
  <div class="grid">{''.join(items_html)}</div>
</body></html>"""

    out_path = Path(f"{out_name}_misclassified.html").resolve()
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def write_results(out_name: str, rows: list, y_true: list, y_pred: list, y_prob: list):
    out_csv = Path(f"{out_name}_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "predicted_label", "probability_metal", "ground_truth_label"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote per-image predictions to {out_csv}")

    n_metal_pred = sum(r["predicted_label"] == "metal" for r in rows)
    print(f"\nPredicted: {n_metal_pred} metal / {len(rows) - n_metal_pred} regular "
          f"(of {len(rows)} images)")

    if y_true:
        print("\nConfusion matrix (rows=actual, cols=predicted) [regular, metal]:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=["regular", "metal"], digits=3))
        if len(set(y_true)) > 1:
            print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

        wrong_rows = [r for r in rows if r["ground_truth_label"] and r["predicted_label"] != r["ground_truth_label"]]
        if wrong_rows:
            gallery_path = build_misclassified_gallery(out_name, wrong_rows)
            print(f"\n{len(wrong_rows)} misclassified - opening {gallery_path}")
            webbrowser.open(gallery_path.as_uri())
        else:
            print("\nNo misclassifications - nothing to show.")


DEFAULT_TEST_FOLDER = Path(__file__).resolve().parent / "test_set"


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--folder", help="Folder of images to evaluate")
    source.add_argument("--split", choices=["train", "val", "test"],
                         help="Evaluate on dataset.py's held-out split instead of a folder "
                              "(use 'test' for an honest, never-trained-on evaluation)")
    parser.add_argument("--model", default=None, help="Path to model.pt (default: best_model/model.pt)")
    args = parser.parse_args()

    if not args.folder and not args.split:
        # No args given - run immediately against the materialized held-out test
        # folder (see build_test_folder.py) if it exists, else fall back to the
        # in-memory --split test.
        if DEFAULT_TEST_FOLDER.is_dir():
            args.folder = str(DEFAULT_TEST_FOLDER)
            print(f"No --folder/--split given - defaulting to {DEFAULT_TEST_FOLDER} "
                  f"(run build_test_folder.py to refresh it).")
        else:
            args.split = "test"
            print(f"No --folder/--split given and {DEFAULT_TEST_FOLDER} doesn't exist "
                  f"(run build_test_folder.py to create it) - defaulting to --split test.")

    classifier = KneeClassifier(Path(args.model)) if args.model else KneeClassifier()
    print(f"Loaded model (epoch {classifier.checkpoint_info['epoch']}, "
          f"val_auc={classifier.checkpoint_info['val_auc']:.4f})")

    rows = []
    y_true, y_pred, y_prob = [], [], []

    if args.split:
        all_samples = dataset_module.load_all_samples()
        train_s, val_s, test_s = dataset_module.split_samples(all_samples)
        split_samples_list = {"train": train_s, "val": val_s, "test": test_s}[args.split]
        print(f"Evaluating on the '{args.split}' split: {len(split_samples_list)} images "
              f"({sum(s.label == 1 for s in split_samples_list)} metal / "
              f"{sum(s.label == 0 for s in split_samples_list)} regular)")

        out_name = f"split_{args.split}"
        for sample in split_samples_list:
            label, prob_metal = classifier.predict_path(sample.path)
            rows.append({
                "filename": str(sample.path.relative_to(dataset_module.PROJECT_ROOT)),
                "predicted_label": label,
                "probability_metal": f"{prob_metal:.4f}",
                "ground_truth_label": classifier.class_names[sample.label],
                "abs_path": str(sample.path),
            })
            y_true.append(sample.label)
            y_pred.append(1 if label == "metal" else 0)
            y_prob.append(prob_metal)

        write_results(out_name, rows, y_true, y_pred, y_prob)
        return

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    label_dirs = find_label_subfolders(folder)

    if label_dirs:
        print(f"Found labeled subfolders: {[(k, v.name) for k, v in label_dirs.items()]}")
        for true_label_idx, subdir in label_dirs.items():
            for path in collect_images(subdir):
                label, prob_metal = classifier.predict_path(path)
                rows.append({
                    "filename": f"{subdir.name}/{path.name}",
                    "predicted_label": label,
                    "probability_metal": f"{prob_metal:.4f}",
                    "ground_truth_label": classifier.class_names[true_label_idx],
                    "abs_path": str(path),
                })
                y_true.append(true_label_idx)
                y_pred.append(1 if label == "metal" else 0)
                y_prob.append(prob_metal)
    else:
        image_paths = collect_images(folder)
        if not image_paths:
            raise SystemExit(f"No images (and no metal/regular subfolders) found in {folder}")
        print(f"No metal/regular subfolders found - running predictions only on "
              f"{len(image_paths)} images in {folder}")
        for path in image_paths:
            label, prob_metal = classifier.predict_path(path)
            rows.append({
                "filename": path.name,
                "predicted_label": label,
                "probability_metal": f"{prob_metal:.4f}",
                "ground_truth_label": "",
            })

    write_results(folder.name, rows, y_true, y_pred, y_prob)


if __name__ == "__main__":
    main()
