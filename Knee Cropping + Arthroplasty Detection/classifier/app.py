"""
Browser app: upload a knee image (or a whole folder) and get metal/regular
predictions.

    conda activate kneelabel
    python app.py

Then open http://127.0.0.1:5050 in your browser (it opens automatically).
"""

import gpu_env  # noqa: F401 - must be imported before torch, see gpu_env.py

import io
import webbrowser
from threading import Timer

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from inference import KneeClassifier

PORT = 5050

app = Flask(__name__)
classifier = KneeClassifier()

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knee Classifier</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; min-height: 100vh; display: flex; flex-direction: column; align-items: center;
    padding: 48px 16px 80px; background: #14161a; color: #eef0f3;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .subtitle { color: #9aa2b1; font-size: 14px; margin-bottom: 28px; text-align: center; }
  .card {
    background: #1d2026; border: 1px solid #2c303a; border-radius: 14px;
    padding: 28px; width: 100%; max-width: 480px; text-align: center;
  }
  .card + .card { margin-top: 28px; }
  .card.wide { max-width: 720px; text-align: left; }
  .card h2 { font-size: 15px; margin: 0 0 16px; text-align: center; }
  #drop {
    border: 2px dashed #2c303a; border-radius: 12px; padding: 32px 16px;
    cursor: pointer; transition: border-color .15s ease;
  }
  #drop.hover { border-color: #4f8fff; }
  #preview { max-width: 100%; max-height: 320px; border-radius: 8px; display: none; margin-top: 16px; }
  input[type=file] { display: none; }
  .result { margin-top: 22px; display: none; }
  .verdict { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
  .verdict.metal { color: #4f8fff; }
  .verdict.regular { color: #ff9f43; }
  .prob { color: #9aa2b1; font-size: 14px; }
  .bar-track { height: 8px; background: #2c303a; border-radius: 999px; overflow: hidden; margin-top: 14px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #ff9f43, #4f8fff); width: 50%; }
  .spinner { margin-top: 18px; color: #9aa2b1; font-size: 14px; display: none; text-align: center; }
  .error { color: #ffb4b4; margin-top: 14px; font-size: 13px; display: none; text-align: center; }
  .modelinfo { color: #5c6270; font-size: 12px; margin-top: 22px; }

  .btn {
    font: inherit; cursor: pointer; border: none; border-radius: 9px;
    padding: 11px 20px; font-weight: 600; font-size: 14px; background: #2a2e37;
    color: #eef0f3; width: 100%;
  }
  .btn:hover { filter: brightness(1.2); }

  .folder-progress { display: none; margin-top: 18px; }
  .progress-wrap { display: flex; align-items: center; gap: 12px; }
  .progress-track { flex: 1; height: 6px; background: #2c303a; border-radius: 999px; overflow: hidden; }
  .progress-fill { height: 100%; background: #35c26a; width: 0%; transition: width .15s ease; }
  .progress-label { font-size: 12px; color: #9aa2b1; white-space: nowrap; font-variant-numeric: tabular-nums; }

  .folder-summary {
    display: none; margin-top: 18px; font-size: 13px; color: #9aa2b1;
    display: none;
  }
  .folder-summary b { color: #eef0f3; }

  .results-list {
    display: none; margin-top: 16px; max-height: 480px; overflow-y: auto;
    border: 1px solid #2c303a; border-radius: 10px;
  }
  .result-row {
    display: flex; align-items: center; gap: 12px; padding: 8px 12px;
    border-bottom: 1px solid #2c303a;
  }
  .result-row:last-child { border-bottom: none; }
  .result-row img {
    width: 44px; height: 44px; object-fit: cover; border-radius: 6px; flex-shrink: 0;
    background: #000;
  }
  .result-row .fname {
    flex: 1; min-width: 0; font-size: 13px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
  }
  .result-row .badge {
    font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px; flex-shrink: 0;
  }
  .result-row .badge.metal { background: rgba(79,143,255,0.15); color: #4f8fff; }
  .result-row .badge.regular { background: rgba(255,159,67,0.15); color: #ff9f43; }
  .result-row .badge.err { background: rgba(255,90,90,0.15); color: #ff5a5a; }
  .result-row .pct {
    font-size: 12px; color: #9aa2b1; width: 48px; text-align: right; flex-shrink: 0;
    font-variant-numeric: tabular-nums;
  }
  .result-row { cursor: pointer; }
  .result-row:hover { background: #22252c; }

  .lightbox {
    display: none; position: fixed; inset: 0; background: rgba(10,11,14,0.92);
    z-index: 100; align-items: center; justify-content: center; flex-direction: column;
    padding: 32px 16px;
  }
  .lightbox.open { display: flex; }
  .lightbox img {
    max-width: min(90vw, 640px); max-height: 55vh; object-fit: contain;
    border-radius: 10px; background: #000;
  }
  .lightbox .lb-fname { margin-top: 16px; font-size: 13px; color: #9aa2b1; text-align: center; word-break: break-all; }
  .lightbox .lb-verdict { font-size: 24px; font-weight: 700; margin-top: 8px; }
  .lightbox .lb-verdict.metal { color: #4f8fff; }
  .lightbox .lb-verdict.regular { color: #ff9f43; }
  .lightbox .lb-verdict.err { color: #ff5a5a; }
  .lightbox .lb-prob { color: #9aa2b1; font-size: 13px; margin-top: 4px; }
  .lightbox .lb-counter { color: #5c6270; font-size: 12px; margin-top: 18px; }
  .lightbox .lb-close {
    position: absolute; top: 20px; right: 28px; font-size: 28px; color: #9aa2b1;
    cursor: pointer; background: none; border: none; line-height: 1;
  }
  .lightbox .lb-close:hover { color: #eef0f3; }
  .lightbox .lb-nav {
    position: absolute; top: 50%; transform: translateY(-50%); background: #1d2026;
    border: 1px solid #2c303a; color: #eef0f3; font-size: 22px; width: 44px; height: 44px;
    border-radius: 999px; cursor: pointer;
  }
  .lightbox .lb-nav:hover { background: #2a2e37; }
  .lightbox .lb-prev { left: 16px; }
  .lightbox .lb-next { right: 16px; }
  .lightbox .lb-hint { color: #5c6270; font-size: 12px; margin-top: 6px; }
</style>
</head>
<body>
  <h1>Knee Classifier</h1>
  <div class="subtitle">Upload a single knee X-ray crop, or analyze a whole folder of them</div>

  <div class="card">
    <h2>Single image</h2>
    <div id="drop">
      <div>Click to choose an image, or drag one here</div>
      <img id="preview">
    </div>
    <input type="file" id="fileInput" accept="image/*">

    <div class="spinner" id="spinner">Analyzing…</div>
    <div class="error" id="error"></div>

    <div class="result" id="result">
      <div class="verdict" id="verdict"></div>
      <div class="prob" id="prob"></div>
      <div class="bar-track"><div class="bar-fill" id="barFill"></div></div>
    </div>
  </div>

  <div class="card wide">
    <h2 style="text-align:left">Analyze a folder</h2>
    <button class="btn" id="folderBtn">Choose a folder…</button>
    <input type="file" id="folderInput" webkitdirectory directory multiple>

    <div class="folder-progress" id="folderProgress">
      <div class="progress-wrap">
        <div class="progress-track"><div class="progress-fill" id="folderProgressFill"></div></div>
        <div class="progress-label" id="folderProgressLabel">0 / 0</div>
      </div>
    </div>
    <div class="error" id="folderError"></div>
    <div class="folder-summary" id="folderSummary"></div>
    <div class="results-list" id="resultsList"></div>
  </div>

  <div class="modelinfo" id="modelinfo"></div>

  <div class="lightbox" id="lightbox">
    <button class="lb-close" id="lbClose">&times;</button>
    <button class="lb-nav lb-prev" id="lbPrev">&#8249;</button>
    <img id="lbImg">
    <div class="lb-verdict" id="lbVerdict"></div>
    <div class="lb-prob" id="lbProb"></div>
    <div class="lb-fname" id="lbFname"></div>
    <div class="lb-counter" id="lbCounter"></div>
    <div class="lb-hint">&larr; &rarr; to navigate &middot; Esc to close</div>
    <button class="lb-nav lb-next" id="lbNext">&#8250;</button>
  </div>

<script>
  // ---------- single image ----------
  const drop = document.getElementById('drop');
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const spinner = document.getElementById('spinner');
  const errorBox = document.getElementById('error');
  const result = document.getElementById('result');
  const verdict = document.getElementById('verdict');
  const prob = document.getElementById('prob');
  const barFill = document.getElementById('barFill');

  drop.addEventListener('click', () => fileInput.click());
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('hover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  async function handleFile(file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    result.style.display = 'none';
    errorBox.style.display = 'none';
    spinner.style.display = 'block';

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      spinner.style.display = 'none';
      if (!res.ok) throw new Error(data.error || 'Prediction failed');

      verdict.textContent = data.label === 'metal' ? 'Metal replacement' : 'Regular knee';
      verdict.className = 'verdict ' + data.label;
      prob.textContent = `${(data.probability_metal * 100).toFixed(1)}% probability of metal`;
      barFill.style.width = `${(data.probability_metal * 100).toFixed(1)}%`;
      result.style.display = 'block';
    } catch (err) {
      spinner.style.display = 'none';
      errorBox.textContent = err.message;
      errorBox.style.display = 'block';
    }
  }

  // ---------- folder batch analysis ----------
  const IMAGE_EXT = /\\.(jpe?g|png|bmp|tiff?|webp)$/i;
  const BATCH_SIZE = 32;

  const folderBtn = document.getElementById('folderBtn');
  const folderInput = document.getElementById('folderInput');
  const folderProgress = document.getElementById('folderProgress');
  const folderProgressFill = document.getElementById('folderProgressFill');
  const folderProgressLabel = document.getElementById('folderProgressLabel');
  const folderError = document.getElementById('folderError');
  const folderSummary = document.getElementById('folderSummary');
  const resultsList = document.getElementById('resultsList');

  folderBtn.addEventListener('click', () => folderInput.click());
  folderInput.addEventListener('change', () => {
    if (folderInput.files.length) analyzeFolder(Array.from(folderInput.files));
  });

  async function analyzeFolder(allFiles) {
    const files = allFiles.filter(f => IMAGE_EXT.test(f.name));

    resultsList.innerHTML = '';
    resultsList.style.display = 'none';
    folderSummary.style.display = 'none';
    folderError.style.display = 'none';

    if (!files.length) {
      folderError.textContent = 'No image files found in that folder.';
      folderError.style.display = 'block';
      return;
    }

    folderProgress.style.display = 'block';
    resultsList.style.display = 'block';
    folderProgressLabel.textContent = `0 / ${files.length}`;
    folderProgressFill.style.width = '0%';

    let metalCount = 0, regularCount = 0, errorCount = 0, done = 0;
    analyzed = [];

    for (let i = 0; i < files.length; i += BATCH_SIZE) {
      const batch = files.slice(i, i + BATCH_SIZE);
      const formData = new FormData();
      for (const f of batch) {
        formData.append('images', f, f.webkitRelativePath || f.name);
      }

      let batchResults;
      try {
        const res = await fetch('/predict_batch', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Batch prediction failed');
        batchResults = data.results;
      } catch (err) {
        folderError.textContent = err.message;
        folderError.style.display = 'block';
        return;
      }

      for (let j = 0; j < batch.length; j++) {
        const file = batch[j];
        const r = batchResults[j];
        const index = analyzed.length;
        analyzed.push({ file, result: r });

        const row = document.createElement('div');
        row.className = 'result-row';
        row.addEventListener('click', () => openLightbox(index));

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        row.appendChild(img);

        const fname = document.createElement('div');
        fname.className = 'fname';
        fname.textContent = file.webkitRelativePath || file.name;
        row.appendChild(fname);

        if (r.error) {
          errorCount++;
          const badge = document.createElement('div');
          badge.className = 'badge err';
          badge.textContent = 'error';
          row.appendChild(badge);
          row.title = r.error;
        } else {
          if (r.label === 'metal') metalCount++; else regularCount++;
          const badge = document.createElement('div');
          badge.className = 'badge ' + r.label;
          badge.textContent = r.label;
          row.appendChild(badge);

          const pct = document.createElement('div');
          pct.className = 'pct';
          pct.textContent = `${(r.probability_metal * 100).toFixed(0)}%`;
          row.appendChild(pct);
        }

        resultsList.appendChild(row);
      }

      done += batch.length;
      folderProgressLabel.textContent = `${done} / ${files.length}`;
      folderProgressFill.style.width = `${(done / files.length) * 100}%`;
    }

    folderSummary.style.display = 'block';
    folderSummary.innerHTML = `<b>${metalCount}</b> metal &middot; <b>${regularCount}</b> regular` +
      (errorCount ? ` &middot; <b>${errorCount}</b> failed to read` : '') +
      ` (of ${files.length} images)`;
  }

  // ---------- lightbox (arrow-key navigation through analyzed images) ----------
  let analyzed = [];
  let lightboxIndex = -1;

  const lightbox = document.getElementById('lightbox');
  const lbImg = document.getElementById('lbImg');
  const lbVerdict = document.getElementById('lbVerdict');
  const lbProb = document.getElementById('lbProb');
  const lbFname = document.getElementById('lbFname');
  const lbCounter = document.getElementById('lbCounter');

  function openLightbox(index) {
    lightboxIndex = index;
    renderLightbox();
    lightbox.classList.add('open');
  }

  function closeLightbox() {
    lightbox.classList.remove('open');
    lightboxIndex = -1;
  }

  function stepLightbox(delta) {
    if (lightboxIndex < 0 || !analyzed.length) return;
    lightboxIndex = (lightboxIndex + delta + analyzed.length) % analyzed.length;
    renderLightbox();
  }

  function renderLightbox() {
    const { file, result: r } = analyzed[lightboxIndex];
    lbImg.src = URL.createObjectURL(file);
    lbFname.textContent = file.webkitRelativePath || file.name;
    lbCounter.textContent = `${lightboxIndex + 1} / ${analyzed.length}`;

    if (r.error) {
      lbVerdict.textContent = 'Failed to read image';
      lbVerdict.className = 'lb-verdict err';
      lbProb.textContent = r.error;
    } else {
      lbVerdict.textContent = r.label === 'metal' ? 'Metal replacement' : 'Regular knee';
      lbVerdict.className = 'lb-verdict ' + r.label;
      lbProb.textContent = `${(r.probability_metal * 100).toFixed(1)}% probability of metal`;
    }
  }

  document.getElementById('lbClose').addEventListener('click', closeLightbox);
  document.getElementById('lbPrev').addEventListener('click', () => stepLightbox(-1));
  document.getElementById('lbNext').addEventListener('click', () => stepLightbox(1));
  lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });

  document.addEventListener('keydown', (e) => {
    if (!lightbox.classList.contains('open')) return;
    if (e.key === 'ArrowRight') stepLightbox(1);
    else if (e.key === 'ArrowLeft') stepLightbox(-1);
    else if (e.key === 'Escape') closeLightbox();
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read that as an image"}), 400

    label, prob_metal = classifier.predict_image(img)
    return jsonify({"label": label, "probability_metal": prob_metal})


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    for file in files:
        try:
            img = Image.open(io.BytesIO(file.read()))
            label, prob_metal = classifier.predict_image(img)
            results.append({"filename": file.filename, "label": label, "probability_metal": prob_metal})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"results": results})


def _open_browser():
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    print(f"Model loaded (epoch {classifier.checkpoint_info['epoch']}, "
          f"val_auc={classifier.checkpoint_info['val_auc']:.4f})")
    Timer(1.0, _open_browser).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
