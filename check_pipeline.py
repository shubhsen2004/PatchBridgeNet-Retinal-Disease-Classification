"""
check_pipeline.py -- run the GUI pipeline once without Streamlit, to see the real error / timing.

    python check_pipeline.py path\\to\\image.jpg
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(4, (os.cpu_count() or 2) // 2))))

import sys
import time
import pickle
import faulthandler
from pathlib import Path

faulthandler.enable()          # prints a traceback even on a hard crash
import torch
from PIL import Image
from pbn_core import (CLASS7_NAMES, CLASS2_NAMES, load_extractors, extract_image_features,
                      predict_from_features, gradcam)

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
D = Path(__file__).parent / "deployment"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev, "| threads", torch.get_num_threads())

t = time.time()
b7 = pickle.load(open(D / "retina_7class.pkl", "rb"))
b2 = pickle.load(open(D / "retina_binary.pkl", "rb"))
ext = load_extractors(dev, str(D / "cnn_weights.pt"))
print(f"models loaded in {time.time() - t:.1f} s")

img = Image.open(sys.argv[1]).convert("RGB")
s = b7.get("settings", {})
t = time.time()
feats = extract_image_features(img, ext, dev, s.get("RESIZE_MODE", "bilinear"), s.get("USE_FLIP_TTA", True))
print(f"features in {time.time() - t:.1f} s:", [f.shape for f in feats])

out = predict_from_features(b7, b2, feats)
c7 = int(out["pred7"][0])
print("7-class:", CLASS7_NAMES[c7], "| binary:", CLASS2_NAMES[int(out["pred2"][0])])
if out["probs7"] is not None:
    print("probabilities:", {n: round(float(p) * 100, 1) for n, p in zip(CLASS7_NAMES, out["probs7"][0])})

t = time.time()
cam = gradcam(img, ext, b7, c7, dev, s.get("RESIZE_MODE", "bilinear"), s.get("USE_FLIP_TTA", True))
print(f"Grad-CAM in {time.time() - t:.1f} s, map {cam['combined'].shape}")
print("OK")
