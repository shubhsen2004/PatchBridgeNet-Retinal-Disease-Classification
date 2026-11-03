"""
app.py -- OCT Retina AI Analyzer (PatchBridgeNet) - Streamlit GUI

Run:
    streamlit run app.py

Folder layout (all files come from the Colab notebook's deployment.zip):
    app.py
    pbn_core.py
    deployment/
        retina_7class.pkl
        retina_binary.pkl
        cnn_weights.pt          (optional; pretrained weights are downloaded if missing)
        metrics.json            (cross-validation accuracy, used by the Model Performance tab)
        training_info.json      (optional)
        confusion_matrix_7class.png / confusion_matrix_binary.png   (optional)
"""
import os
# Windows + PyTorch: two OpenMP runtimes (torch and numpy/MKL) get loaded and the process is killed
# without a Python error ("OMP: Error #15"). This must be set before torch / numpy are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(4, (os.cpu_count() or 2) // 2))))

import io
import json
import time
import pickle
import hashlib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from pbn_core import (CLASS7_NAMES, CLASS7_FULL, CLASS2_NAMES, load_extractors,
                      extract_image_features, predict_from_features, gradcam,
                      overlay_heatmap, describe_region, resize_cam)

DEPLOY_DIR = Path(__file__).parent / "deployment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))   # fewer threads = less RAM, no thread oversubscription
torch.set_grad_enabled(False)                              # Grad-CAM turns gradients on only where needed

PAPER_ACC = {"7class": 92.30, "binary": 97.40}
PAPER_F1_7 = {"AMD": 96.98, "DME": 81.11, "ERM": 83.00, "NO": 91.25,
              "RAO": 81.00, "RVO": 74.46, "VID": 85.29}
PAPER_BINARY = {"Sensitivity": 98.44, "Specificity": 91.86, "Precision": 98.44,
                "F1-score": 98.44, "Geometric Mean": 95.09}

st.set_page_config(page_title="OCT Retina AI Analyzer", page_icon="👁️", layout="wide")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (the first run takes a little while)...")
def load_models():
    with open(DEPLOY_DIR / "retina_7class.pkl", "rb") as f:
        b7 = pickle.load(f)
    with open(DEPLOY_DIR / "retina_binary.pkl", "rb") as f:
        b2 = pickle.load(f)
    extractors = load_extractors(DEVICE, str(DEPLOY_DIR / "cnn_weights.pt"))
    return b7, b2, extractors


def to_rgb(img):
    """PIL image -> 8-bit RGB (handles 16-bit / 32-bit grayscale PNG files too)."""
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        a = np.asarray(img, dtype=np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255.0
        img = Image.fromarray(a.astype(np.uint8))
    return img.convert("RGB")


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def load_json(name):
    p = DEPLOY_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_prediction(img, b7, b2, extractors, settings):
    s = settings
    t0 = time.time()
    feats = extract_image_features(img, extractors, DEVICE, s["RESIZE_MODE"], s["USE_FLIP_TTA"])
    t_feat = time.time() - t0
    out = predict_from_features(b7, b2, feats)
    return {
        "pred7": int(out["pred7"][0]),
        "probs7": None if out["probs7"] is None else out["probs7"][0],
        "dec7": out["dec7"][0],
        "pred2": int(out["pred2"][0]),
        "probs2": None if out["probs2"] is None else out["probs2"][0],
        "dec2": float(out["dec2"][0]),
        "t_feat": t_feat,
        "t_total": time.time() - t0,
    }


def prob_chart(names, probs, pred_name):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    order = np.argsort(probs)
    colors = ["#d9534f" if names[i] == pred_name else "#9aa5b1" for i in order]
    bars = ax.barh([names[i] for i in order], [probs[i] * 100 for i in order], color=colors)
    for bar, i in zip(bars, order):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{probs[i] * 100:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Probability (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def cam_figure(img, cam, threshold):
    w, h = img.size
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[1].imshow(overlay_heatmap(img, cam))
    axes[1].set_title("Grad-CAM heatmap (red = strongest evidence)")
    axes[2].imshow(img)
    c = resize_cam(cam, (w, h))
    region = describe_region(cam, threshold)
    if c.max() >= threshold:
        axes[2].contour(c, levels=[threshold], colors="lime", linewidths=2)
    if region["found"]:
        x0, y0, x1, y1 = region["box"]
        axes[2].add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                                    fill=False, edgecolor="yellow", linewidth=2, linestyle="--"))
    axes[2].set_title(f"Highlighted region (evidence >= {threshold:.0%} of maximum)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    return fig, region


# -----------------------------------------------------------------------------
# Sidebar and state
# -----------------------------------------------------------------------------
st.sidebar.title("👁️ OCT Retina AI Analyzer")
st.sidebar.caption("PatchBridgeNet: patch-based deep features + INCA + Chi2 + SVM")
st.sidebar.markdown(f"**Device:** `{DEVICE}`")

if not (DEPLOY_DIR / "retina_7class.pkl").exists() or not (DEPLOY_DIR / "retina_binary.pkl").exists():
    st.error(f"retina_7class.pkl / retina_binary.pkl were not found in `{DEPLOY_DIR}`. "
             "Unzip deployment.zip from the Colab notebook next to app.py.")
    st.stop()

b7, b2, extractors = load_models()
metrics = load_json("metrics.json")
info = load_json("training_info.json")
settings = {"RESIZE_MODE": "bilinear", "USE_FLIP_TTA": True, **(b7.get("settings") or {})}

if info:
    st.sidebar.markdown("**Training info**")
    st.sidebar.write(f"Images used: {info.get('n_images', '-')}")
    st.sidebar.write(f"INCA sizes (7-class): {info.get('S_7class', '-')}")
    st.sidebar.write(f"INCA sizes (binary): {info.get('S_binary', '-')}")
    st.sidebar.write(f"Chi2 features: {info.get('Smerge', '-')}")
with st.sidebar.expander("Pipeline settings"):
    st.json(settings)

for key, default in (("history", []), ("results", {}), ("cams", {})):
    if key not in st.session_state:
        st.session_state[key] = default

tab_pred, tab_perf, tab_hist, tab_about = st.tabs(
    ["🔍 Prediction", "📊 Model Performance", "🕘 History", "ℹ️ About"])

# -----------------------------------------------------------------------------
# Tab 1: prediction
# -----------------------------------------------------------------------------
with tab_pred:
    st.header("Upload an OCT retinal image")
    up = st.file_uploader("JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])

    if up is None:
        st.info("Upload a retinal OCT B-scan to start.")
    else:
        raw = up.getvalue()
        key = hashlib.md5(raw).hexdigest()
        try:
            img_orig = Image.open(io.BytesIO(raw))
            img_orig.load()
        except Exception as e:
            st.error(f"Could not read this file as an image: {e}")
            st.stop()
        fmt, mode = img_orig.format, img_orig.mode
        img = to_rgb(img_orig)
        W, H = img.size

        if key not in st.session_state.results:
            with st.spinner("Extracting features and predicting..."):
                try:
                    st.session_state.results[key] = run_prediction(img, b7, b2, extractors, settings)
                except Exception as e:
                    st.exception(e)
                    st.stop()
            r0 = st.session_state.results[key]
            has_p = r0["probs7"] is not None
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": up.name,
                "7-class prediction": CLASS7_NAMES[r0["pred7"]],
                "7-class probability (%)": round(float(r0["probs7"][r0["pred7"]]) * 100, 1) if has_p else None,
                "binary prediction": CLASS2_NAMES[r0["pred2"]],
                "P(Diseased) (%)": round(float(r0["probs2"][1]) * 100, 1) if has_p else None,
                "processing time (s)": round(r0["t_total"], 2),
            })
        r = st.session_state.results[key]
        pred7_name, pred2_name = CLASS7_NAMES[r["pred7"]], CLASS2_NAMES[r["pred2"]]
        p7, p2 = r["probs7"], r["probs2"]

        col_img, col_res = st.columns([1, 1.25])

        with col_img:
            st.subheader("Uploaded image")
            st.image(img, width="stretch")
            st.caption(f"**{up.name}** | {W} x {H} px | format {fmt} | mode {mode} | {len(raw) / 1024:.1f} KB")

        with col_res:
            st.subheader("Binary classification (Normal vs Diseased)")
            if pred2_name == "Diseased":
                st.error("Prediction: **Diseased**")
            else:
                st.success("Prediction: **Normal**")
            if p2 is not None:
                m1, m2 = st.columns(2)
                m1.metric("Probability of disease", f"{p2[1] * 100:.1f}%")
                m2.metric("Probability of normal", f"{p2[0] * 100:.1f}%")
                st.progress(float(p2[1]))
                if 0.4 <= p2[1] <= 0.6:
                    st.warning("Borderline case: the disease probability is close to 50%.")
            st.caption(f"SVM decision value: {r['dec2']:+.3f} (positive = Diseased)")

            st.subheader("7-class classification")
            st.markdown(f"Predicted class: **{pred7_name}** ({CLASS7_FULL[pred7_name]})")
            if p7 is not None:
                st.write(f"**Probability of {pred7_name}: {p7[r['pred7']] * 100:.1f}%**")
                show_fig(prob_chart(CLASS7_NAMES, p7, pred7_name))
                st.dataframe(pd.DataFrame({
                    "Class": CLASS7_NAMES,
                    "Full name": [CLASS7_FULL[c] for c in CLASS7_NAMES],
                    "Probability (%)": np.round(p7 * 100, 2)}).sort_values("Probability (%)", ascending=False),
                    hide_index=True, width="stretch")
                st.caption(f"Disease probability from the 7-class model (100% minus Normal): "
                           f"{(1 - p7[CLASS7_NAMES.index('NO')]) * 100:.1f}%")
                if int(np.argmax(p7)) != r["pred7"]:
                    st.warning("The SVM's final vote and the highest probability point to different classes. "
                               "Treat this result as borderline.")
            else:
                st.info("Probabilities are not stored in this pickle file; only the predicted class is shown.")

            if (pred7_name == "NO") != (pred2_name == "Normal"):
                st.warning("The 7-class and binary models disagree (one says Normal, the other Diseased). "
                           "Treat this result as uncertain.")

        # ---------------------------------------------------------------- Grad-CAM
        st.divider()
        st.subheader("Where is the disease? (Grad-CAM)")
        st.caption("Grad-CAM shows which parts of the scan pushed the model towards the selected class. "
                   "It is an explanation of the model, not a clinical segmentation.")

        c1, c2, c3 = st.columns([1, 1, 1])
        which = c1.radio("Explain which model?", ["7-class model", "Binary model"], horizontal=True,
                         key=f"which_{key}")
        if which == "7-class model":
            target_name = c2.selectbox("Class to explain", CLASS7_NAMES, index=r["pred7"], key=f"t7_{key}")
            bundle, class_idx = b7, CLASS7_NAMES.index(target_name)
        else:
            target_name = c2.selectbox("Class to explain", CLASS2_NAMES, index=r["pred2"], key=f"t2_{key}")
            bundle, class_idx = b2, CLASS2_NAMES.index(target_name)
        threshold = c3.slider("Highlight threshold", 0.30, 0.90, 0.60, 0.05)

        cam_key = (key, which, class_idx)
        if cam_key not in st.session_state.cams:
            with st.spinner("Computing Grad-CAM..."):
                st.session_state.cams[cam_key] = gradcam(
                    img, extractors, bundle, class_idx, DEVICE,
                    settings["RESIZE_MODE"], settings["USE_FLIP_TTA"])
        cams = st.session_state.cams[cam_key]

        fig, region = cam_figure(img, cams["combined"], threshold)
        show_fig(fig)
        st.info(f"Evidence for **{target_name}** ({which}): {region['text']}")

        with st.expander("Full-image map vs patch-level map"):
            e1, e2 = st.columns(2)
            e1.image(overlay_heatmap(img, cams["full"]), caption="Full image view", width="stretch")
            e2.image(overlay_heatmap(img, cams["patches"]), caption="Four vertical patches (stitched)",
                     width="stretch")

        # ---------------------------------------------------------------- extra info + report
        st.subheader("Additional information")
        a1, a2, a3 = st.columns(3)
        a1.metric("Image size", f"{W} x {H}")
        a2.metric("Feature extraction", f"{r['t_feat']:.2f} s")
        a3.metric("Total processing time", f"{r['t_total']:.2f} s")

        row = {
            "file": up.name, "width": W, "height": H,
            "7class_prediction": pred7_name, "7class_full_name": CLASS7_FULL[pred7_name],
            "binary_prediction": pred2_name, "binary_decision_value": r["dec2"],
            "gradcam_target": f"{which}: {target_name}", "gradcam_region": region["text"],
            "processing_time_s": round(r["t_total"], 3),
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": "Research prototype, not for clinical diagnosis",
        }
        if p7 is not None:
            row.update({f"prob_{c}_%": round(float(p) * 100, 2) for c, p in zip(CLASS7_NAMES, p7)})
            row["prob_diseased_binary_%"] = round(float(p2[1]) * 100, 2)
        st.download_button("⬇️ Download prediction report (CSV)",
                           pd.DataFrame([row]).to_csv(index=False).encode(),
                           file_name=f"report_{Path(up.name).stem}.csv", mime="text/csv")

    st.caption("⚠️ Educational / research prototype. Not for standalone medical diagnosis.")

# -----------------------------------------------------------------------------
# Tab 2: model performance (cross-validation results from the notebook)
# -----------------------------------------------------------------------------
with tab_perf:
    st.header("Model performance (leak-free 10-fold cross-validation)")
    st.caption("These are evaluation results on unseen folds, not results for the uploaded image.")

    r7 = (metrics or {}).get("7class") or b7.get("metrics")
    r2 = (metrics or {}).get("binary") or b2.get("metrics")
    if not r7 or not r2:
        st.info("No metrics found. Run the accuracy cells of the Colab notebook and copy metrics.json to deployment/.")
    else:
        k1, k2 = st.columns(2)
        k1.metric("7-class accuracy", f"{r7['Accuracy']:.2f}%", f"{r7['Accuracy'] - PAPER_ACC['7class']:+.2f}% vs paper")
        k2.metric("Binary accuracy", f"{r2['Accuracy']:.2f}%", f"{r2['Accuracy'] - PAPER_ACC['binary']:+.2f}% vs paper")

        st.subheader("Accuracy: yours vs paper")
        st.dataframe(pd.DataFrame([
            {"Model": "7-class", "Your accuracy (%)": r7["Accuracy"], "Paper (%)": PAPER_ACC["7class"],
             "Difference (%)": r7["Accuracy"] - PAPER_ACC["7class"]},
            {"Model": "Binary", "Your accuracy (%)": r2["Accuracy"], "Paper (%)": PAPER_ACC["binary"],
             "Difference (%)": r2["Accuracy"] - PAPER_ACC["binary"]},
        ]).round(2), hide_index=True, width="stretch")

        st.subheader("7-class: per-class metrics")
        rows = []
        for c in CLASS7_NAMES:
            m = r7[c]
            rows.append({"Class": c, "Sensitivity": m["Sensitivity"], "Specificity": m["Specificity"],
                         "Precision": m["Precision"], "F1-score": m["F1-score"],
                         "Geometric Mean": m["Geometric Mean"],
                         "Paper F1": PAPER_F1_7[c], "F1 difference": m["F1-score"] - PAPER_F1_7[c]})
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

        st.subheader("Binary (Diseased): metrics vs paper")
        d = r2.get("Diseased", {})
        st.dataframe(pd.DataFrame([{"Metric": k, "Yours (%)": d.get(k, np.nan), "Paper (%)": v,
                                    "Difference (%)": d.get(k, np.nan) - v}
                                   for k, v in PAPER_BINARY.items()]).round(2),
                     hide_index=True, width="stretch")

        if r7.get("Folds"):
            st.subheader("Fold-wise accuracy (7-class)")
            st.bar_chart(pd.DataFrame(r7["Folds"])["acc"])

    cal7, cal2 = b7.get("calibration_report"), b2.get("calibration_report")
    if cal7 and cal2:
        st.subheader("Reliability of the percentages")
        st.dataframe(pd.DataFrame([
            {"Model": "7-class", "Log-loss": cal7["log_loss"], "Brier": cal7["brier"],
             "ECE (%)": cal7["ece"] * 100, "Same class as SVM (%)": cal7["agreement_with_svm"]},
            {"Model": "Binary", "Log-loss": cal2["log_loss"], "Brier": cal2["brier"],
             "ECE (%)": cal2["ece"] * 100, "Same class as SVM (%)": cal2["agreement_with_svm"]},
        ]).round(3), hide_index=True, width="stretch")
        st.caption("Percentages are calibrated on out-of-fold SVM scores. Lower ECE means the percentages "
                   "match the real hit-rate more closely.")

    cm1, cm2 = st.columns(2)
    for col, name, title in [(cm1, "7class", "7-class confusion matrix"), (cm2, "binary", "Binary confusion matrix")]:
        p = DEPLOY_DIR / f"confusion_matrix_{name}.png"
        if p.exists():
            col.subheader(title)
            col.image(str(p), width="stretch")

# -----------------------------------------------------------------------------
# Tab 3: history
# -----------------------------------------------------------------------------
with tab_hist:
    st.header("Predictions of this session")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, hide_index=True, width="stretch")
        st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False).encode(),
                           file_name="prediction_history.csv", mime="text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.results = {}
            st.session_state.cams = {}
            st.rerun()
    else:
        st.info("No predictions yet.")

# -----------------------------------------------------------------------------
# Tab 4: about
# -----------------------------------------------------------------------------
with tab_about:
    st.header("How it works")
    st.markdown("""
1. **Preprocess**: RGB, 224 x 224, ImageNet normalisation.
2. **Patches**: the original image plus 4 vertical patches (224 x 56, resized to 224 x 224).
3. **Deep features**: MobileNetV2, DarkNet53 and DenseNet201 (GAP + FC), sum of the 5 views, horizontal-flip averaging.
4. **INCA** per network, fusion, **Chi2** selection.
5. **Quadratic SVM** (one-vs-one) for the 7-class and the binary (Normal / Diseased) task.
6. **Percentages**: SVM scores are converted to probabilities with a calibrator fitted on out-of-fold cross-validation scores.
7. **Grad-CAM**: the gradient of the selected class evidence is propagated through the SVM, Chi2, INCA and fusion steps
   back to the last convolutional maps of the three networks, for the full image and each patch.

**Please note**
- The SVM always returns a class, even for images that are not retinal OCT scans. Upload OCT B-scans only.
- Accuracy depends on how similar the image is to the training data (OCTDL). Images from very different scanners can be less accurate.
- This is an educational / research prototype and must not be used as a standalone medical diagnosis.
""")
