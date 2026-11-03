"""
pbn_core.py -- PatchBridgeNet shared pipeline (used by the Colab notebook AND the local Streamlit app)

Pipeline (identical to the training notebook):
    RGB image -> resize 224x224 -> ImageNet normalize
    -> original + 4 vertical patches (224x56 -> resized to 224x224)
    -> MobileNetV2 / DarkNet53 / DenseNet201 (GAP + FC features)
    -> sum of the 5 views (flip TTA: each view = (normal + flipped) / 2)
    -> per-model INCA (MinMax + selected indices) -> concatenate
    -> Chi2 indices -> StandardScaler -> quadratic SVM (one-vs-one)

The trained classifier is stored as a plain dictionary of NumPy arrays (no scikit-learn objects),
so the pickle files load on any machine regardless of the scikit-learn version.

Also provided here:
    * calibrated probabilities (percentages) computed from SVM decision values
    * Grad-CAM through the full pipeline (CNN feature maps -> INCA -> Chi2 -> SVM)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from PIL import Image

IMAGE_SIZE = 224
N_PATCHES = 4

CLASS7_NAMES = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
CLASS7_FULL = {
    "AMD": "Age-related Macular Degeneration",
    "DME": "Diabetic Macular Edema",
    "ERM": "Epiretinal Membrane",
    "NO": "Normal",
    "RAO": "Retinal Artery Occlusion",
    "RVO": "Retinal Vein Occlusion",
    "VID": "Vitreomacular Interface Disease",
}
CLASS2_NAMES = ["Normal", "Diseased"]  # 0 = Normal, 1 = Diseased
MODEL_ORDER = ["mobilenetv2", "darknet53", "densenet201"]


# =============================================================================
# Preprocessing and patches
# =============================================================================
def get_transform(resize_mode="bilinear"):
    interp = (transforms.InterpolationMode.BICUBIC if resize_mode == "bicubic"
              else transforms.InterpolationMode.BILINEAR)
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(path):
    return Image.open(path).convert("RGB")


def make_patches(img_tensor, n_patches=N_PATCHES):
    width = img_tensor.shape[-1]
    patch_w = width // n_patches
    return [img_tensor[:, :, i * patch_w:(i + 1) * patch_w] for i in range(n_patches)]


def resize_patch(patch_tensor, resize_mode="bilinear"):
    p = patch_tensor.unsqueeze(0)
    mode = "bicubic" if resize_mode == "bicubic" else "bilinear"
    p = nn.functional.interpolate(p, size=(224, 224), mode=mode, align_corners=False)
    return p.squeeze(0)


def build_views(pil_img, resize_mode="bilinear"):
    """Return a (5, 3, 224, 224) tensor: [original, patch1, patch2, patch3, patch4]."""
    tensor = get_transform(resize_mode)(pil_img.convert("RGB"))
    patches = [resize_patch(p, resize_mode) for p in make_patches(tensor)]
    return torch.stack([tensor] + patches)


# =============================================================================
# Feature extractor (GAP + FC)
# =============================================================================
class FeatureExtractor(nn.Module):
    """
    mobilenetv2 -> GAP=1280, FC=1000
    darknet53   -> GAP=1024, FC=1000  (timm)
    densenet201 -> GAP=1920, FC=1000
    pretrained=False is used only when a saved state_dict is loaded afterwards (offline use).
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name

        if model_name == "mobilenetv2":
            base = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = base.features
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = base.classifier
        elif model_name == "darknet53":
            self.backbone = timm.create_model("darknet53", pretrained=pretrained)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif model_name == "densenet201":
            base = models.densenet201(
                weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = base.features
            self.fc_layer = base.classifier
        else:
            raise ValueError(model_name)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def feature_map(self, x):
        """Last convolutional feature map (used by Grad-CAM)."""
        if self.model_name == "darknet53":
            return self.backbone.forward_features(x)
        return self.features(x)

    def head(self, fm):
        """GAP + FC features computed from a feature map -> (B, D) tensor."""
        b = fm.size(0)
        if self.model_name == "mobilenetv2":
            gap_feat = self.gap(fm).reshape(b, -1)
            fc_feat = self.fc(gap_feat)
        elif self.model_name == "darknet53":
            gap_feat = self.gap(fm).reshape(b, -1)
            fc_feat = self.backbone.forward_head(fm)
        else:  # densenet201
            g = nn.functional.relu(fm, inplace=False)
            g = nn.functional.adaptive_avg_pool2d(g, (1, 1))
            gap_feat = g.reshape(b, -1)
            fc_feat = self.fc_layer(gap_feat)
        return torch.cat([gap_feat, fc_feat], dim=1)

    def forward(self, x):
        return self.head(self.feature_map(x)).cpu().numpy()


def load_extractors(device, weights_path=None):
    """Load the three CNNs. If weights_path exists the saved weights are used (offline),
    otherwise the ImageNet pretrained weights are downloaded."""
    state = None
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    extractors = {}
    for name in MODEL_ORDER:
        ext = FeatureExtractor(name, pretrained=(state is None))
        if state is not None:
            ext.load_state_dict(state[name])
        extractors[name] = ext.to(device).eval()
    return extractors


def extract_image_features(pil_img, extractors, device, resize_mode="bilinear", use_flip_tta=True):
    """
    Features of ONE image, identical to one row of the training feature matrices.
    Returns [F_mobilenet (1, 2280), F_darknet (1, 2024), F_densenet (1, 2920)].
    """
    batch = build_views(pil_img, resize_mode)
    feats = []
    for name in MODEL_ORDER:
        ext = extractors[name]
        with torch.no_grad():
            v = ext(batch.to(device)).astype(np.float32)
            if use_flip_tta:
                vf = ext(torch.flip(batch, dims=[3]).to(device)).astype(np.float32)
                v = 0.5 * (v + vf)
        feats.append(v.sum(axis=0, keepdims=True).astype(np.float32))
    return feats


# =============================================================================
# Classifier bundle (pure NumPy): INCA -> Chi2 -> StandardScaler -> quadratic SVM (OvO)
# =============================================================================
def svm_pair_decisions(x, sv):
    """
    Pairwise decision values of a libsvm / scikit-learn SVC, re-implemented in NumPy.
    Multi-class: column p corresponds to the pair (i, j), i < j, positive value -> class i.
    Binary: the single column follows the scikit-learn convention, positive -> classes[1].
    """
    k = len(sv["classes"])
    ns = np.asarray(sv["n_support"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(ns)])
    kern = (sv["gamma"] * (x @ sv["support_vectors"].T) + sv["coef0"]) ** sv["degree"]
    dec = np.empty((x.shape[0], k * (k - 1) // 2), dtype=np.float64)
    p = 0
    for i in range(k):
        for j in range(i + 1, k):
            si, sj = starts[i], starts[j]
            dec[:, p] = (kern[:, si:si + ns[i]] @ sv["dual_coef"][j - 1, si:si + ns[i]]
                         + kern[:, sj:sj + ns[j]] @ sv["dual_coef"][i, sj:sj + ns[j]]
                         + sv["intercept"][p])
            p += 1
    return dec


def bundle_features(bundle, F_list):
    """
    INCA (MinMax + indices) -> concat -> Chi2 -> StandardScaler.
    Same arithmetic as scikit-learn on float32 features (in-place float32 results), then float64 like SVC.
    """
    def f32(a):
        return np.asarray(a, dtype=np.float64).astype(np.float32)

    parts = []
    for F, d in zip(F_list, bundle["inca"]):
        x = np.asarray(F, dtype=np.float32)[:, d["idx"]]
        x = f32(x.astype(np.float64) * d["scale"])
        x = f32(x.astype(np.float64) + d["min"])
        parts.append(x)
    x = np.concatenate(parts, axis=1)[:, bundle["chi_idx"]]
    x = f32(x.astype(np.float64) - bundle["std_mean"])
    x = f32(x.astype(np.float64) / bundle["std_scale"])
    return x.astype(np.float64)


def bundle_decisions(bundle, F_list):
    return svm_pair_decisions(bundle_features(bundle, F_list), bundle["svm"])


def votes_from_decisions(dec, k):
    n = dec.shape[0]
    votes = np.zeros((n, k))
    if k == 2:
        win1 = dec[:, 0] >= 0          # scikit-learn sign; a tie goes to class 1, as in libsvm
        votes[:, 1] = win1
        votes[:, 0] = ~win1
        return votes
    p = 0
    for i in range(k):
        for j in range(i + 1, k):
            win_i = dec[:, p] > 0
            votes[:, i] += win_i
            votes[:, j] += ~win_i
            p += 1
    return votes


def bundle_predict(bundle, F_list):
    """Returns (predicted class values, pairwise decisions, votes)."""
    dec = bundle_decisions(bundle, F_list)
    classes = np.asarray(bundle["svm"]["classes"])
    votes = votes_from_decisions(dec, len(classes))
    pred = classes[np.argmax(votes, axis=1)]  # ties -> lowest class index (same as libsvm)
    return pred, dec, votes


def calibration_features(dec, k):
    """Inputs of the probability calibrator: binary -> decision value; multi-class -> decisions + votes."""
    if k == 2:
        return dec
    return np.hstack([dec, votes_from_decisions(dec, k)])


def bundle_probabilities(bundle, dec):
    """Calibrated probabilities (rows sum to 1) or None when no calibrator is stored."""
    cal = bundle.get("calibrator")
    if cal is None:
        return None
    feats = calibration_features(dec, len(bundle["svm"]["classes"]))
    z = feats @ np.asarray(cal["coef"]).T + np.asarray(cal["intercept"])
    if cal["kind"] == "sigmoid":
        p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_from_features(b7, b2, F_list):
    """Run both models on the features of one image (or a batch)."""
    pred7, dec7, votes7 = bundle_predict(b7, F_list)
    pred2, dec2, _ = bundle_predict(b2, F_list)
    return {
        "pred7": pred7, "dec7": dec7, "votes7": votes7, "probs7": bundle_probabilities(b7, dec7),
        "pred2": pred2, "dec2": dec2[:, 0], "probs2": bundle_probabilities(b2, dec2),
    }


# =============================================================================
# Grad-CAM through the full pipeline
# =============================================================================
class TorchHead:
    """Differentiable copy of INCA -> Chi2 -> StandardScaler -> SVM, built from a bundle."""

    def __init__(self, bundle, device):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
        li = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        self.inca = [(li(d["idx"]), t(d["scale"]), t(d["min"])) for d in bundle["inca"]]
        self.chi = li(bundle["chi_idx"])
        self.mean, self.std = t(bundle["std_mean"]), t(bundle["std_scale"])
        sv = bundle["svm"]
        self.sv, self.dual, self.b = t(sv["support_vectors"]), t(sv["dual_coef"]), t(sv["intercept"])
        self.gamma, self.coef0, self.degree = float(sv["gamma"]), float(sv["coef0"]), int(sv["degree"])
        self.ns = [int(v) for v in sv["n_support"]]
        self.starts = [0]
        for v in self.ns:
            self.starts.append(self.starts[-1] + v)
        self.k = len(sv["classes"])

    def decisions(self, F_list):
        parts = [F.to(torch.float64)[:, idx] * sc + mn for F, (idx, sc, mn) in zip(F_list, self.inca)]
        x = torch.cat(parts, dim=1)[:, self.chi]
        x = (x - self.mean) / self.std
        kern = (self.gamma * (x @ self.sv.T) + self.coef0) ** self.degree
        cols, p = [], 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                si, sj = self.starts[i], self.starts[j]
                d = ((kern[:, si:si + self.ns[i]] * self.dual[j - 1, si:si + self.ns[i]]).sum(1)
                     + (kern[:, sj:sj + self.ns[j]] * self.dual[i, sj:sj + self.ns[j]]).sum(1)
                     + self.b[p])
                cols.append(d)
                p += 1
        return torch.stack(cols, dim=1)[0]

    def class_margin(self, F_list, c):
        """Differentiable evidence for class c: sum of its pairwise margins against all other classes."""
        dec = self.decisions(F_list)
        if self.k == 2:
            return dec[0] if c == 1 else -dec[0]
        total, p = 0.0, 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if i == c:
                    total = total + dec[p]
                elif j == c:
                    total = total - dec[p]
                p += 1
        return total


def gradcam(pil_img, extractors, bundle, class_idx, device, resize_mode="bilinear", use_flip_tta=True):
    """
    Grad-CAM for the given class of one bundle (7-class or binary).
    The gradient of the class evidence (SVM margins) is propagated back through
    Chi2 / INCA / feature fusion to the last convolutional maps of the three CNNs
    for all 5 views (full image + 4 patches), then mapped back onto the image.

    Returns dict: combined, full, patches (all float32 arrays 224x224 scaled to [0, 1]).
    """
    head = TorchHead(bundle, device)
    batch = build_views(pil_img, resize_mode).to(device)
    variants = [batch] + ([torch.flip(batch, dims=[3])] if use_flip_tta else [])
    weight = 1.0 / len(variants)

    all_maps, sums = [], []
    with torch.enable_grad():
        for name in MODEL_ORDER:
            ext = extractors[name]
            acc, model_maps = None, []
            for xb in variants:
                with torch.no_grad():
                    fm = ext.feature_map(xb)
                fm = fm.detach().requires_grad_(True)
                part = weight * ext.head(fm).sum(dim=0, keepdim=True)
                acc = part if acc is None else acc + part
                model_maps.append(fm)
            all_maps.append(model_maps)
            sums.append(acc)

        score = head.class_margin(sums, int(class_idx))
        flat = [fm for mm in all_maps for fm in mm]
        grads = torch.autograd.grad(score, flat)

    gi, per_model = 0, []
    for model_maps in all_maps:
        per_variant = []
        for vi, fm in enumerate(model_maps):
            g = grads[gi]
            gi += 1
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * fm.detach()).sum(dim=1, keepdim=True))        # (5, 1, h, w)
            cam = TF.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
            if vi == 1:                                                         # flipped variant -> flip back
                cam = torch.flip(cam, dims=[2])
            per_variant.append(cam)
        cam_m = torch.stack(per_variant).mean(dim=0)                            # (5, 224, 224)
        mx = cam_m.flatten(1).max(dim=1)[0].clamp_min(1e-8)
        per_model.append(cam_m / mx[:, None, None])
    cams = torch.stack(per_model).mean(dim=0)                                   # (5, 224, 224)

    full = cams[0]
    mosaic = torch.zeros(224, 224, device=cams.device)
    pw = 224 // N_PATCHES
    for i in range(N_PATCHES):                                                  # patch maps back to strips
        strip = TF.interpolate(cams[i + 1][None, None], size=(224, pw), mode="bilinear",
                               align_corners=False)[0, 0]
        mosaic[:, i * pw:(i + 1) * pw] = strip

    def norm(a):
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        a = np.asarray(a, dtype=np.float32)
        m = a.max()
        return a / m if m > 1e-8 else a

    combined = norm(0.5 * (norm(full) + norm(mosaic)))
    return {"combined": combined, "full": norm(full), "patches": norm(mosaic)}


# =============================================================================
# Visualisation helpers (NumPy / PIL only)
# =============================================================================
def resize_cam(cam, size):
    """Resize a 224x224 map to (width, height) = size, values stay in [0, 1]."""
    img = Image.fromarray(np.clip(np.asarray(cam, dtype=np.float32), 0, 1), mode="F")
    return np.clip(np.asarray(img.resize(tuple(size), Image.BICUBIC), dtype=np.float32), 0, 1)


def overlay_heatmap(pil_img, cam, max_alpha=0.65):
    """Blend a jet heatmap over the image; stronger regions are more opaque. Returns uint8 RGB."""
    import matplotlib
    base = pil_img.convert("RGB")
    c = resize_cam(cam, base.size)
    heat = matplotlib.colormaps["jet"](c)[..., :3]
    a = (max_alpha * c)[..., None]
    out = np.asarray(base).astype(np.float32) / 255.0 * (1 - a) + heat * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def describe_region(cam, threshold=0.6):
    """Summarise where the strongest evidence is (relative position, size, bounding box)."""
    cam = np.asarray(cam, dtype=np.float32)
    mask = cam >= threshold
    if not mask.any():
        return {"found": False, "text": "No strongly highlighted region."}
    ys, xs = np.nonzero(mask)
    h, w = cam.shape
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = "upper" if cy < 1 / 3 else ("central" if cy < 2 / 3 else "lower")
    horiz = "left" if cx < 1 / 3 else ("middle" if cx < 2 / 3 else "right")
    place = "central" if (vert == "central" and horiz == "middle") else f"{vert}-{horiz}"
    box = (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
    area = float(mask.mean() * 100)
    return {"found": True, "place": place, "area_pct": area, "box": box,
            "text": f"Strongest evidence is in the {place} part of the scan "
                    f"(highlighted area covers about {area:.1f}% of the image)."}
