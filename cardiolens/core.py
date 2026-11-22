"""CardioLens model (frozen ImageNet ResNet18 + new head), Grad-CAM and the attention audit."""
import json

import numpy as np
from PIL import Image

import resnet

F32 = np.float32


class CardioLens:
    """Transfer learning: ImageNet ResNet18 backbone (frozen) + a replaced classification head."""

    def __init__(self, backbone, W, b, meta):
        self.backbone, self.W, self.b, self.meta = backbone, W.astype(F32), b.astype(F32), meta

    def features(self, gray, batch=16):
        g = np.asarray(gray)
        if g.ndim == 2:
            g = g[None]
        return np.concatenate([self.backbone.features(resnet.prep_gray(g[i:i + batch]))
                               for i in range(0, len(g), batch)])

    def probs(self, A):
        return softmax(A.mean((-2, -1)) @ self.W + self.b)

    def cam(self, A, target, size=(224, 224)):
        # Grad-CAM: d logit_c / dA_k = W[k, c] / (h*w) for a GAP + linear head, so alpha_k is exact
        alpha = self.W[:, target] / (A.shape[-1] * A.shape[-2])
        cam = np.maximum((alpha[:, None, None] * A).sum(0), 0).astype(F32)
        cam = np.asarray(Image.fromarray(cam, mode="F").resize(size[::-1], Image.BILINEAR))
        return cam / (cam.max() + 1e-8)

    def save(self, path):
        np.savez_compressed(path, W=self.W, b=self.b, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path, backbone):
        d = np.load(path, allow_pickle=False)
        return cls(backbone, d["W"], d["b"], json.loads(str(d["meta"])))


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def gradcam(model, gray, target=None, A=None):
    """Return (class probabilities, explained class, heatmap in [0, 1] at the image size)."""
    A = model.features(gray)[0] if A is None else A
    probs = model.probs(A)
    target = int(probs.argmax()) if target is None else int(target)
    return probs, target, model.cam(A, target, gray.shape)


def _dilate(mask, r):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def outline(mask, r=1):
    return _dilate(mask, r) & ~mask


def ellipse_mask(cx, cy, a, b, size=224):
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1


def audit(cam, landmark, tolerance_px=12):
    """Compare a heatmap with a clinical landmark mask; returns metrics and a verdict.

    The layer4 heatmap is a 7x7 grid (32px cells), so the peak counts as on-landmark within
    `tolerance_px`, and alignment is judged by lift: attention share / landmark area share.
    """
    landmark = landmark.astype(bool)
    energy = float((cam * landmark).sum() / (cam.sum() + 1e-8))
    chance = float(landmark.mean())
    lift = energy / max(chance, 1e-6)
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    hit = bool(_dilate(landmark, tolerance_px)[peak])

    if hit and lift >= 2:
        verdict, level = "Aligned: the model looked where a cardiologist looks", "ok"
    elif hit or lift >= 1.5:
        verdict, level = "Partially aligned: attention is split, review before trusting", "warn"
    else:
        verdict, level = "Misaligned: possible shortcut, the model may be right for the wrong reason", "bad"
    return {"energy": energy, "chance": chance, "lift": lift, "hit": hit, "peak": peak,
            "verdict": verdict, "level": level}


def describe_peak(gray, peak, hit, landmark_name):
    y, x = peak
    if hit:
        return f"on the {landmark_name.lower()}"
    if x < 48 and y > 176:
        return "the burned-in scanner annotation (bottom-left corner)"
    if (gray[max(0, y - 12):y + 12, max(0, x - 12):x + 12] > 25).mean() < 0.2:
        return "black background outside the scan sector"
    return ("the ventricles / apex" if y < 120 else "the lower atria and valve plane, away from the septum")


def jet(x):
    x = np.clip(x, 0, 1)[..., None]
    return np.concatenate([np.clip(1.5 - np.abs(4 * x - k), 0, 1) for k in (3, 2, 1)], -1)


def overlay(gray, cam, alpha=0.5, contour=None, contour_rgb=(255, 183, 3)):
    base = np.repeat(gray[..., None], 3, -1).astype(F32) / 255
    a = (alpha * np.clip((cam - 0.25) / 0.45, 0, 1))[..., None]
    out = ((base * (1 - a) + jet(cam) * a) * 255).astype(np.uint8)
    if contour is not None:
        out[outline(contour)] = contour_rgb
    return out
