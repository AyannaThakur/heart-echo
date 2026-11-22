"""Transfer-learn CardioLens on the echo dataset: frozen ImageNet ResNet18 + new ASD-vs-NORM head.

  python train.py                       # uses data/{train,validation,test}/{ASD,NORM}
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
from PIL import Image

import core
import resnet

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["ASD", "NORM"]
DISPLAY = {"ASD": "Atrial septal defect", "NORM": "Normal"}


def patient_of(path):
    m = re.match(r"([A-Za-z]+\d+)-", os.path.basename(path))
    return m.group(1) if m else "unknown"


def list_split(data, split):
    items = []
    for ci, c in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(data, split, c, "*.png"))):
            items.append((f, ci, patient_of(f)))
    return items


def load_gray(path):
    im = Image.open(path).convert("L")
    if im.size != (224, 224):
        im = im.resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.uint8)


def split_features(model, data, split, cache_dir):
    items = list_split(data, split)
    cache = os.path.join(cache_dir, f"{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if list(d["files"]) == [os.path.basename(f) for f, _, _ in items]:
            return items, d["A"]
    A = []
    for i in range(0, len(items), 32):
        A.append(model.features(np.stack([load_gray(f) for f, _, _ in items[i:i + 32]])))
        print(f"  {split}: {min(i + 32, len(items))}/{len(items)}", end="\r")
    A = np.concatenate(A).astype(np.float16)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, A=A, files=np.array([os.path.basename(f) for f, _, _ in items]))
    print()
    return items, A


def fit_head(G, y, n_classes, wd, steps=800, lr=0.02):
    """Softmax regression on standardised pooled features, folded back to raw-feature weights."""
    mu, sd = G.mean(0), G.std(0) + 1e-6
    Z = (G - mu) / sd
    W, b = np.zeros((G.shape[1], n_classes)), np.zeros(n_classes)
    mW = vW = mb = vb = 0
    Y = np.eye(n_classes)[y]
    cw = len(y) / (n_classes * np.bincount(y, minlength=n_classes))
    sw = cw[y][:, None]
    for t in range(1, steps + 1):
        P = core.softmax(Z @ W + b)
        gW = Z.T @ ((P - Y) * sw) / len(y) + wd * W
        gb = ((P - Y) * sw).mean(0)
        mW, vW = 0.9 * mW + 0.1 * gW, 0.999 * vW + 0.001 * gW ** 2
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb ** 2
        W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return (W / sd[:, None]).astype(np.float32), (b - (mu / sd) @ W).astype(np.float32)


def evaluate(model, items, A, landmarks=None):
    y = np.array([c for _, c, _ in items])
    P = np.stack([model.probs(a.astype(np.float32)) for a in A])
    pred = P.argmax(1)
    res = {"frames": len(y), "frame_acc": float((pred == y).mean()), "patients": {}}
    for pid in sorted({p for _, _, p in items}):
        idx = [i for i, (_, _, p) in enumerate(items) if p == pid]
        mean_p = P[idx].mean(0)
        res["patients"][pid] = {"truth": CLASSES[y[idx[0]]], "frames": len(idx),
                                "frame_acc": float((pred[idx] == y[idx]).mean()),
                                "video_pred": CLASSES[int(mean_p.argmax())], "p_asd": float(mean_p[0])}
        if landmarks and pid in landmarks:
            mask = core.ellipse_mask(**landmarks[pid])
            au = [core.audit(model.cam(A[i].astype(np.float32), int(pred[i])), mask) for i in idx]
            res["patients"][pid]["landmark_energy"] = float(np.mean([a["energy"] for a in au]))
            res["patients"][pid]["peak_on_landmark"] = float(np.mean([a["hit"] for a in au]))
    res["video_acc"] = float(np.mean([v["video_pred"] == v["truth"] for v in res["patients"].values()]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "cardiolens_asd.npz"))
    ap.add_argument("--cache", default=os.path.join(HERE, ".cache"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    lm_path = os.path.join(HERE, "landmarks.json")
    landmarks = json.load(open(lm_path)) if os.path.exists(lm_path) else {}
    model = core.CardioLens(resnet.ResNet18(), np.zeros((512, 2)), np.zeros(2), {})
    splits = {s: split_features(model, args.data, s, args.cache) for s in ("train", "validation", "test")}
    print(f"features ready in {time.time() - t0:.0f}s")

    # one validation patient per class is too few to tune on, so train on train + validation
    # and estimate generalisation with leave-one-patient-out cross-validation instead
    items = splits["train"][0] + splits["validation"][0]
    A = np.concatenate([splits["train"][1], splits["validation"][1]])
    G = A.astype(np.float32).mean((2, 3))
    y = np.array([c for _, c, _ in items])
    pid = np.array([p for _, _, p in items])
    wd = 1.0

    cv = {"patients": {}}
    for p in sorted(set(pid)):
        tr = pid != p
        model.W, model.b = fit_head(G[tr], y[tr], len(CLASSES), wd)
        acc = float((core.softmax(G[~tr] @ model.W + model.b).argmax(1) == y[~tr]).mean())
        cv["patients"][p] = {"truth": CLASSES[y[~tr][0]], "frame_acc": acc}
        print(f"  CV hold out {p}: {acc:.2f}")
    cv["frame_acc"] = float(np.mean([v["frame_acc"] for v in cv["patients"].values()]))

    model.W, model.b = fit_head(G, y, len(CLASSES), wd)

    meta = {
        "name": "CardioLens - ASD vs Normal", "classes": CLASSES, "display": [DISPLAY[c] for c in CLASSES],
        "backbone": "ResNet18, ImageNet weights (torchvision), frozen",
        "head": f"linear, class-balanced softmax, weight decay {wd}",
        "train_frames": int(len(y)), "train_patients": sorted({p for _, _, p in items}),
        "landmark": "Interatrial septum", "cv": cv,
    }
    print(f"leave-one-patient-out CV frame acc {cv['frame_acc']:.3f}")
    for s in ("train", "validation", "test"):
        meta[s] = evaluate(model, *splits[s], landmarks)
        r = meta[s]
        print(f"{s:<11} frame acc {r['frame_acc']:.3f}  video acc {r['video_acc']:.2f}  "
              + "  ".join(f"{k}:{v['frame_acc']:.2f}" for k, v in r["patients"].items()))
    model.meta = meta
    model.save(args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
