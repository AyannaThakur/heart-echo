import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import core
import resnet
import train

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "models", "cardiolens_asd.npz")
AMBER, RED, GREEN, MUTED = "#FFB703", "#E63946", "#22C55E", "#9AA5B1"
DEFAULT_LM = {"cx": 110, "cy": 172, "a": 20, "b": 28}

st.set_page_config(page_title="CardioLens", page_icon=":material/ecg_heart:", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1250px;}}
.cl-card {{background:#1B2230; border:1px solid #232C3D; border-radius:14px; padding:18px 20px; margin-bottom:12px;}}
.cl-kicker {{color:{MUTED}; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:4px;}}
.cl-pred {{font-size:1.6rem; font-weight:700; line-height:1.2;}}
.cl-bar {{height:8px; border-radius:4px; background:#232C3D; margin:4px 0 10px;}}
.cl-bar > div {{height:8px; border-radius:4px;}}
.cl-row {{display:flex; justify-content:space-between; font-size:.9rem; gap:8px;}}
.cl-verdict {{border-radius:12px; padding:14px 18px; font-weight:600; font-size:1.05rem; margin-bottom:10px;}}
.cl-ok {{background:rgba(34,197,94,.12); border:1px solid {GREEN}; color:#BBF7D0;}}
.cl-warn {{background:rgba(255,183,3,.10); border:1px solid {AMBER}; color:#FDE68A;}}
.cl-bad {{background:rgba(230,57,70,.12); border:1px solid {RED}; color:#FECACA;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def backbone():
    return resnet.ResNet18()


@st.cache_resource
def load_model():
    return core.CardioLens.load(MODEL_PATH, backbone())


@st.cache_resource
def dataset_index():
    """patient -> {split, label, frames: [(frame_no, path)]}, plus cached layer4 features by path."""
    pats, feats = {}, {}
    for split in ("train", "validation", "test"):
        items = train.list_split(DATA, split)
        cache = os.path.join(HERE, ".cache", f"{split}.npz")
        A = np.load(cache)["A"] if os.path.exists(cache) else None
        for i, (f, c, pid) in enumerate(items):
            p = pats.setdefault(pid, {"split": split, "label": c, "frames": []})
            p["frames"].append((int(os.path.basename(f).rsplit("-", 1)[1][:-4]), f))
            if A is not None and len(A) == len(items):
                feats[f] = A[i]
    for p in pats.values():
        p["frames"].sort()
    return pats, feats


@st.cache_data(max_entries=128, show_spinner="Running ResNet18...")
def compute_features(gray):
    return backbone().features(resnet.prep_gray(gray))[0]


def features_for(gray, path=None):
    feats = dataset_index()[1]
    if path in feats:
        return feats[path].astype(np.float32)
    return compute_features(gray)


def load_landmarks():
    p = os.path.join(HERE, "landmarks.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_upload(f):
    im = ImageOps.exif_transpose(Image.open(f)).convert("L")
    return np.asarray(ImageOps.fit(im, (224, 224), Image.BILINEAR), np.uint8)


def bars(probs, names, highlight):
    html = ""
    for i, (c, p) in enumerate(zip(names, probs)):
        col = RED if i == highlight else MUTED
        html += (f'<div class="cl-row"><span>{c}</span><span>{p * 100:.1f}%</span></div>'
                 f'<div class="cl-bar"><div style="width:{p * 100:.1f}%; background:{col}"></div></div>')
    return html


if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

model = load_model()
meta = model.meta
names = meta["display"]
pats, _ = dataset_index()
landmarks = load_landmarks()
SPLIT_TAG = {"test": "held-out test", "validation": "seen in training", "train": "seen in training"}

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### CardioLens")
    st.caption("Looking at where the AI looks.")
    st.markdown("#### Model card")
    rows = ""
    for split, label in (("test", "Test (2 unseen patients)"), ("cv", "Leave-one-patient-out CV")):
        r = meta.get(split)
        if r:
            rows += f'<div class="cl-row"><span>{label}</span><b>{r["frame_acc"] * 100:.0f}%</b></div>'
    pat_rows = "".join(
        f'<div class="cl-row"><span>{pid} <span style="color:{MUTED}">({v["truth"]})</span></span>'
        f'<b>{v["frame_acc"] * 100:.0f}%</b></div>'
        for split in ("test",) for pid, v in meta.get(split, {}).get("patients", {}).items())
    st.markdown(f"""
<div class="cl-card" style="padding:12px 14px">
<div class="cl-kicker">Frame accuracy</div>{rows}
<div class="cl-kicker" style="margin-top:12px">Test patients (never seen)</div>{pat_rows}
</div>""", unsafe_allow_html=True)
    st.caption(f"Transfer learning: {meta['backbone']}; {meta['head']}. Trained on {meta['train_frames']} frames "
               f"from {len(meta['train_patients'])} patients ({', '.join(meta['train_patients'])}). "
               f"Grad-CAM on layer4. Landmark: {meta['landmark']}.")
    st.warning("Only 10 patient videos in the dataset (5 ASD, 5 normal). Treat accuracy as a demo, "
               "not a clinical result.", icon=":material/info:")
    st.caption("Research prototype. It assists a doctor, it never replaces one. Not for diagnostic use.")

# ---------------- header ----------------
st.markdown("# CardioLens")
st.markdown("Atrial septal defect vs normal from an apical four-chamber echo frame. The classifier "
            "**shows its work**, and the audit checks whether it looked at the interatrial septum, "
            "where a cardiologist looks for an ASD.")

# ---------------- 1. input ----------------
st.subheader("1 · Choose an echo frame")
source = st.radio("Source", ["Dataset patients", "Upload your own"], horizontal=True, label_visibility="collapsed")

path, truth, pid = None, None, None
if source == "Dataset patients":
    order = sorted(pats, key=lambda p: ({"test": 0, "validation": 1, "train": 2}[pats[p]["split"]], p))
    c1, c2 = st.columns([1, 2])
    pid = c1.selectbox("Patient video", order,
                       format_func=lambda p: f"{p} · {train.DISPLAY[train.CLASSES[pats[p]['label']]]} · "
                                             f"{SPLIT_TAG[pats[p]['split']]}")
    frames = pats[pid]["frames"]
    k = c2.slider(f"Frame ({len(frames)} in this video)", 0, len(frames) - 1, 0)
    path = frames[k][1]
    gray = train.load_gray(path)
    truth = pats[pid]["label"]
    lm0 = landmarks.get(pid, DEFAULT_LM)
    if pats[pid]["split"] != "test":
        st.caption("This patient was used for training, so the prediction is not a fair test. "
                   "Pick ASD3 or NORM5 for unseen patients.")
else:
    up = st.file_uploader("Echo frame (PNG / JPG, apical four-chamber view)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload a frame to continue, or switch back to the dataset patients.")
        st.stop()
    gray = load_upload(up)
    lm0 = DEFAULT_LM

with st.expander("Clinical landmark: interatrial septum region (adjust to the scan)"):
    key = pid or "upload"
    c1, c2, c3, c4 = st.columns(4)
    lm = {"cx": c1.slider("Centre x", 0, 224, lm0["cx"], key=f"cx{key}"),
          "cy": c2.slider("Centre y", 0, 224, lm0["cy"], key=f"cy{key}"),
          "a": c3.slider("Half-width", 5, 110, lm0["a"], key=f"a{key}"),
          "b": c4.slider("Half-height", 5, 110, lm0["b"], key=f"b{key}")}
    st.caption("Approximate regions placed by eye per patient video for this demo; a cardiologist's "
               "annotation would replace them.")
landmark = core.ellipse_mask(**lm)

# ---------------- 2. prediction + heatmap ----------------
st.subheader("2 · Prediction and model attention")
A = features_for(gray, path)
probs = model.probs(A)
pred = int(probs.argmax())

ctl1, ctl2, ctl3, ctl4 = st.columns([1, 1.2, 1.2, 1.6])
show_overlay = ctl1.toggle("Heatmap overlay", value=True)
show_landmark = ctl2.toggle("Show landmark outline", value=True)
alpha = ctl3.slider("Overlay strength", 0.1, 1.0, 0.55, 0.05, disabled=not show_overlay)
explain = ctl4.selectbox("Explain which class", ["Predicted class"] + names,
                         help="Grad-CAM for the other class answers 'what would make it say that?'")
target = pred if explain == "Predicted class" else names.index(explain)
cam = model.cam(A, target)

col_img, col_cam, col_pred = st.columns([1, 1, 0.9])
with col_img:
    st.markdown('<div class="cl-kicker">Echo frame</div>', unsafe_allow_html=True)
    raw = np.repeat(gray[..., None], 3, -1).copy()
    if show_landmark:
        raw[core.outline(landmark)] = (255, 183, 3)
    st.image(raw, width="stretch")
with col_cam:
    st.markdown(f'<div class="cl-kicker">Grad-CAM · why "{names[target]}"</div>', unsafe_allow_html=True)
    st.image(core.overlay(gray, cam, alpha if show_overlay else 0.0, landmark if show_landmark else None),
             width="stretch")
    st.caption("Red = regions that pushed the decision most. Amber outline = interatrial septum.")
with col_pred:
    extra = ""
    if truth is not None:
        ok = pred == truth
        extra = (f'<div style="margin-top:6px;color:{GREEN if ok else RED}">{"Correct" if ok else "Incorrect"}'
                 f' · truth: {names[truth]}</div>')
    st.markdown(f"""
<div class="cl-card"><div class="cl-kicker">Predicted</div>
<div class="cl-pred">{names[pred]}</div>
<div style="color:{MUTED}">Confidence {probs[pred] * 100:.1f}%</div>{extra}</div>
<div class="cl-card"><div class="cl-kicker">All classes</div>{bars(probs, names, pred)}</div>
""", unsafe_allow_html=True)

# ---------------- 3. attention audit ----------------
st.subheader("3 · Attention audit")
au = core.audit(cam, landmark)
st.markdown(f'<div class="cl-verdict cl-{au["level"]}">{au["verdict"]}</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Attention on septum region", f"{au['energy'] * 100:.0f}%",
          f"{(au['energy'] - au['chance']) * 100:+.0f} pts vs chance")
m2.metric("Lift over chance", f"{au['lift']:.1f}x", help="Attention share in the landmark / landmark's share of the frame")
m3.metric("Peak attention", "On septum" if au["hit"] else "Off septum")
st.markdown(f"Peak attention landed on **{core.describe_peak(gray, au['peak'], au['hit'], meta['landmark'])}**. "
            f"By chance alone the septum region would get {au['chance'] * 100:.1f}% of the attention.")
if truth is not None:
    right, reason = pred == truth, au["level"] == "ok"
    quad = {(True, True): ("Right, for the right reason", GREEN),
            (True, False): ("Right, but possibly for the wrong reason", AMBER),
            (False, True): ("Wrong, though it looked in the right place: a hard case", AMBER),
            (False, False): ("Wrong, and looking in the wrong place", RED)}[(right, reason)]
    st.markdown(f'<div class="cl-card"><div class="cl-kicker">Outcome</div>'
                f'<span style="color:{quad[1]};font-weight:600">{quad[0]}</span></div>', unsafe_allow_html=True)

# ---------------- 4. whole-video audit ----------------
if source == "Dataset patients":
    st.subheader("4 · Whole-video audit")
    st.caption("Agreement becomes a measurable result, not a claim: the same audit over every frame of this patient.")
    rows, cams = [], []
    for fno, fpath in frames:
        g = train.load_gray(fpath)
        Ai = features_for(g, fpath)
        p = model.probs(Ai)
        pr = int(p.argmax())
        c = model.cam(Ai, pr)
        a = core.audit(c, landmark)
        cams.append(c)
        rows.append({"frame": fno, "predicted": names[pr], "correct": pr == truth, "p_ASD": float(p[0]),
                     "septum_attention": a["energy"], "lift": a["lift"], "peak_on_septum": a["hit"],
                     "aligned": a["level"] == "ok"})
    df = pd.DataFrame(rows)
    v1, v2 = st.columns([1, 2])
    with v1:
        st.markdown('<div class="cl-kicker">Mean heatmap over the video</div>', unsafe_allow_html=True)
        mean_cam = np.mean(cams, 0)
        st.image(core.overlay(gray, mean_cam / (mean_cam.max() + 1e-8), 0.6, landmark), width="stretch")
    with v2:
        vp = int(df.p_ASD.mean() < 0.5)
        r1, r2, r3 = st.columns(3)
        r1.metric("Video-level prediction", train.CLASSES[vp], "correct" if vp == truth else "incorrect",
                  delta_color="normal" if vp == truth else "inverse")
        r2.metric("Frames correct", f"{df.correct.mean() * 100:.0f}%")
        r3.metric("Mean lift on septum", f"{df.lift.mean():.1f}x")
        r4, r5, r6 = st.columns(3)
        r4.metric("Frames with peak on septum", f"{df.peak_on_septum.mean() * 100:.0f}%")
        r5.metric("Right for the right reason", f"{(df.correct & df.aligned).sum()} / {len(df)}")
        r6.metric("Right, wrong reason", f"{(df.correct & ~df.aligned).sum()} / {len(df)}")
        st.line_chart(df.set_index("frame")[["p_ASD"]], height=160, color=RED)
    with st.expander("Per-frame table"):
        st.dataframe(df.style.format({"p_ASD": "{:.0%}", "septum_attention": "{:.0%}", "lift": "{:.1f}x"}),
                     width="stretch", hide_index=True)
