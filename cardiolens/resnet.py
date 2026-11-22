"""torchvision ResNet18 (ImageNet weights) re-implemented in NumPy for inference.

PyTorch's DLLs are blocked by Windows Smart App Control on this machine, so the official
.pth checkpoint is read with a restricted unpickler and run with NumPy matmuls.
"""
import collections
import os
import pickle
import zipfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

F32 = np.float32
MEAN = np.array([0.485, 0.456, 0.406], F32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], F32).reshape(1, 3, 1, 1)
PTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "resnet18-f37072fd.pth")
NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "resnet18_imagenet.npz")


class _StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


_STORAGES = {"FloatStorage": np.float32, "LongStorage": np.int64, "HalfStorage": np.float16}


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    if not size:
        return storage[offset].copy()
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=[s * item for s in stride]).copy()


class _Unpickler(pickle.Unpickler):
    """Only allows the handful of globals a plain state_dict needs."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_: data
        if module == "torch" and name in _STORAGES:
            return _StorageType(_STORAGES[name])
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        _, st, key, _loc, _n = pid
        return np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=st.dtype)


def load_pth(path=PTH):
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        with zf.open(pkl) as f:
            return dict(_Unpickler(f, zf, prefix).load())


def _fold(sd, conv, bn, eps=1e-5):
    W = sd[conv + ".weight"].astype(F32)
    s = sd[bn + ".weight"] / np.sqrt(sd[bn + ".running_var"] + eps)
    return (W * s[:, None, None, None]).astype(F32), (sd[bn + ".bias"] - sd[bn + ".running_mean"] * s).astype(F32)


def convert(pth=PTH, out=NPZ):
    """Fold BatchNorm into the convs once and cache as a compact .npz."""
    sd = load_pth(pth)
    p = {}
    p["conv1.W"], p["conv1.b"] = _fold(sd, "conv1", "bn1")
    for L in range(1, 5):
        for B in range(2):
            pre = f"layer{L}.{B}"
            p[pre + ".c1.W"], p[pre + ".c1.b"] = _fold(sd, pre + ".conv1", pre + ".bn1")
            p[pre + ".c2.W"], p[pre + ".c2.b"] = _fold(sd, pre + ".conv2", pre + ".bn2")
            if pre + ".downsample.0.weight" in sd:
                p[pre + ".ds.W"], p[pre + ".ds.b"] = _fold(sd, pre + ".downsample.0", pre + ".downsample.1")
    p["fc.W"], p["fc.b"] = sd["fc.weight"].astype(F32), sd["fc.bias"].astype(F32)
    np.savez(out, **p)
    return p


def conv2d(x, W, b, stride=1, pad=0):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    Ho, Wo = win.shape[2], win.shape[3]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)
    out = cols @ W.reshape(F, -1).T + b
    return np.ascontiguousarray(out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2))


def maxpool3s2(x):
    x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    return sliding_window_view(x, (3, 3), axis=(2, 3))[:, :, ::2, ::2].max((-1, -2))


class ResNet18:
    def __init__(self, path=NPZ):
        if not os.path.exists(path):
            convert(out=path)
        d = np.load(path)
        self.p = {k: d[k] for k in d.files}

    def _block(self, x, pre, stride):
        p = self.p
        h = np.maximum(conv2d(x, p[pre + ".c1.W"], p[pre + ".c1.b"], stride, 1), 0)
        h = conv2d(h, p[pre + ".c2.W"], p[pre + ".c2.b"], 1, 1)
        sc = conv2d(x, p[pre + ".ds.W"], p[pre + ".ds.b"], stride, 0) if pre + ".ds.W" in p else x
        return np.maximum(h + sc, 0)

    def features(self, x):
        """Nx3x224x224 normalised input -> Nx512x7x7 layer4 activations."""
        h = np.maximum(conv2d(x, self.p["conv1.W"], self.p["conv1.b"], 2, 3), 0)
        h = maxpool3s2(h)
        for L in range(1, 5):
            for B in range(2):
                h = self._block(h, f"layer{L}.{B}", 2 if (L > 1 and B == 0) else 1)
        return h

    def imagenet_logits(self, x):
        return self.features(x).mean((2, 3)) @ self.p["fc.W"].T + self.p["fc.b"]


def prep_rgb(rgb_uint8):
    """HxWx3 or NxHxWx3 uint8 at 224 -> normalised Nx3x224x224."""
    x = np.asarray(rgb_uint8, F32) / 255
    if x.ndim == 3:
        x = x[None]
    return ((x.transpose(0, 3, 1, 2) - MEAN) / STD).astype(F32)


def prep_gray(gray_uint8):
    g = np.asarray(gray_uint8, F32)
    if g.ndim == 2:
        g = g[None]
    return prep_rgb(np.repeat(g[..., None], 3, -1))
