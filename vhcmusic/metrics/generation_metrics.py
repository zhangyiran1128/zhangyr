"""Generation quality and cross-modal matching metrics."""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def audio_feature_kld(reference_audio: Iterable[str | Path], generated_audio: Iterable[str | Path], bins: int = 64) -> float:
    """Compute KLD over log-mel energy distributions."""

    import librosa

    def collect(paths):
        values = []
        for path in paths:
            y, sr = librosa.load(path, sr=22050, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            values.extend(np.log1p(mel).ravel().tolist())
        return np.asarray(values, dtype=np.float32)

    ref = collect(reference_audio)
    gen = collect(generated_audio)
    lo, hi = float(min(ref.min(), gen.min())), float(max(ref.max(), gen.max()))
    p, _ = np.histogram(ref, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(gen, bins=bins, range=(lo, hi), density=True)
    return kl_divergence(p, q)


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """Frechet distance used by FAD-style embedding metrics."""

    from scipy.linalg import sqrtm

    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def fad_from_embeddings(reference_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> float:
    ref = np.asarray(reference_embeddings, dtype=np.float64)
    gen = np.asarray(generated_embeddings, dtype=np.float64)
    return frechet_distance(ref.mean(axis=0), np.cov(ref, rowvar=False), gen.mean(axis=0), np.cov(gen, rowvar=False))


def fad_score(reference, generated, checkpoint_dir=None):
    from frechet_audio_distance import FrechetAudioDistance

    model = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False,
                                 use_activation=False, ckpt_dir=checkpoint_dir)
    with TemporaryDirectory() as directory:
        base = Path(directory)
        for folder, paths in (("reference", reference), ("generated", generated)):
            destination = base / folder
            destination.mkdir()
            import librosa
            import soundfile as sf

            for i, path in enumerate(paths):
                audio, _ = librosa.load(path, sr=44100, mono=False)
                sf.write(destination / f"{i:06}.wav", audio.T, 44100, subtype="PCM_16")
        return float(model.score(str(base / "reference"), str(base / "generated"), dtype="float32"))


@lru_cache(maxsize=2)
def _clap(checkpoint, device):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt(ckpt=checkpoint)
    return model


def _cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / max(1e-8, np.linalg.norm(a) * np.linalg.norm(b)))


def clap_similarity(audio, prompt, checkpoint, device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _clap(checkpoint, device)
    with torch.inference_mode():
        text = model.get_text_embedding([prompt, ""], use_tensor=False)[0]
        music = model.get_audio_embedding_from_filelist(x=[str(audio)], use_tensor=False)[0]
    return _cosine(text, music)


@lru_cache(maxsize=2)
def _imagebind(checkpoint, device):
    import torch
    from imagebind.models import imagebind_model

    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval().to(device)


def imagebind_similarity(audio, frames, checkpoint, device=None):
    import torch
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    if not frames:
        raise ValueError("ImageBind requires at least one video keyframe")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _imagebind(checkpoint, device)
    with torch.inference_mode():
        embeddings = model({
            ModalityType.VISION: data.load_and_transform_vision_data(list(map(str, frames)), device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([str(audio)], device),
        })
    music = embeddings[ModalityType.AUDIO][0].cpu().numpy()
    return float(np.mean([_cosine(frame, music) for frame in embeddings[ModalityType.VISION].cpu().numpy()]))


def mos_summary(path: str | Path) -> dict[str, float]:
    """Read human MOS CSV/JSON and compute mean/std/count."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = np.asarray([float(x["score"]) if isinstance(x, dict) else float(x) for x in data], dtype=np.float32)
    else:
        import pandas as pd

        df = pd.read_csv(path)
        col = "score" if "score" in df.columns else df.columns[-1]
        scores = df[col].astype(float).to_numpy()
    return {"mean": float(scores.mean()), "std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0, "n": int(scores.size)}
