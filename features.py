from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

_FEATURE_CACHE: List["FrameFeatures"] | None = None
_WEIGHTS: Dict[str, float] | None = None
_BF_MATCHER: cv2.BFMatcher | None = None


@dataclass
class FrameFeatures:
    gray: np.ndarray
    hsv_hist: np.ndarray
    descriptors: Optional[np.ndarray]
    descriptor_count: int


def prepare_features(frames: Sequence[np.ndarray]) -> List[FrameFeatures]:
    orb = cv2.ORB_create(nfeatures=1000)
    features: List[FrameFeatures] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=[32, 32, 32],
            ranges=[0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        hist_sum = float(hist.sum())
        if hist_sum > 0.0:
            hist /= hist_sum

        keypoints, descriptors = orb.detectAndCompute(gray, mask=None)
        descriptor_count = 0 if descriptors is None else int(descriptors.shape[0])

        features.append(
            FrameFeatures(
                gray=gray,
                hsv_hist=hist,
                descriptors=descriptors,
                descriptor_count=descriptor_count,
            )
        )
    return features


def compute_similarity_matrix(
    features: Sequence[FrameFeatures],
    *,
    weight_ssim: float,
    weight_hist: float,
    weight_orb: float,
    workers: Optional[int] = None,
) -> np.ndarray:
    validate_weights(weight_ssim, weight_hist, weight_orb)

    n = len(features)
    similarity = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(similarity, 1.0)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    worker_count = _resolve_worker_count(workers)
    weights = {
        "weight_ssim": weight_ssim,
        "weight_hist": weight_hist,
        "weight_orb": weight_orb,
    }

    if worker_count == 1:
        iterator = map(
            lambda pair: _pair_similarity_direct(pair, features, weights),
            pairs,
        )
    else:
        with Pool(
            processes=worker_count,
            initializer=_pool_initializer,
            initargs=(features, weights),
        ) as pool:
            iterator = pool.imap_unordered(_pair_similarity_worker, pairs, chunksize=64)
            iterator = tqdm(iterator, total=len(pairs), desc="Pairwise similarities")
            results = list(iterator)
    if worker_count == 1:
        results = list(iterator)

    for i, j, fused in results:
        similarity[i, j] = fused
        similarity[j, i] = fused

    return similarity


def validate_weights(w_ssim: float, w_hist: float, w_orb: float) -> None:
    total = w_ssim + w_hist + w_orb
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("Similarity weights must sum to 1.0.")
    for name, value in [("w_ssim", w_ssim), ("w_hist", w_hist), ("w_orb", w_orb)]:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")


def _resolve_worker_count(workers: Optional[int]) -> int:
    if workers is None or workers <= 0:
        return max(1, cpu_count() - 1)
    return workers


def _pool_initializer(features: Sequence[FrameFeatures], weights: Dict[str, float]) -> None:
    global _FEATURE_CACHE, _WEIGHTS, _BF_MATCHER
    _FEATURE_CACHE = list(features)
    _WEIGHTS = dict(weights)
    _BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _pair_similarity_worker(pair: Tuple[int, int]) -> Tuple[int, int, float]:
    assert _FEATURE_CACHE is not None and _WEIGHTS is not None
    return _pair_similarity_direct(pair, _FEATURE_CACHE, _WEIGHTS, matcher=_BF_MATCHER)


def _pair_similarity_direct(
    pair: Tuple[int, int],
    features: Sequence[FrameFeatures],
    weights: Dict[str, float],
    matcher: Optional[cv2.BFMatcher] = None,
) -> Tuple[int, int, float]:
    i, j = pair
    fi = features[i]
    fj = features[j]

    ssim_score = structural_similarity(
        fi.gray,
        fj.gray,
        data_range=255,
    )
    ssim_score = float(np.clip(ssim_score, 0.0, 1.0))

    hist_intersection = float(np.minimum(fi.hsv_hist, fj.hsv_hist).sum())
    hist_intersection = float(np.clip(hist_intersection, 0.0, 1.0))

    orb_score = _orb_similarity(fi, fj, matcher)

    fused = (
        weights["weight_ssim"] * ssim_score
        + weights["weight_hist"] * hist_intersection
        + weights["weight_orb"] * orb_score
    )
    fused = float(np.clip(fused, 0.0, 1.0))
    return i, j, fused


def _orb_similarity(
    fi: FrameFeatures,
    fj: FrameFeatures,
    matcher: Optional[cv2.BFMatcher],
) -> float:
    if fi.descriptors is None or fj.descriptors is None:
        return 0.0

    if matcher is None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(fi.descriptors, fj.descriptors, k=2)

    good = 0
    for m in matches:
        if len(m) == 2:
            best, second = m
            if best.distance < 0.75 * second.distance:
                good += 1

    denom = max(1, min(fi.descriptor_count, fj.descriptor_count))
    score = good / float(denom)
    return float(np.clip(score, 0.0, 1.0))
