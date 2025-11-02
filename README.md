# Jumbled Frames Reconstruction — Classical Cues

This project reconstructs the correct chronological order of a shuffled single-shot video (10 s at 30 fps → 300 frames). The output is a 30 fps clip whose pacing matches the original as closely as possible.

## Method Overview

1. **Frame Extraction & Downscaling** — Decode all frames, keep the full-resolution images for rendering, and create a 320 px wide working copy for feature computation.
2. **Multi-Cue Similarity** — For every pair of downscaled frames:
   - Structural Similarity Index (SSIM) on grayscale images.
   - HSV colour histogram intersection with 32×32×32 bins (L1-normalised).
   - ORB keypoints (500 features) matched with BF-Hamming and Lowe’s 0.75 ratio test.
   - Fuse the scores with weights (`w_ssim + w_hist + w_orb = 1`).
3. **Path Construction** — Treat the problem as an open TSP:
   - Seed with the farthest pair of frames (largest fused distance).
   - Grow the path by attaching the best candidate to either end (bi-directional greedy).
   - Refine the ordering via 2-opt (optionally band-limited) to reduce the total distance.
4. **Output & Logging** — Reorder the original frames, render the clip at exactly 30 fps, and save artefacts: frame order CSV, timing breakdown JSON, and an optional similarity report.

Pairwise computations are parallelised (multiprocessing), keeping precomputed per-frame descriptors cached in each worker to stay within laptop-friendly performance bounds (≈45 k pairs for 300 frames).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r jumbled_classical/requirements.txt
```

## Usage

```bash
python jumbled_classical/reconstruct.py \
  --input jumbled_video.mp4 \
  --out output/reconstructed.mp4 \
  --fps 30 \
  --workers 8 \
  --w_ssim 0.7 \
  --w_hist 0.2 \
  --w_orb 0.1
```

Key flags:

- `--width` sets the downscaled feature width (default 320).
- `--twoopt_iter` upper-bounds 2-opt iterations (default 20 000).
- `--band` enables band-limited 2-opt when >0.
- Automatic orientation heuristics favour smooth early transitions; `--reverse` still flips manually if needed.
- Weights must sum to 1.0; the CLI enforces it.

## Outputs

- `output/reconstructed.mp4` — reconstructed 30 fps video.
- `output/frame_order.csv` — frame indices in recovered order.
- `logs/timing.json` — extraction, feature, pairwise, path build, 2-opt, render, total timings.
- `logs/similarity_report.csv` — optional fused similarity matrix (set with `--similarity_report`).

The script prints the **average consecutive-frame similarity**, which provides a quick sanity check for temporal smoothness. Higher values indicate smoother recovered pacing.
