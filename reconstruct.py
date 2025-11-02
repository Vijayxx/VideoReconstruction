from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from features import compute_similarity_matrix, prepare_features
from io_utils import (
    read_video_frames,
    write_frame_order_csv,
    write_similarity_report,
    write_timings,
    write_video,
)
from ordering import bidirectional_greedy, farthest_pair, two_opt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the temporal order of a shuffled single-shot video using classical CV cues.",
    )
    parser.add_argument("--input", required=True, help="Path to the shuffled input video.")
    parser.add_argument("--out", default="output/reconstructed.mp4", help="Destination for the reconstructed video.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output frame rate (must be 30).")
    parser.add_argument("--width", type=int, default=480, help="Downscaled width used for feature computation.")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes for pairwise similarities.")
    parser.add_argument("--w_ssim", type=float, default=0.35, help="Weight for the SSIM similarity component.")
    parser.add_argument("--w_hist", type=float, default=0.2, help="Weight for the HSV histogram similarity component.")
    parser.add_argument("--w_orb", type=float, default=0.45, help="Weight for the ORB similarity component.")
    parser.add_argument("--twoopt_iter", type=int, default=20000, help="Maximum 2-opt iterations.")
    parser.add_argument("--band", type=int, default=0, help="Band-limiting parameter for 2-opt (0 disables).")
    parser.add_argument(
        "--similarity_report",
        default="logs/similarity_report.csv",
        help="Optional path for dumping the fused similarity matrix.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the recovered frame order prior to rendering the output.",
    )
    return parser.parse_args()


def _front_loaded_cost(order: np.ndarray, distance: np.ndarray) -> float:
    if len(order) < 2:
        return 0.0
    weights = np.linspace(len(order) - 1, 1.0, len(order) - 1, dtype=np.float32)
    edge_costs = np.array(
        [distance[a, b] for a, b in zip(order[:-1], order[1:])],
        dtype=np.float32,
    )
    return float(np.dot(weights, edge_costs))


def main() -> None:
    args = parse_args()

    if args.fps != 30.0:
        raise ValueError("The output FPS must be fixed at 30 according to the specification.")

    weights = (args.w_ssim, args.w_hist, args.w_orb)
    if not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise ValueError("Weights must sum to 1.0.")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Similarity weights must be non-negative.")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    full_frames, small_frames, fps_reported = read_video_frames(
        args.input,
        downscale_width=args.width,
    )
    timings["extraction"] = time.perf_counter() - t0
    frame_count = len(full_frames)
    if frame_count != 300:
        print(f"[warn] Expected 300 frames but decoded {frame_count}. Continuing with fixed 30 fps output.")

    if fps_reported and abs(fps_reported - 30.0) > 1e-2:
        print(f"[warn] Input video reports {fps_reported:.2f} FPS; output will be forced to 30 FPS.")

    t0 = time.perf_counter()
    cached_features = prepare_features(small_frames)
    timings["feature_prep"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    similarity = compute_similarity_matrix(
        cached_features,
        weight_ssim=args.w_ssim,
        weight_hist=args.w_hist,
        weight_orb=args.w_orb,
        workers=args.workers,
    )
    timings["pairwise"] = time.perf_counter() - t0

    distance = 1.0 - similarity

    t0 = time.perf_counter()
    start_pair = farthest_pair(distance)
    initial_order = bidirectional_greedy(distance, start_pair)
    timings["path_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    refined_order = two_opt(
        initial_order,
        distance,
        max_iter=args.twoopt_iter,
        band=args.band,
    )
    timings["two_opt"] = time.perf_counter() - t0

    forward_cost = _front_loaded_cost(refined_order, distance)
    reversed_order = refined_order[::-1]
    reverse_cost = _front_loaded_cost(reversed_order, distance)

    auto_reverse = reverse_cost + 1e-9 < forward_cost
    final_order = reversed_order if auto_reverse else refined_order

    if auto_reverse:
        print("Heuristic orientation check selected the reversed ordering for smoother start transitions.")

    if args.reverse:
        final_order = final_order[::-1]

    avg_similarity = float(np.mean([similarity[a, b] for a, b in zip(final_order[:-1], final_order[1:])]))
    print(f"Average consecutive-frame similarity: {avg_similarity:.4f}")

    ordered_frames = [full_frames[idx] for idx in final_order]

    t0 = time.perf_counter()
    write_video(ordered_frames, args.out, fps=args.fps)
    timings["render"] = time.perf_counter() - t0

    frame_order_csv = "output/frame_order.csv"
    write_frame_order_csv(final_order, frame_order_csv)

    if args.similarity_report:
        write_similarity_report(similarity, args.similarity_report)

    timings["frame_count"] = frame_count
    timings["total"] = sum(
        timings[key]
        for key in ["extraction", "feature_prep", "pairwise", "path_build", "two_opt", "render"]
        if key in timings
    )
    timings["average_similarity"] = avg_similarity

    write_timings(timings, "logs/timing.json")


if __name__ == "__main__":
    main()
