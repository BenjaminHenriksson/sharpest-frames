# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python-headless>=4.10",
#   "numpy>=2.0",
# ]
# ///
"""Extract the sharpest frame from each fixed-size chunk of a video.

Sharpness is measured as the variance of the Laplacian of the grayscale
frame. For each chunk of `--chunk` consecutive frames the frame with the
highest variance is written to `<out>/<video-stem>/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_video(video_path: Path, out_dir: Path, chunk_size: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}", file=sys.stderr)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {video_path.name}: ~{total} frames @ {fps:.3f} fps -> {out_dir}")

    pad = max(6, len(str(max(total, 1))))
    best_frame: np.ndarray | None = None
    best_score = -1.0
    best_idx = -1
    chunk_start = 0
    frame_idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = laplacian_variance(gray)
        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = frame_idx

        frame_idx += 1
        if frame_idx - chunk_start >= chunk_size:
            if best_frame is not None:
                name = f"frame_{best_idx:0{pad}d}_chunk_{chunk_start:0{pad}d}_lapvar_{best_score:.2f}.png"
                cv2.imwrite(
                    str(out_dir / name),
                    best_frame,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                )
                written += 1
            chunk_start = frame_idx
            best_frame = None
            best_score = -1.0
            best_idx = -1

    if best_frame is not None:
        name = f"frame_{best_idx:0{pad}d}_chunk_{chunk_start:0{pad}d}_lapvar_{best_score:.2f}.png"
        cv2.imwrite(
            str(out_dir / name),
            best_frame,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        written += 1

    cap.release()
    print(f"    wrote {written} frames (read {frame_idx})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", type=Path, help="Video files (default: *.mp4 in cwd)")
    ap.add_argument("--chunk", type=int, default=30, help="Frames per chunk (default 30)")
    ap.add_argument("--out", type=Path, default=Path("sharp_frames"), help="Output root dir")
    args = ap.parse_args()

    if args.chunk < 1:
        ap.error("--chunk must be >= 1")

    videos = args.videos or sorted(Path.cwd().glob("*.mp4"))
    if not videos:
        ap.error("no video files supplied or found")

    print(f"chunk size: {args.chunk}, output: {args.out}")
    for v in videos:
        if not v.exists():
            print(f"  ! skipping missing file {v}", file=sys.stderr)
            continue
        process_video(v, args.out / v.stem, args.chunk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
