# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python-headless>=4.10",
#   "numpy>=2.0",
# ]
# ///
"""Extract the sharpest frame from each fixed-size chunk of a drone video,
inject per-frame DJI telemetry (SRT) into EXIF, and globally reject chunks
where the camera was refocusing.

Designed for DJI Mavic 3 Enterprise (M3E) MP4 + SRT pairs but works for any
video; metadata injection is silently skipped if no SRT is found.

Pipeline:

  Pass 1: read every frame, compute Laplacian-variance sharpness, keep the
          best frame per `--chunk` block (default 30 ≈ 1 s @ 30 fps).
  Filter: rolling local-baseline check (default ±5 chunks).  A chunk is
          rejected if its best score is below `--rel-floor` × the local
          median, or below `--abs-floor`.
  Pass 2: write surviving frames as JPEG (default Q95) and run exiftool to
          bake in GPS + gimbal pose + DateTimeOriginal + ISO/shutter/fnum.

External requirement: `exiftool` on PATH (for EXIF injection).  If absent,
frames still get written but without metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------- SRT parsing -------------------------------------------------------

SRT_RE = re.compile(
    r"FrameCnt:\s*(?P<fcnt>\d+).*?"
    r"DiffTime:\s*(?P<diff>\d+)\s*ms.*?"
    r"(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+).*?"
    r"\[iso:\s*(?P<iso>\d+)\].*?"
    r"\[shutter:\s*(?P<shutter>[^\]]+)\].*?"
    r"\[fnum:\s*(?P<fnum>[\d.]+)\].*?"
    r"\[ev:\s*(?P<ev>[-+]?[\d.]+)\].*?"
    r"\[focal_len:\s*(?P<focal>[\d.]+)\].*?"
    r"\[latitude:\s*(?P<lat>-?\d+\.\d+)\].*?"
    r"\[longitude:\s*(?P<lon>-?\d+\.\d+)\].*?"
    r"\[rel_alt:\s*(?P<rel>-?\d+\.\d+)\s+abs_alt:\s*(?P<abs>-?\d+\.\d+)\].*?"
    r"\[gb_yaw:\s*(?P<gy>-?\d+\.\d+)\s+gb_pitch:\s*(?P<gp>-?\d+\.\d+)\s+gb_roll:\s*(?P<gr>-?\d+\.\d+)\]",
    re.S,
)


def parse_srt(path: Path) -> dict[int, dict]:
    """Return {frame_idx (0-based): {tag: value_str}} for every parsable SRT block."""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="ignore")
    for blk in re.split(r"\n\s*\n", text):
        m = SRT_RE.search(blk)
        if not m:
            continue
        fidx = int(m["fcnt"]) - 1  # DJI SRT is 1-indexed; OpenCV is 0-indexed
        out[fidx] = {k: m[k] for k in m.groupdict()}
    return out


# ---------- Sharpness ---------------------------------------------------------


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class ChunkRecord:
    chunk_start: int
    best_idx: int
    score: float
    tmp_path: Path | None = None
    kept: bool = True
    rejection_reason: str = ""
    local_median: float = 0.0


# ---------- Per-video processing ---------------------------------------------


def pass1_extract(
    video_path: Path,
    work_dir: Path,
    chunk_size: int,
    jpeg_quality: int,
) -> tuple[list[ChunkRecord], float]:
    """Scan the video; write each chunk's sharpest frame as JPEG into work_dir.

    Returns (records, fps).  Records carry the per-chunk best score and tmp path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}", file=sys.stderr)
        return [], 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    pad = max(6, len(str(max(total, 1))))
    work_dir.mkdir(parents=True, exist_ok=True)

    records: list[ChunkRecord] = []
    best_frame: np.ndarray | None = None
    best_score = -1.0
    best_idx = -1
    chunk_start = 0
    frame_idx = 0

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
        if frame_idx - chunk_start >= chunk_size and best_frame is not None:
            name = f"{video_path.stem}_f{best_idx:0{pad}d}_lapvar{best_score:.0f}.jpg"
            tmp_path = work_dir / name
            cv2.imwrite(str(tmp_path), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            records.append(
                ChunkRecord(chunk_start=chunk_start, best_idx=best_idx, score=best_score, tmp_path=tmp_path)
            )
            chunk_start = frame_idx
            best_frame, best_score, best_idx = None, -1.0, -1

    # tail chunk
    if best_frame is not None:
        name = f"{video_path.stem}_f{best_idx:0{pad}d}_lapvar{best_score:.0f}.jpg"
        tmp_path = work_dir / name
        cv2.imwrite(str(tmp_path), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        records.append(
            ChunkRecord(chunk_start=chunk_start, best_idx=best_idx, score=best_score, tmp_path=tmp_path)
        )

    cap.release()
    return records, fps


def apply_focus_filter(
    records: list[ChunkRecord],
    rel_floor: float,
    abs_floor: float,
    window_half: int,
) -> None:
    """Mark each record as kept / rejected based on local + absolute thresholds.

    For each record, the local baseline is the median chunk score over
    [i - window_half, i + window_half] (excluding i itself), allowing the
    refocus dip to stand out against the surrounding sharp baseline.
    """
    scores = [r.score for r in records]
    n = len(records)
    for i, r in enumerate(records):
        lo = max(0, i - window_half)
        hi = min(n, i + window_half + 1)
        nbrs = scores[lo:i] + scores[i + 1 : hi]
        local = statistics.median(nbrs) if nbrs else r.score
        r.local_median = local
        reasons = []
        if r.score < abs_floor:
            reasons.append(f"score<{abs_floor:.1f} (abs floor)")
        if local > 0 and r.score < rel_floor * local:
            reasons.append(f"score<{rel_floor:.2f}×local_median ({rel_floor * local:.1f})")
        if reasons:
            r.kept = False
            r.rejection_reason = "; ".join(reasons)


# ---------- EXIF injection ---------------------------------------------------

EXIFTOOL_BIN = shutil.which("exiftool")


def shutter_to_seconds(shutter_str: str) -> float | None:
    """SRT shutter is 'N/D' (e.g. '1/1342.42').  Return seconds, or None on parse failure."""
    s = shutter_str.strip()
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except ValueError:
        return None


def srt_datetime_to_exif(srt_dt: str) -> tuple[str, str]:
    """SRT '2026-05-26 10:57:45.846' → ('2026:05:26 10:57:45', '846')."""
    main, _, sub = srt_dt.partition(".")
    main = main.replace("-", ":", 2)
    return main, sub.split("Z")[0][:3] if sub else ""


def build_exif_args(meta: dict, model: str, make: str) -> list[str]:
    """Build the list of -tag=value args for one image, from one SRT block's data."""
    args: list[str] = []
    if "lat" in meta:
        lat = float(meta["lat"])
        args += [f"-GPSLatitude={abs(lat)}", f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}"]
    if "lon" in meta:
        lon = float(meta["lon"])
        args += [f"-GPSLongitude={abs(lon)}", f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}"]
    if "abs" in meta:
        a = float(meta["abs"])
        args += [
            f"-GPSAltitude={abs(a)}",
            f"-GPSAltitudeRef={'0' if a >= 0 else '1'}",     # 0 = above MSL, 1 = below
            f"-XMP-drone-dji:AbsoluteAltitude={a:+.3f}",
        ]
    if "rel" in meta:
        args.append(f"-XMP-drone-dji:RelativeAltitude={float(meta['rel']):+.3f}")
    if "gy" in meta:
        args.append(f"-XMP-drone-dji:GimbalYawDegree={float(meta['gy']):+.2f}")
    if "gp" in meta:
        args.append(f"-XMP-drone-dji:GimbalPitchDegree={float(meta['gp']):+.2f}")
    if "gr" in meta:
        args.append(f"-XMP-drone-dji:GimbalRollDegree={float(meta['gr']):+.2f}")
    if "iso" in meta:
        args.append(f"-ISO={int(meta['iso'])}")
    if "fnum" in meta:
        args.append(f"-FNumber={float(meta['fnum'])}")
    if "focal" in meta:
        args.append(f"-FocalLength={float(meta['focal'])}")
    if "shutter" in meta:
        s = shutter_to_seconds(meta["shutter"])
        if s is not None:
            args.append(f"-ExposureTime={s}")
    if "ev" in meta:
        args.append(f"-ExposureCompensation={float(meta['ev'])}")
    if "dt" in meta:
        main, sub = srt_datetime_to_exif(meta["dt"])
        args += [f"-DateTimeOriginal={main}", f"-CreateDate={main}", f"-ModifyDate={main}"]
        if sub:
            args.append(f"-SubSecTimeOriginal={sub}")
    args += [f"-Make={make}", f"-Model={model}"]
    return args


def inject_exif_batch(
    pairs: list[tuple[Path, dict]],
    model: str,
    make: str,
    exiftool_extra: list[str] | None = None,
) -> tuple[int, int]:
    """Run a single exiftool invocation via stdin args file, one set of args per image.

    Returns (n_updated, n_failed).
    """
    if not EXIFTOOL_BIN or not pairs:
        return (0, len(pairs))
    arg_lines: list[str] = []
    for path, meta in pairs:
        for a in build_exif_args(meta, model=model, make=make):
            arg_lines.append(a)
        arg_lines.append("-overwrite_original")
        arg_lines.append(str(path))
        arg_lines.append("-execute")
    cmd = [EXIFTOOL_BIN, "-@", "-"]
    if exiftool_extra:
        cmd = [EXIFTOOL_BIN, *exiftool_extra, "-@", "-"]
    res = subprocess.run(
        cmd, input="\n".join(arg_lines), capture_output=True, text=True, check=False
    )
    out = (res.stdout or "") + (res.stderr or "")
    n_updated = out.count("1 image files updated")
    n_failed = len(pairs) - n_updated
    if n_failed and res.returncode != 0:
        print(f"    ! exiftool returned {res.returncode}; tail of stderr:", file=sys.stderr)
        print("    " + "\n    ".join(res.stderr.splitlines()[-10:]), file=sys.stderr)
    return (n_updated, n_failed)


# ---------- Per-video orchestrator -------------------------------------------


def process_video(
    video_path: Path,
    out_root: Path,
    chunk_size: int,
    jpeg_quality: int,
    rel_floor: float,
    abs_floor: float,
    window_half: int,
    skip_filter: bool,
    skip_exif: bool,
    srt_suffix: str,
    keep_rejected: bool,
    model: str,
    make: str,
) -> dict:
    out_dir = out_root / video_path.stem
    work_dir = out_dir / "_extracted"
    rejected_dir = out_dir / "rejected"

    srt_path = video_path.with_suffix(srt_suffix)
    srt = parse_srt(srt_path)
    print(f"  {video_path.name}: SRT={'found ('+str(len(srt))+' frames)' if srt else 'none'}")

    records, fps = pass1_extract(video_path, work_dir, chunk_size, jpeg_quality)
    if not records:
        return {"video": video_path.name, "n_chunks": 0}

    if not skip_filter:
        apply_focus_filter(records, rel_floor=rel_floor, abs_floor=abs_floor, window_half=window_half)
    n_kept = sum(1 for r in records if r.kept)
    n_rej = len(records) - n_kept

    # Move surviving frames to the clean output dir; rejected to /rejected (or delete).
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_pairs: list[tuple[Path, dict]] = []
    for r in records:
        if r.tmp_path is None or not r.tmp_path.exists():
            continue
        target_dir = out_dir if r.kept else (rejected_dir if keep_rejected else None)
        if target_dir is None:
            r.tmp_path.unlink(missing_ok=True)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        new_path = target_dir / r.tmp_path.name
        r.tmp_path.rename(new_path)
        r.tmp_path = new_path
        if r.kept and r.best_idx in srt:
            kept_pairs.append((new_path, srt[r.best_idx]))

    # Clean up the now-empty work dir.
    try:
        work_dir.rmdir()
    except OSError:
        pass

    # EXIF injection
    updated, failed = 0, 0
    if not skip_exif and kept_pairs:
        if EXIFTOOL_BIN is None:
            print("    ! exiftool not on PATH; skipping EXIF injection", file=sys.stderr)
        else:
            updated, failed = inject_exif_batch(kept_pairs, model=model, make=make)
            print(f"    exiftool: updated {updated}, failed {failed}")

    # Per-video chunk log (CSV) — useful for tuning thresholds.
    csv_path = out_dir / "_chunks.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["chunk_start", "best_idx", "score", "local_median", "kept", "rejection_reason", "tmp_path"]
        )
        for r in records:
            w.writerow(
                [
                    r.chunk_start,
                    r.best_idx,
                    f"{r.score:.2f}",
                    f"{r.local_median:.2f}",
                    int(r.kept),
                    r.rejection_reason,
                    str(r.tmp_path) if r.tmp_path else "",
                ]
            )

    summary = {
        "video": video_path.name,
        "fps": round(fps, 3),
        "n_chunks": len(records),
        "n_kept": n_kept,
        "n_rejected": n_rej,
        "score_min": round(min(r.score for r in records), 1),
        "score_p50": round(statistics.median(r.score for r in records), 1),
        "score_max": round(max(r.score for r in records), 1),
        "exif_updated": updated,
        "exif_failed": failed,
        "srt_available": bool(srt),
    }
    print(f"    -> {n_kept} kept, {n_rej} rejected  (score min/p50/max = {summary['score_min']}/{summary['score_p50']}/{summary['score_max']})")
    return summary


# ---------- CLI --------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", type=Path, help="Video files (default: *.MP4 + *.mp4 in cwd)")
    ap.add_argument("--chunk", type=int, default=30, help="Frames per chunk (default 30)")
    ap.add_argument("--out", type=Path, default=Path("sharp_frames"), help="Output root dir")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1–100 (default 95)")

    fg = ap.add_argument_group("global focus filter")
    fg.add_argument("--rel-floor", type=float, default=0.4,
                    help="Reject chunk if score < this fraction of the local median (default 0.4)")
    fg.add_argument("--abs-floor", type=float, default=0.0,
                    help="Absolute lapvar floor; chunks below this are always rejected (default 0)")
    fg.add_argument("--window-half", type=int, default=5,
                    help="Half-width (in chunks) of the local-median window (default 5 → 11 total)")
    fg.add_argument("--no-filter", action="store_true",
                    help="Disable global focus filter (keep all per-chunk maxima)")
    fg.add_argument("--no-keep-rejected", action="store_true",
                    help="Delete rejected frames instead of moving them to ./rejected/")

    eg = ap.add_argument_group("EXIF injection")
    eg.add_argument("--no-exif", action="store_true", help="Skip EXIF/XMP metadata injection")
    eg.add_argument("--srt-suffix", default=".SRT", help="SRT sibling file suffix (default .SRT)")
    eg.add_argument("--camera-model", default="M3E", help="EXIF Model tag to bake in (default M3E)")
    eg.add_argument("--camera-make", default="DJI", help="EXIF Make tag to bake in (default DJI)")

    args = ap.parse_args()
    if args.chunk < 1:
        ap.error("--chunk must be >= 1")
    if args.window_half < 1:
        ap.error("--window-half must be >= 1")

    videos = list(args.videos)
    if not videos:
        videos = sorted({*Path.cwd().glob("*.MP4"), *Path.cwd().glob("*.mp4")})
    if not videos:
        ap.error("no video files supplied or found")

    print(
        f"chunk={args.chunk}  out={args.out}  filter="
        f"{'OFF' if args.no_filter else f'rel<{args.rel_floor}×median(±{args.window_half}), abs<{args.abs_floor}'}  "
        f"exif={'OFF' if args.no_exif else ('ON' if EXIFTOOL_BIN else 'OFF (exiftool missing)')}"
    )
    summaries = []
    for v in videos:
        if not v.exists():
            print(f"  ! skipping missing file {v}", file=sys.stderr)
            continue
        s = process_video(
            v,
            out_root=args.out,
            chunk_size=args.chunk,
            jpeg_quality=args.quality,
            rel_floor=args.rel_floor,
            abs_floor=args.abs_floor,
            window_half=args.window_half,
            skip_filter=args.no_filter,
            skip_exif=args.no_exif,
            srt_suffix=args.srt_suffix,
            keep_rejected=not args.no_keep_rejected,
            model=args.camera_model,
            make=args.camera_make,
        )
        summaries.append(s)

    (args.out).mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
