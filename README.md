# sharpest-frames

Extract the sharpest frame from each fixed-size chunk of a drone video,
inject per-frame DJI telemetry into EXIF/XMP, and **globally reject chunks
where the camera was refocusing** — designed as a pre-processing step for
photogrammetry / 3D reconstruction pipelines (RealityScan/RealityCapture,
COLMAP, Gaussian splatting, etc.).

Built for **DJI Mavic 3 Enterprise (M3E)** `.MP4 + .SRT` pairs, but the SRT
half is optional — without an SRT file you still get sharpness-filtered
frames, just with no metadata baked in.

## What's new vs. naive per-chunk extraction

Per-chunk maxima are myopic: when the lens refocuses (commonly >1 s on the
M3E in video mode), every frame in 1–3 consecutive chunks is blurry — the
"sharpest of a blurry batch" is still blurry, and gets fed to your SfM
pipeline, hurting alignment.

This tool adds a **second-pass global filter** comparing each chunk's
score against a rolling local median across ±N neighbouring chunks. A
refocus event shows up as a clear dip surrounded by sharp baseline, so
it's caught and routed to a `rejected/` subfolder instead of getting
fed into the SfM stage.

It also **mirrors the DJI stills' EXIF layout** for every extracted frame
(GPS, gimbal pose, absolute/relative altitude in `XMP-drone-dji`, ISO,
shutter, f-number, focal length, sub-second timestamps) — so video frames
and stills can be ingested into the same SfM project and the bundle
adjuster treats them the same way.

## Usage

The script declares dependencies inline via [PEP 723](https://peps.python.org/pep-0723/),
so [`uv`](https://docs.astral.sh/uv/) runs it directly with no setup.
You also need [`exiftool`](https://exiftool.org/) on PATH for EXIF injection
(automatically skipped if missing).

```bash
uv run sharpest_frames.py [VIDEO ...] [OPTIONS]
```

With no positional arguments, every `*.MP4` / `*.mp4` in the current
directory is processed. For each video the script looks for a sibling SRT
(default `<stem>.SRT`) and pulls per-frame telemetry from it.

### Output

Frames are written as **lossless PNG at maximum compression effort** — no
quality flag, no JPEG escape hatch. The whole point of this tool is to
hand frames to SfM, and lossy compression measurably degrades sub-pixel
feature matching; even Q95 JPEG introduces enough chroma sub-sampling and
8×8-block artifacts to hurt bundle adjustment.

```
sharp_frames/
├── summary.json                          # one record per video
└── <video-stem>/
    ├── _chunks.csv                       # per-chunk diagnostic: score, local_median, kept/rejected, reason
    ├── <video-stem>_f{idx}_lapvar{score}.png   # kept frames (with EXIF/XMP)
    └── rejected/
        └── <video-stem>_f{idx}_lapvar{score}.png   # rejected frames (so you can eyeball false positives)
```

### Main options

| Flag | Default | Meaning |
|---|---|---|
| `--chunk N` | `30` | Frames per chunk (one candidate per chunk). 30 ≈ 1 s @ 30 fps. |
| `--out DIR` | `sharp_frames` | Output root. |

### Focus-filter options

| Flag | Default | Meaning |
|---|---|---|
| `--rel-floor F` | `0.4` | Reject chunk if its score < F × the local median of its ±N neighbours. |
| `--abs-floor F` | `0` | Absolute Laplacian-variance floor; chunks below are always rejected. Off by default. |
| `--window-half N` | `5` | Half-width of the rolling-median window (so the window has 2N+1 chunks). |
| `--no-filter` | off | Disable the global filter; behave like the original "per-chunk max" extractor. |
| `--no-keep-rejected` | off | Delete rejected frames instead of moving to `rejected/`. |

### EXIF options

| Flag | Default | Meaning |
|---|---|---|
| `--no-exif` | off | Skip EXIF/XMP injection entirely. |
| `--srt-suffix S` | `.SRT` | Sibling SRT filename suffix. |
| `--camera-model M` | `M3E` | Value to bake into the EXIF `Model` tag. |
| `--camera-make M` | `DJI` | Value to bake into the EXIF `Make` tag. |

### Telemetry written to each kept frame

From the per-frame SRT block matching the kept frame's index:

- Standard EXIF: `Make`, `Model`, `DateTimeOriginal`/`CreateDate`/`ModifyDate` + `SubSecTimeOriginal`, `ISO`, `ExposureTime` (parsed from SRT's `1/N` shutter), `FNumber`, `FocalLength`, `ExposureCompensation`
- Standard EXIF GPS: `GPSLatitude`/`GPSLatitudeRef`, `GPSLongitude`/`GPSLongitudeRef`, `GPSAltitude`/`GPSAltitudeRef`
- DJI XMP (`XMP-drone-dji` namespace, matching the stills): `AbsoluteAltitude`, `RelativeAltitude`, `GimbalYawDegree`, `GimbalPitchDegree`, `GimbalRollDegree`

## Tuning the focus filter

Start with the defaults and inspect `_chunks.csv` after a run. If too many
real chunks land in `rejected/`, lower `--rel-floor` (e.g. `0.3`). If
in-focus-but-low-contrast scenes are slipping through (rare for buildings,
common for sky/water), set a non-zero `--abs-floor` based on what you see
in the CSV's `score` column.

For a continuous "everything was out of focus" period that the relative
filter can't catch (because all neighbours are also bad), the absolute
floor is the right tool.

## Requirements

- Python ≥ 3.11
- `opencv-python-headless` ≥ 4.10 — resolved by `uv run`
- `numpy` ≥ 2.0 — resolved by `uv run`
- `exiftool` on PATH — only needed for EXIF injection
