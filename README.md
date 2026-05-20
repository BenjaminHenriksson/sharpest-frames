# sharpest-frames

Extract the sharpest frame from each fixed-size chunk of a video, where
sharpness is measured as the variance of the Laplacian of the grayscale frame.

Useful as a pre-processing step for photogrammetry / 3D reconstruction
pipelines (COLMAP, Gaussian splatting, etc.) where blurry frames hurt
feature matching.

## Usage

The script declares its own dependencies inline via [PEP 723](https://peps.python.org/pep-0723/),
so [`uv`](https://docs.astral.sh/uv/) can run it directly with no setup:

```bash
uv run sharpest_frames.py [VIDEO ...] [--chunk N] [--out DIR]
```

With no arguments it processes every `*.mp4` in the current directory and
writes results to `sharp_frames/<video-stem>/`.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--chunk` | `30` | Frames per chunk (one output frame per chunk) |
| `--out` | `sharp_frames` | Output root directory |

### Output

Each kept frame is written as PNG with a filename of the form:

```
frame_{best_idx}_chunk_{chunk_start}_lapvar_{score}.png
```

## Requirements

- Python ≥ 3.11
- `opencv-python-headless` ≥ 4.10
- `numpy` ≥ 2.0

(All resolved automatically by `uv run`.)
