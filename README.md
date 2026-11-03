# SmartVision

Context-aware PPE compliance and danger-zone monitoring. A frame goes through a COCO person detector and a fine-tuned PPE model, person localization, a danger-zone check, and a compliance check. Workers are tracked so a missing-PPE timer can persist, and an elapsed timer becomes a critical alert with a saved evidence frame.

Thresholds, class names, zones, and PPE rules are in `config/`.

## Pipeline

1. **Input.** A numpy frame (`H, W, 3`), an image file, a video file, an RTSP URL, or a webcam index.
2. **Detect.** `models/best.pt` supplies helmet, vest, gloves, boots, and goggles. People come from pretrained COCO `yolov8s.pt` unless `model.person_source` is `ppe_model`. `auto` probes MPS, then CUDA, then CPU.
3. **Localize.** Each person is represented by the bottom-center of their box.
4. **Zone check.** That point is tested with `cv2.pointPolygonTest`. The highest `risk_level` wins when zones overlap.
5. **Comply.** PPE is assigned by body region. Inside a zone the zone's `required_ppe` is required. Outside, `compliance.default_required_ppe` is required (helmet and vest).
6. **Track.** Person boxes are matched across frames by IoU and keep a stable id.
7. **Alert.** Outside a zone, missing PPE is an amber warning. Inside a zone, missing PPE starts a 2 second timer, then one critical alert, an evidence JPEG under `outputs/evidence/`, a row in `outputs/alerts.db`, and a per-worker cooldown.
8. **Draw.** Boxes are green, amber, or red, with the worker id and the missing items.

## Layout

```text
src/smartvision/     detector, zones, tracker, compliance, alerts, pipeline, visualize
app/streamlit_app.py dashboard skeleton (not wired to the pipeline yet)
config/              settings.yaml, zones.yaml, dataset_classes.yaml
scripts/run_video.py annotated video runner
tests/               one pytest module per pipeline stage, plus dataset prep
data/raw/            downloaded source datasets
data/merged/         remapped six-class YOLO set
data/samples/        video files for scripts/run_video.py
outputs/evidence/    critical-alert frames
outputs/alerts.db    critical-alert log
```

## Setup (macOS)

Use Python 3.10 or newer. With `model.device: auto`, detection probes MPS, then CUDA, then CPU. PyTorch on this Mac cannot use MPS, so auto resolves to CPU.

```bash
cd smartvision_coe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `.venv` is already present and the packages in `requirements.txt` are installed, skip straight to the commands below.

Run the tests:

```bash
source .venv/bin/activate
python -m pytest
```

Start the dashboard from the repo root:

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

The app listens on `http://localhost:8501`. The tiles are still placeholders; the dashboard is not wired to the pipeline yet.

Run a video through the backend and write the annotated file under `outputs/`:

```bash
python scripts/run_video.py data/samples/<video>
python scripts/run_video.py data/samples/<video> --skip 2
```

`--skip` is the number of frames left unprocessed between inferences. `0` runs every frame. A webcam is `python scripts/run_video.py 0`.

## Dataset

`scripts/prepare_dataset.py` downloads the Roboflow sets listed in `config/dataset_classes.yaml`, remaps them to `person`, `helmet`, `vest`, `gloves`, `boots`, and `goggles`, drops every other class, removes duplicate images by file hash, and writes an 80/10/10 split to `data/merged/`. The API key is `ROBOFLOW_API_KEY` in `.env`.

```bash
source .venv/bin/activate
python scripts/prepare_dataset.py
```

## Train and evaluate

Fine-tuning reads `training` in `config/settings.yaml` (YOLOv8s, image size 640, 50 epochs, batch 16, patience 10). CLI flags override those values. Device selection is MPS, then CUDA, then CPU. Runs are written under `outputs/runs/`, and the best checkpoint is copied to `models/best.pt`.

```bash
source .venv/bin/activate
python scripts/train.py
python scripts/evaluate.py --weights models/best.pt
```

Evaluation scores the test split and writes per-class precision, recall, mAP50, and mAP50-95 to `outputs/eval_results.csv` and `outputs/eval_results.md`.

A one-epoch check on about 200 training images:

```bash
python scripts/train.py --epochs 1 --fraction 200 40 --name smoke --close-mosaic 0
```

On this Mac, PyTorch 2.11 cannot use MPS (it requires macOS 14 or newer), so that check ran on CPU. Scaled to the full set it is about 53 hours, so the full run belongs on a Colab GPU. Open `notebooks/train_colab.ipynb`, set the runtime to GPU, and put `ROBOFLOW_API_KEY` in a Colab secret.

## Configuration

Edit `config/settings.yaml` for weights, `person_source`, confidence, per-class thresholds, image size, device, body regions, the default PPE list, the 2 second hold, the cooldown, tracker limits, and evidence paths.

Edit `config/zones.yaml` for polygon vertices, `risk_level`, and each zone's PPE list. Vertices are normalized (`0`–`1` on x and y). Machinery Zone requires helmet, vest, and boots. Welding Zone requires helmet, vest, and goggles. Gloves are detected but not required. People outside both zones are checked for helmet and vest.

Place clips in `data/samples/`. `models/best.pt` is required at runtime and is gitignored.
