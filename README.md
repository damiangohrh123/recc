# OCR Pipeline (RK3588 / RECC)

The RECC board reads a machine's HMI screen over HDMI. This allows the machine to be monitored, and operated when a rule matches, with no person at the controls. **System Architecture** below describes the full flow, including the parts that live outside this repo.

This repo holds the production C++ implementation: PP-OCRv6 tiny detection and recognition on the Rockchip RK3588 NPU via RKNN, with alarm-banner detection over the raw OCR output. Input is raw BGR888 only, because uncompressed pixel data preserves small text better.

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; raw BGR in |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | `[0,1]` normalisation |

Detection runs a single full-image pass at the detector's native 480x480 input. Alarm detection converts the frame to HSV, masks for red, and filters by area, aspect ratio, and screen position; if a banner is found, it reuses the OCR text already collected for that region instead of running a second pass.

### Why There Is No Preprocessing

Preprocessing was tested on both stages and dropped. Binarization and upscaling made detection worse, so detection runs on unprocessed input. CLAHE contrast enhancement and unsharp-mask sharpening, applied to each crop before resizing, hurt recognition accuracy on every test screen: the 82.5% baseline fell to 59.6% with CLAHE and 77.4% with sharpening. Recognition therefore also runs on unprocessed crops.

## System Architecture

Nothing in this repo drives the machine. `kvmd` captures the screen, `ocr_server` turns pixels into text, and `pipeline/automation/` decides whether a rule matched. Acting on that decision, by sending keyboard and mouse input back over USB HID, is the job of `recc_gen5_test_kit/automation_driver/`, which runs the binaries built here. The board as a whole can therefore control the machine, while this repo covers only the reading and the deciding. The HID output path has not yet been verified against real hardware.

Two independent services run on the board and never talk to each other directly:

- **`kvmd`** captures the HDMI input from the screen. It is a separate binary from this repo (not built here). Over a TCP socket on port 39000 it serves several channels: a continuous H.264 stream for live viewing, a JPEG mode, continuous raw BGR888 (full-frame or a crop), and a one-shot raw BGR888 frame on request. Only that last one is used by the OCR loop; `recc_gen5_test_kit/test_h264_raw.py` exercises the others. It has no knowledge of OCR.
- **`ocr_server`** (this repo) reads one frame at a time. It has no knowledge of `kvmd`; it only accepts raw BGR888 bytes over HTTP on port 8080 and returns text and alarm results.

A driving script on the board, for example `recc_gen5_test_kit/test_ocr_continuous.py` run over SSH, calls both. It requests one frame from `kvmd`, then sends those bytes to `ocr_server`, both over localhost. The raw frame never leaves the board; only the JSON result is small enough to be worth sending on to a separate host PC. Read frequency is set entirely by whatever calls the two services, as neither polls or pushes on a schedule of its own.

```mermaid
flowchart LR
    M[Machine HMI Screen] -- HDMI --> K["kvmd  (:39000)"]
    K -- "one-shot BGR888 frame, localhost" --> D[Driving script, on board]
    D -- "POST /ocr, localhost" --> O["ocr_server  (:8080)"]
    O -- "boxes + alarm JSON" --> D
    D -- "JSON result, over network" --> H[Host PC]
    K -. "H.264 live stream, over network" .-> V[Viewer]
    D -. "USB HID keyboard/mouse, if a rule matched" .-> M
```

## Directory Structure

```
recc/
  pipeline/                        # the two-stage OCR + decision/control system
    ocr/                           # Stage 1: OCR + alarm detection pipeline
      benchmark.cpp                # one-shot CLI tool; reports timing/CPU/memory (averaged over cycles)
      ppocr_det.cpp/h              # detection
      ppocr_rec.cpp/h              # recognition
      ppocr_system.cpp/h           # det + rec pipeline, NMS
      alarm_detector.cpp/h         # HSV-based alarm banner detection
      text_correction.cpp/h        # narrow post-processing fix for two known recognition mistakes
      rknn_executor.cpp/h          # low-level RKNN model runner
      preprocess.cpp/h             # normalisation shared by both models
    automation/                    # Stage 2: rule matching (see "Rule-Based Automation" below)
      step_matcher.cpp             # CLI: one rule step vs one screen -> JSON match
      rule_matcher.cpp/h           # the fuzzy keyword matcher itself
      json_value.cpp/h             # minimal JSON reader for /ocr responses
  api/                             # Internal OCR API (see "ocr_server (HTTP API)" below)
    ocr_server.cpp                 # the server: loads models once, serves POST /ocr
    http_server.cpp/h              # minimal HTTP layer over POSIX sockets
    json_write.cpp/h               # JSON string escaping, shared with step_matcher
  cmake/aarch64-toolchain.cmake    # cross-compile toolchain file (see Build below)
  third_party/                     # rknn_api.h only (librknnrt.so lives on the board)
  aarch64-ubuntu20.04-toolchain.tar.gz  # cached aarch64 cross-compile toolchain (gitignored), see the section below
  board_deploy/                    # testdata images, systemd service, sweep_det_thresholds.sh; ready to pscp to a board once built (see the section below)
  model/
    PP-OCRv6_tiny_det.onnx         # conversion source
    PP-OCRv6_tiny_rec.onnx         # conversion source
    PP-OCRv6_tiny_det_rk3588.rknn  # runtime model
    PP-OCRv6_tiny_rec_rk3588.rknn  # runtime model
    ppocr_keys_v6.txt              # character dictionary
```

## Getting Started

Follow these steps in order: build once, deploy to a board, run it, then install it as a service.

### Prerequisites

- An aarch64 gcc-9 toolchain matching the board's OS (Ubuntu 20.04, glibc 2.31), plus statically-built OpenCV 4.5.4 (core+imgproc only), Clipper/polyclipping, and zlib for aarch64. `aarch64-ubuntu20.04-toolchain.tar.gz` in the repo root provides all of it pre-built, saving the roughly 20-minute bootstrap.
- `librknnrt.so` on the board itself. It is already present at `/usr/lib` as part of the board's OS image, so no installation is needed.

### 1. Build

`board_deploy/benchmark`, `board_deploy/ocr_server`, and `board_deploy/step_matcher` are aarch64 binaries. The first two are statically linked against OpenCV, Clipper, and zlib, leaving only `librknnrt.so` dynamic; `step_matcher` requires none of them. They are gitignored rather than committed, to keep binary blobs out of git history, so a fresh clone needs a build before first use. Rebuild only when the C++ source changes.

The build produces four targets: the `ocr_core` library, `step_matcher`, and, with `-DBUILD_TOOLS=ON`, `benchmark` and `ocr_server`. Copy the three executables from `build/` into `board_deploy/` before deploying.

Extract the cached toolchain once:

```bash
tar xzf aarch64-ubuntu20.04-toolchain.tar.gz -C ~
bash ~/fix_toolchain_paths.sh
```

This creates `~/cross` and `~/cross20` (the aarch64 toolchain), `~/deps-arm64` (OpenCV/Clipper/zlib), and `~/rknn-arm64` (RKNN SDK libs), and rewrites the archive's baked-in absolute paths to your `$HOME`. `fix_toolchain_paths.sh` is safe to re-run.

The archive already contains the wrapper compilers (`~/cross20/wrap/aarch64-linux-gnu-g++` and `-gcc`), which pass the correct `-B` flags for the target binutils and the gcc-9 frontend. `cmake/aarch64-toolchain.cmake` in this repo points at them.

Export `LD_LIBRARY_PATH` so the wrappers can find their own host libraries, then configure and build:

```bash
export LD_LIBRARY_PATH=$HOME/cross20/usr/lib/x86_64-linux-gnu:$HOME/cross/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Use $HOME, not ~: the shell does not expand a tilde inside a word that starts
# with -D, so cmake would receive the literal "~/deps-arm64/..." and the build
# would fail later on a missing opencv2/core.hpp.
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
      -DOPENCV_INCLUDE_DIR=$HOME/deps-arm64/include/opencv4 -DOPENCV_LIB_DIR=$HOME/deps-arm64/lib \
      -DCLIPPER_INCLUDE_DIR=$HOME/deps-arm64/include -DCLIPPER_LIB_DIR=$HOME/deps-arm64/lib \
      -DZLIB_LIB_DIR=$HOME/deps-arm64/lib -DRKNN_SDK_LIB_DIR=$HOME/rknn-arm64/lib \
      -DBUILD_TOOLS=ON
cmake --build build -j
```

### 2. Deploy to a Board

New board:

1. Copy the binaries and model files over (from Windows/WSL):

   ```bash
   pscp -r board_deploy tpsadmin@<board-ip>:/home/tpsadmin/board_deploy
   pscp -r model tpsadmin@<board-ip>:/home/tpsadmin/model
   ```

2. Test manually before adding anything to systemd. Run `ocr_server` by hand (see "Run It" below) and confirm `/health` responds, or run `benchmark` against one of the `testdata/*.bgr888` files. This catches a wrong path or a missing model file in plain view, rather than inside a service that retries silently.
3. Install it as a service (see "Run as a Service" below) so it starts on every boot and restarts itself if it crashes.
4. Record the board's IP address wherever it needs to be reached from, such as the host PC. `ocr_server`, the models, and the service file are identical on every board; only the IP differs.

Existing board, after a rebuild: re-upload just the changed binary and restart the service.

```bash
pscp board_deploy/ocr_server tpsadmin@192.168.1.101:/home/tpsadmin/board_deploy/ocr_server
```

Then on the board: `sudo systemctl restart ocr_server`.

### 3. Run It

```bash
cd ~/board_deploy
./ocr_server /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt 8080
```

Confirm it's up:

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/health').read())"
```

`/ocr` requires the 8-byte width and height header before the pixel bytes, so it is not a plain `curl --data-binary @file` call. See `recc_gen5_test_kit/test_ocr_continuous.py` for a working example, or the wire format under Usage below.

### 4. Run as a Service (Survives Reboot)

`board_deploy/ocr_server.service` is a systemd unit that starts `ocr_server` on boot and restarts it if it crashes. `kvmd` has no equivalent yet. It is backgrounded once by `kvmd_run.sh`, with no restart-on-crash supervision; giving it the same systemd treatment is a future improvement.

```bash
sudo cp ~/board_deploy/ocr_server.service /etc/systemd/system/ocr_server.service
sudo systemctl daemon-reload
sudo systemctl enable ocr_server
sudo systemctl start ocr_server
```

Check status or logs:

```bash
sudo systemctl status ocr_server --no-pager
journalctl -u ocr_server -f
```

## Usage

### `benchmark` (One-Shot CLI)

Loads a raw `.bgr888` frame, runs detection, recognition, and alarm detection, prints the results, and reports timing, CPU, and memory. Its detection defaults match `ocr_server`'s production values (det_thresh 0.2, box_thresh 0.4, unclip_ratio 1.5, max_candidates 3000), so a plain run reproduces server behaviour. Pass alternatives positionally to sweep.

```bash
cd ~/board_deploy
./benchmark /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt testdata/alarm_1024x768.bgr888
```

`testdata/` also has `auto_mode_1_1024x768.bgr888`, `auto_mode_2_1024x768.bgr888`, `normal_run_1024x768.bgr888`, and `full_test_1024x384.bgr888`. Pass a cycle count to average the timing, CPU, and memory numbers over repeated runs, for example `... testdata/alarm_1024x768.bgr888 10`.

The detector's own thresholds (`det_thresh`, `box_thresh`, `unclip_ratio`, `max_candidates`) are also CLI-configurable, as optional positional args after cycles and drop_score: `... testdata/alarm_1024x768.bgr888 1 0.4 <det_thresh> <box_thresh> <unclip_ratio> <max_candidates>`. These had been hardcoded since the project's first commit, with no record of being tested against alternatives. `board_deploy/sweep_det_thresholds.sh` sweeps a small grid of them against every file in `testdata/` and logs box counts, timing, and recognized text per combination to `sweep_results.csv`. Run it from `board_deploy/` after rebuilding `benchmark`.

### `ocr_server` (HTTP API)

`ocr_server` (`api/`) loads the detection and recognition models once at startup, then serves OCR and alarm-detection results over HTTP, rather than once per invocation as `benchmark` does.

#### `GET /health`

Returns `200` with `{"status":"ok"}` once models are loaded.

#### `POST /ocr`

Request body: an 8-byte header followed by tightly-packed pixel data, no padding:

```
[4 bytes: width,  big-endian uint32]
[4 bytes: height, big-endian uint32]
[width * height * 3 bytes: BGR888 pixel data, row-major, 3 bytes/pixel]
```

This is the same byte order `kvmd`'s raw channel uses for its width, height, and size fields, so a client that already speaks that protocol, such as `recc_gen5_test_kit/test_ocr_continuous.py`, needs no second convention for this endpoint.

Response `200`:

```json
{
  "boxes": [
    { "text": "ALARM 0089", "score": 0.97, "box": [[102,18],[210,18],[210,40],[102,40]] }
  ],
  "alarm": {
    "detected": true,
    "bbox": [80, 10, 900, 60],
    "text": "ALARM 0089 Kerf check: off center Z-EN"
  },
  "timing_ms": 412.3
}
```

`alarm` is always present. The `bbox` and `text` fields inside it appear only when `detected` is `true`. `boxes` holds the same per-box text, score, and quad data `benchmark` prints, serialized as JSON.

Response `400` if the body is too short, or if its size does not match `8 + width*height*3` for the given width and height. The body is `{"error": "..."}` describing the mismatch.

## Rule-Based Automation (Prototype)

`pipeline/automation/` matches a person-written rule, meaning a keyword to look for and an action to take once it is found, against real OCR output using fuzzy keyword matching. It never touches an image, only the text and coordinates OCR has already produced. Unlike the rest of `pipeline/`, it has no OpenCV or RKNN dependency and is not gated behind `BUILD_TOOLS` or a cross-compile toolchain, so it builds on any machine with a C++17 compiler:

```bash
cmake --build build --target step_matcher
```

Its single binary, `step_matcher`, checks one rule step against one screen file and prints the match as JSON. It is driven by `recc_gen5_test_kit/automation_driver/`, which reads the screen live and can send real USB HID input. The driver's `--dry-run --replay` mode performs the same check against a saved screen without touching hardware. See that folder's README for what is tested and what still needs real-board verification, and `recc_gen5_test_kit/automation_poc/` for the captured screen and rule files it runs against.

## Current Limitations

- One request is handled at a time, on the calling thread. There is no concurrency.
- No keep-alive, chunked encoding, or HTTPS. Every request opens a new connection.
- No authentication. Anyone who can reach the port can call it.
- The HTTP layer (`http_server.h/.cpp`) is a small implementation over POSIX sockets, not a vendored library.

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host, for model conversion) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2+ (tested: 2.4.2a2) |
| OpenCV / Clipper / zlib (board) | statically linked, core+imgproc only (see `CMakeLists.txt`) |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board IP (this dev board) | 192.168.1.101, user tpsadmin |
| Board deploy path | `/home/tpsadmin/board_deploy` |
