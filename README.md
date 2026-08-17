# OCR Pipeline (RK3588 / RECC)

PP-OCRv6 tiny detection and recognition on the Rockchip RK3588 NPU via RKNN, reading text off machine HMI screens, with alarm-banner detection on top of the raw OCR output. This repo covers the production C++ implementation. Input is raw BGR888 only, since uncompressed pixel data preserves small text better.

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; raw BGR in |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | `[0,1]` normalisation |

Detection runs a single full-image pass at the detector's native 480x480 input. Alarm detection converts the frame to HSV, masks for red, and filters by area, aspect ratio, and screen position; if a banner is found, it reuses the OCR text already collected for that region instead of running a second pass. Preprocessing (binarization, upscaling) was tested for detection and dropped: it made results worse, so detection runs on unprocessed input. Preprocessing was also tried on the recognition side (CLAHE contrast enhancement, unsharp-mask sharpening on each crop before resizing) and dropped for the same reason: both measurably hurt accuracy on every test screen (82.5% baseline dropped to 59.6% with CLAHE, 77.4% with sharpening), so recognition also runs on unprocessed crops.

## System Architecture

The RECC board sits between a machine's HMI screen and the network. It only reads the screen and reports what OCR finds; it does not control the machine.

Two independent services run on the board and never talk to each other directly:

- **`kvmd`** captures the HDMI input from the screen. It is a separate binary from this repo (not built here). Over a TCP socket on port 39000, it serves a continuous H.264 stream for live viewing, and a one-shot raw BGR888 frame on request. It has no knowledge of OCR.
- **`ocr_server`** (this repo) reads one frame at a time. It has no knowledge of `kvmd`; it only accepts raw BGR888 bytes over HTTP on port 8080 and returns text and alarm results.

A driving script, running locally on the board (e.g. `recc_gen5_test_kit/test_ocr_continuous.py` over SSH), calls both: it requests one frame from `kvmd`, then sends those bytes to `ocr_server`, both over localhost. The raw frame never leaves the board. Only the small JSON result is compact enough to be worth sending onward over the network, to a separate host PC. Read frequency is controlled entirely by whatever calls `kvmd` and `ocr_server`; neither service polls or pushes on a schedule of its own.

```mermaid
flowchart LR
    M[Machine HMI Screen] -- HDMI --> K["kvmd  (:39000)"]
    K -- "one-shot BGR888 frame, localhost" --> D[Driving script, on board]
    D -- "POST /ocr, localhost" --> O["ocr_server  (:8080)"]
    O -- "boxes + alarm JSON" --> D
    D -- "JSON result, over network" --> H[Host PC]
    K -. "H.264 live stream, over network" .-> V[Viewer, e.g. nano Virtual Console]
```

Each OCR read is a full round trip on the board itself: request a frame, wait for it, send it, get a result back, all over localhost. Nothing is cached or streamed automatically. Only the compact result, not the raw frame, is meant to cross the network to a host PC.

## Directory Structure

```
recc/
  pipeline/                        # the two-stage OCR + decision/control system
    ocr/                           # Stage 1: OCR + alarm detection pipeline
      benchmark.cpp                # one-shot CLI tool; also reports timing/CPU/memory with cycles > 1
      ppocr_det.cpp/h              # detection
      ppocr_rec.cpp/h              # recognition
      ppocr_system.cpp/h           # det + rec pipeline, NMS
      alarm_detector.cpp/h         # HSV-based alarm banner detection
      text_correction.cpp/h        # narrow post-processing fix for two known recognition mistakes
      rknn_executor.cpp/h          # low-level RKNN model runner
      preprocess.cpp/h
      image_io.cpp/h               # raw .bgr888 file loading, used by benchmark
    automation/                    # Stage 2 prototype: rule matching against a saved OCR result, see README below
  api/                             # Internal OCR API (ocr_server + minimal HTTP layer, see README below)
  third_party/                     # RKNN SDK headers + librknnrt.so
  aarch64-ubuntu20.04-toolchain.tar.gz  # cached aarch64 cross-compile toolchain (gitignored), see README below
  board_deploy/                    # sample images + systemd service, ready to pscp to a board once built (see README below)
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

- An aarch64 gcc-9 toolchain matching the board's OS (Ubuntu 20.04, glibc 2.31), plus statically-built OpenCV 4.5.4 (core+imgproc only), Clipper/polyclipping, and zlib for aarch64. `aarch64-ubuntu20.04-toolchain.tar.gz` (repo root) has all of this pre-built, saving the ~20-minute bootstrap.
- `librknnrt.so` on the board itself. This is already present at `/usr/lib` as part of the board's OS image, so there's nothing to install there.

### 1. Build

`board_deploy/benchmark` and `board_deploy/ocr_server` are aarch64 binaries, statically linked against OpenCV, Clipper, and zlib. They're gitignored, not committed (to avoid bloating git history with binary blobs), so a fresh clone needs a build before first use; rebuild only when the C++ source changes.

Extract the cached toolchain once:

```bash
tar xzf aarch64-ubuntu20.04-toolchain.tar.gz -C ~
bash ~/fix_toolchain_paths.sh
```

This creates `~/cross` and `~/cross20` (the aarch64 toolchain), `~/deps-arm64` (OpenCV/Clipper/zlib), and `~/rknn-arm64` (RKNN SDK libs), and rewrites the archive's baked-in absolute paths to your `$HOME`. `fix_toolchain_paths.sh` is safe to re-run.

Next, build wrapper compilers pointing `-B` at `~/cross/usr/aarch64-linux-gnu/bin` (target binutils) and `~/cross20/usr/bin` (gcc-9 frontend); see `~/cross20/wrap/aarch64-linux-gnu-g++` in the archive for the exact form. Then export `LD_LIBRARY_PATH` to include `~/cross20/usr/lib/x86_64-linux-gnu` and `~/cross/usr/lib/x86_64-linux-gnu`.

Configure and build with those wrapper compilers and the extracted dependency paths:

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=<toolchain file using the wrapper compilers above> \
         -DOPENCV_INCLUDE_DIR=~/deps-arm64/include/opencv4 -DOPENCV_LIB_DIR=~/deps-arm64/lib \
         -DCLIPPER_INCLUDE_DIR=~/deps-arm64/include -DCLIPPER_LIB_DIR=~/deps-arm64/lib \
         -DZLIB_LIB_DIR=~/deps-arm64/lib -DRKNN_SDK_LIB_DIR=~/rknn-arm64/lib \
         -DBUILD_TOOLS=ON
make
```

`BUILD_TOOLS=ON` builds both `benchmark` and `ocr_server`; both need `RKNN_SDK_LIB_DIR` pointing at a real `librknnrt.so`, since they link the actual board runtime.

### 2. Deploy to a Board

New board:

1. Copy the binaries and model files over (from Windows/WSL):

   ```bash
   pscp -r board_deploy tpsadmin@<board-ip>:/home/tpsadmin/board_deploy
   pscp -r model tpsadmin@<board-ip>:/home/tpsadmin/model
   ```

   `librknnrt.so` should already be present at `/usr/lib` as part of the board's own OS image.

2. Test manually first, before wiring anything into systemd: run `ocr_server` by hand (see "Run It" below) and confirm `/health` responds, or run `benchmark` against one of the `testdata/*.bgr888` files. This catches a wrong path or missing model file while watching it directly, instead of inside a service that silently retries.
3. Install it as a service (see "Run as a Service" below) so it starts on every boot and restarts itself if it crashes.
4. Note this board's IP address wherever it needs to be reachable from (e.g. the host PC). `ocr_server`, the models, and the service file are identical across every board; only the IP differs.

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

`/ocr` needs the 8-byte width/height header prepended before the pixel bytes, so it isn't a plain `curl --data-binary @file` call. See `recc_gen5_test_kit/test_ocr_continuous.py` for a working example that builds this request correctly, or the wire format under Usage below.

### 4. Run as a Service (Survives Reboot)

`board_deploy/ocr_server.service` is a systemd unit that starts `ocr_server` on boot and restarts it automatically if it crashes. `kvmd` has no equivalent yet; it's only backgrounded once by `kvmd_run.sh`, with no restart-on-crash supervision. Giving it the same systemd treatment is a future improvement.

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

Decodes an image, runs detection, recognition, and alarm detection, prints results, and (with a cycle count) reports timing/CPU/memory.

```bash
cd ~/board_deploy
./benchmark /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt testdata/alarm_1024x768.bgr888
```

`testdata/` also has `auto_mode_1_1024x768.bgr888`, `auto_mode_2_1024x768.bgr888`, `normal_run_1024x768.bgr888`, and `full_test_1024x384.bgr888`. Add a cycle count for timing/CPU/memory stats, e.g. `... testdata/alarm_1024x768.bgr888 10`.

The detector's own thresholds (`det_thresh`, `box_thresh`, `unclip_ratio`, `max_candidates`) are also CLI-configurable, as optional positional args after cycles and drop_score: `... testdata/alarm_1024x768.bgr888 1 0.4 <det_thresh> <box_thresh> <unclip_ratio> <max_candidates>`. These had been hardcoded since the project's first commit with no record of ever being tested against alternatives. `board_deploy/sweep_det_thresholds.sh` sweeps a small grid of these against every file in `testdata/` and logs box counts, timing, and recognized text per combination to `sweep_results.csv`, run it from `board_deploy/` after rebuilding `benchmark`.

### `ocr_server` (HTTP API)

`ocr_server` (`api/`) loads the detection/recognition models once at startup, then serves OCR and alarm-detection results over HTTP, rather than running once per invocation like `benchmark` does.

#### `GET /health`

Returns `200` with `{"status":"ok"}` once models are loaded.

#### `POST /ocr`

Request body: an 8-byte header followed by tightly-packed pixel data, no padding:

```
[4 bytes: width,  big-endian uint32]
[4 bytes: height, big-endian uint32]
[width * height * 3 bytes: BGR888 pixel data, row-major, 3 bytes/pixel]
```

Same byte order `kvmd`'s own raw channel uses for its width/height/size fields, so a client that already speaks that protocol (e.g. `recc_gen5_test_kit/test_ocr_continuous.py`) needs no second convention to talk to this endpoint.

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

`alarm` is always present; `bbox`/`text` inside it only appear when `detected` is `true`. `boxes` is the same per-box text/score/quad data `benchmark` prints, just serialized as JSON.

Response `400` if the body is too short, or its size doesn't match `8 + width*height*3` for the given width/height: `{"error": "..."}` describing the mismatch.

## Current Limitations

- One request handled at a time, on the calling thread; no concurrency.
- No keep-alive, chunked encoding, or HTTPS; every request opens a new connection.
- No authentication; anyone who can reach the port can call it.
- The HTTP layer (`http_server.h/.cpp`) is a small implementation over POSIX sockets, not a vendored library.

## Rule-Based Automation (Prototype)

`pipeline/automation/` matches a person-written rule (a keyword to look for and what to do once found) against one saved OCR result, via fuzzy keyword matching, and prints the resulting KVM action sequence for a person to approve. It never touches an image, only the text and coordinates OCR already produced, so unlike everything else in `pipeline/`, it has no OpenCV/RKNN dependency and isn't gated behind `BUILD_TOOLS` or a cross-compile toolchain. It builds and runs on any machine with a C++17 compiler:

```bash
g++ -std=c++17 -O2 -I. pipeline/automation/*.cpp -o automation_runner
./automation_runner
```

It reads every rule from `rules/` and every saved OCR result from `ocr_output/` (both plain JSON files; see `pipeline/automation/rule.h` and `pipeline/automation/screens.h` for the shapes), and matches each rule against each screen. Nothing is sent to a machine; the resulting sequence is printed and waits for a person to approve it.

This is a prototype of the fuller goal/step design described in `Automation_Pipeline.docx` (Section 2), not a complete implementation of it: a rule here is a flat list of `{keyword, action, value}` steps, checked once against one already-saved screen, rather than the closed loop that design calls for (re-reading the screen via OCR after every action, retrying on a mismatch, and running live against a real KVM connection). Section 2.4 of that document lists what's left to build to close that gap.

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host, for model conversion) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2+ (tested: 2.4.2a2) |
| OpenCV / Clipper / zlib (board) | statically linked, core+imgproc only (see `CMakeLists.txt`) |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board IP (this dev board) | 192.168.1.101, user tpsadmin |
| Board deploy path | `/home/tpsadmin/board_deploy` |
