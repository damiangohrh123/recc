# OCR Pipeline (RK3588 / RECC)

PP-OCRv6 tiny detection and recognition on the Rockchip RK3588 NPU via RKNN, reading text off machine HMI screens, with alarm-banner detection on top of the raw OCR output. This repo covers the production C++ implementation. Input is raw BGR888 only, since uncompressed pixel data preserves small text better.

| Component | Model | Format | Notes |
|-----------|-------|--------|-------|
| Det | PP-OCRv6 tiny det | INT8, 480×480 | ImageNet norm baked in; raw BGR in |
| Rec | PP-OCRv6 tiny rec | FP16, 320×48 | `[0,1]` normalisation |

Detection runs a single full-image pass at the detector's native 480x480 input. Alarm detection converts the frame to HSV, masks for red, and filters by area, aspect ratio, and screen position; if a banner is found, it reuses the OCR text already collected for that region instead of running a second pass. Detection has no preprocessing step (no binarization, no upscaling); the model performs best on unprocessed input.

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
      image_io.cpp/h               # raw .bgr888 file loading, used by benchmark and ocr_server
    automation/                    # Stage 2 prototype: matches a hand-written rule against one
                                    # saved OCR result and prints the resulting KVM action sequence
                                    # (simpler than the goal/step closed loop design, see README below)
  api/                             # Internal OCR API (ocr_server + minimal HTTP layer, see README below)
  third_party/                     # RKNN SDK headers + librknnrt.so
  aarch64-ubuntu20.04-toolchain.tar.gz  # cached aarch64 cross-compile toolchain (gitignored), see README below
  board_deploy/                    # sample images + systemd service, ready to pscp to a board
                                    # (benchmark/ocr_server/automation_runner are gitignored build
                                    # outputs -- build them first, see "Setup" below)
  model/
    PP-OCRv6_tiny_det.onnx         # conversion source
    PP-OCRv6_tiny_rec.onnx         # conversion source
    PP-OCRv6_tiny_det_rk3588.rknn  # runtime model
    PP-OCRv6_tiny_rec_rk3588.rknn  # runtime model
    ppocr_keys_v6.txt              # character dictionary
```

## Setup

### Building the Binaries

`board_deploy/benchmark` and `board_deploy/ocr_server` are aarch64 binaries, cross-compiled for the RK3588 board and statically linked against OpenCV 4.5.4 (core+imgproc only), Clipper/polyclipping, and zlib. The only runtime dependency either needs beyond standard libc/libm/libpthread is `librknnrt.so`, already provided by the board at `/usr/lib`.

These binaries are gitignored, not committed, since rebuilding them on every source change would otherwise bloat git history with binary blobs. A fresh clone of this repo won't have them; build them once using the steps below, then only rebuild again if the C++ source changes.

Rebuilding needs an aarch64 gcc-9 toolchain matching the board's Ubuntu 20.04/glibc 2.31, plus statically-built OpenCV/Clipper/zlib for aarch64. That environment is cached in `aarch64-ubuntu20.04-toolchain.tar.gz` (repo root) so a rebuild skips the ~20+ minute bootstrap:

1. Extract: `tar xzf aarch64-ubuntu20.04-toolchain.tar.gz -C ~` (creates `~/cross`, `~/cross20`, `~/deps-arm64`, `~/rknn-arm64`, `~/fix_toolchain_paths.sh`).
2. Run `bash ~/fix_toolchain_paths.sh` once, right after extracting, to rewrite absolute paths baked into the archive to the current `$HOME`. Safe to re-run.
3. Build wrapper compilers pointing `-B` at `~/cross/usr/aarch64-linux-gnu/bin` (target binutils) and `~/cross20/usr/bin` (gcc-9 frontend); see `~/cross20/wrap/aarch64-linux-gnu-g++` inside the archive for the exact form.
4. Set `LD_LIBRARY_PATH` to include `~/cross20/usr/lib/x86_64-linux-gnu` and `~/cross/usr/lib/x86_64-linux-gnu`.
5. Configure with `-DCMAKE_TOOLCHAIN_FILE=...` using those wrapper compilers, plus `-DOPENCV_INCLUDE_DIR=~/deps-arm64/include/opencv4`, `-DOPENCV_LIB_DIR=~/deps-arm64/lib`, `-DCLIPPER_INCLUDE_DIR=~/deps-arm64/include`, `-DCLIPPER_LIB_DIR=~/deps-arm64/lib`, `-DZLIB_LIB_DIR=~/deps-arm64/lib`, `-DRKNN_SDK_LIB_DIR=~/rknn-arm64/lib`, `-DBUILD_TOOLS=ON`.

```bash
mkdir build && cd build
cmake .. -DOPENCV_INCLUDE_DIR=... -DOPENCV_LIB_DIR=... -DCLIPPER_INCLUDE_DIR=... \
         -DCLIPPER_LIB_DIR=... -DZLIB_LIB_DIR=... -DRKNN_SDK_LIB_DIR=... \
         -DBUILD_TOOLS=ON
make
```

`BUILD_TOOLS=ON` builds both `benchmark` and `ocr_server`; both need `RKNN_SDK_LIB_DIR` pointing at a real `librknnrt.so`, since they link the actual board runtime.

### Deploying to a Board

New board:

1. Copy the binaries and model files over (from Windows/WSL):

   ```bash
   pscp -r board_deploy tpsadmin@<board-ip>:/home/tpsadmin/board_deploy
   pscp -r model tpsadmin@<board-ip>:/home/tpsadmin/model
   ```

   `librknnrt.so` should already be present at `/usr/lib` as part of the board's own OS image.

2. Test manually first, before wiring anything into systemd: run `ocr_server` by hand (see "Running It" below) and confirm `/health` responds, or run `benchmark` against one of the `testdata/*.bgr888` files. This catches a wrong path or missing model file while you're watching it directly, instead of inside a service that silently retries.

3. Install it as a service (see "Running as a Service" below) so it starts automatically on every boot and restarts itself if it crashes.

4. Note this board's IP address wherever it needs to be reachable from (e.g. the Host Application). `ocr_server`, the models, and the service file are identical across every board; only the IP differs.

Existing board, after a rebuild: re-upload just the changed binary and restart the service.

```bash
pscp board_deploy/ocr_server tpsadmin@192.168.1.101:/home/tpsadmin/board_deploy/ocr_server
```

Then on the board: `sudo systemctl restart ocr_server`.

## Running `benchmark`

A one-shot command-line tool: decodes an image, runs detection + recognition + alarm detection, prints results, and (with a cycle count) reports timing/CPU/memory measurement.

```bash
cd ~/board_deploy
./benchmark /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt testdata/alarm_1024x768.bgr888
```

`testdata/` also has `auto_mode_1_1024x768.bgr888`, `auto_mode_2_1024x768.bgr888`, `normal_run_1024x768.bgr888`, and `full_test_1024x384.bgr888`. Add a cycle count for timing/CPU/memory stats, e.g. `... testdata/alarm_1024x768.bgr888 10`.

## Internal OCR API

`ocr_server` (`api/`) loads the detection/recognition models once at startup, then serves OCR and alarm-detection results over HTTP instead of running once per invocation like `benchmark`.

### `GET /health`

Returns `200` with `{"status":"ok"}` once models are loaded.

### `POST /ocr`

Request body: an 8-byte header followed by tightly-packed pixel data, no padding:

```
[4 bytes: width,  big-endian uint32]
[4 bytes: height, big-endian uint32]
[width * height * 3 bytes: BGR888 pixel data, row-major, 3 bytes/pixel]
```

Same byte order kvmd's own raw channel uses for its width/height/size fields, so a client that already speaks that protocol (e.g. `recc_gen5_test_kit/test_ocr_continuous.py`) needs no second convention to talk to this endpoint.

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

### Running It

```bash
cd ~/board_deploy
./ocr_server /home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn /home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn /home/tpsadmin/model/ppocr_keys_v6.txt 8080
```

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8080/health').read())"
```

`/ocr` needs the 8-byte width/height header prepended before the pixel bytes, so it isn't a plain `curl --data-binary @file` call. See `recc_gen5_test_kit/test_ocr_continuous.py` for a working example that builds this request correctly.

### Running as a Service (Survives Reboot)

`board_deploy/ocr_server.service` is a systemd unit that starts `ocr_server` on boot and restarts it automatically if it crashes. `kvmd` has no equivalent yet, it's just backgrounded once by `kvmd_run.sh` with no restart-on-crash supervision; giving it the same systemd treatment is a future improvement.

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

### Current Limitations

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

It reads every rule from `rules/` and every real OCR result from `ocr_output/` (both plain JSON files; see `pipeline/automation/rule.h` and `pipeline/automation/screens.h` for the shapes), and matches each rule against each screen. Nothing is sent to a machine; the resulting sequence is printed and waits for a person to approve it.

This is a prototype of the fuller goal/step design described in `AI_automation_pipeline.docx` (Section 2), not a complete implementation of it: a rule here is a flat list of `{keyword, action, value}` steps, checked once against one already-saved screen, rather than the closed loop that design calls for (re-reading the screen via OCR after every action, retrying on a mismatch, and running live against a real KVM connection). Section 2.4 of that document lists what's left to build to close that gap.

## Environment

| Item | Version |
|------|---------|
| RKNN Toolkit2 (host, for model conversion) | 2.4.2a8 |
| librknnrt (board) | 2.4.2a2+ (tested: 2.4.2a2) |
| OpenCV / Clipper / zlib (board) | statically linked, core+imgproc only (see `CMakeLists.txt`) |
| Board | RK3588, Ubuntu 20.04 aarch64 |
| Board IP (this dev board) | 192.168.1.101, user tpsadmin |
| Board deploy path | `/home/tpsadmin/board_deploy` |
