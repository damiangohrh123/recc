#!/bin/bash
# Sweeps benchmark's detection thresholds (det_thresh, box_thresh,
# unclip_ratio) across a small grid, against every image in testdata/,
# and logs box counts + recognized text + timing for each combination to
# a CSV. Run this on the board, from board_deploy/, after rebuilding
# benchmark with the new CLI args (see recc/pipeline/ocr/benchmark.cpp).
#
# These three values have been hardcoded since the project's first commit
# with no record of ever being tested against alternatives -- this script
# is that test. max_candidates is left at its default (3000); it only
# matters on a very cluttered screen, unlikely to be the bottleneck here.
#
# Usage: ./sweep_det_thresholds.sh
# Output: sweep_results.csv (one row per image x parameter combination)

set -euo pipefail

DET_MODEL="/home/tpsadmin/model/PP-OCRv6_tiny_det_rk3588.rknn"
REC_MODEL="/home/tpsadmin/model/PP-OCRv6_tiny_rec_rk3588.rknn"
CHAR_DICT="/home/tpsadmin/model/ppocr_keys_v6.txt"

IMAGES=(
    "testdata/alarm_1024x768.bgr888"
    "testdata/auto_mode_1_1024x768.bgr888"
    "testdata/auto_mode_2_1024x768.bgr888"
    "testdata/normal_run_1024x768.bgr888"
    "testdata/full_test_1024x384.bgr888"
)

DET_THRESHOLDS=(0.2 0.3 0.4)
BOX_THRESHOLDS=(0.3 0.4 0.5 0.6)
UNCLIP_RATIOS=(1.0 1.5 2.0)

OUT="sweep_results.csv"
echo "image,det_thresh,box_thresh,unclip_ratio,n_boxes,det_ms,recognized_text" > "$OUT"

total=$(( ${#IMAGES[@]} * ${#DET_THRESHOLDS[@]} * ${#BOX_THRESHOLDS[@]} * ${#UNCLIP_RATIOS[@]} ))
count=0

for image in "${IMAGES[@]}"; do
    for det_t in "${DET_THRESHOLDS[@]}"; do
        for box_t in "${BOX_THRESHOLDS[@]}"; do
            for unclip in "${UNCLIP_RATIOS[@]}"; do
                count=$((count + 1))
                echo "[$count/$total] $image  det=$det_t box=$box_t unclip=$unclip" >&2

                # cycles=1, drop_score=0.4 (unchanged), then the three swept values.
                output=$(./benchmark "$DET_MODEL" "$REC_MODEL" "$CHAR_DICT" "$image" \
                    1 0.4 "$det_t" "$box_t" "$unclip" 3000 2>&1)

                n_boxes=$(echo "$output" | grep -c '^\s*\[' || true)
                det_ms=$(echo "$output" | grep "Det      :" | grep -oP '[0-9.]+(?= ms)' || echo "NA")
                # Flatten all recognized text onto one semicolon-separated field.
                # || true like the two lines above: with `set -o pipefail`, an
                # image that produced no recognized text makes grep exit 1 and
                # would abort the whole sweep partway through.
                texts=$(echo "$output" | grep -oP 'text="\K[^"]*' | paste -sd ';' - || true)

                echo "\"$image\",$det_t,$box_t,$unclip,$n_boxes,$det_ms,\"$texts\"" >> "$OUT"
            done
        done
    done
done

echo "Done. Results in $OUT ($total rows)." >&2
