// Runs the OCR + alarm-detection pipeline (TextSystem::run + AlarmDetector)
// on one image N times, timing det/rec/alarm separately, plus CPU, memory,
// and recognized text. Calls the same shared pipeline code ocr_server.cpp
// uses, so there's one implementation of detection/recognition/alarm
// assembly, not a separate copy. With cycles=1 this doubles as a single-shot
// CLI tool: prints the same detection/recognition/alarm output plus
// timing/CPU/memory stats.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>
#include "alarm_detector.h"
#include "ppocr_det.h"
#include "ppocr_rec.h"
#include "ppocr_system.h"
#include "image_io.h"

namespace {

// Prints one box as "(x,y)-(x,y)-(x,y)-(x,y)", preserving the full quad
// shape (useful for tilted/skewed boxes) that a top-left+center summary
// would hide.
void print_box(const Quad& box) {
    for (size_t i = 0; i < box.size(); ++i) {
        printf("%s(%.0f,%.0f)", i ? "-" : "", box[i].x, box[i].y);
    }
}

// Reads process memory/CPU stats from /proc.

// Returns this process's resident memory (RSS) in MB.
double get_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = 0;
            iss >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
}

// Returns {utime, stime} in clock ticks, fields 14 and 15 of
// /proc/self/stat (1-indexed) -- this process's own CPU time.
std::pair<long, long> read_cpu_times() {
    std::ifstream f("/proc/self/stat");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::istringstream iss(content);
    std::vector<std::string> fields;
    std::string tok;
    while (iss >> tok) fields.push_back(tok);
    if (fields.size() < 15) return {0, 0};
    return {std::stol(fields[13]), std::stol(fields[14])};
}

// Sums every field on /proc/stat's "cpu" line: total CPU time across
// all cores since boot.
long read_system_cpu_total() {
    std::ifstream f("/proc/stat");
    std::string cpu_label;
    f >> cpu_label;  // "cpu"
    long total = 0, val = 0;
    while (f >> val) total += val;
    return total;
}

// Arithmetic mean of v; 0 if empty.
double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// Population standard deviation of v; 0 if empty.
double stddev(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = mean(v);
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - m) * (x - m);
    return std::sqrt(sq_sum / v.size());
}

// Per-cycle timing/resource samples, plus the last cycle's OCR + alarm results.
struct CycleTotals {
    std::vector<double> times_ms;     // wall-clock time per cycle (det+rec+alarm+bookkeeping)
    std::vector<double> cpu_pcts;     // CPU% estimate per cycle
    std::vector<double> mem_mbs;      // RSS sampled per cycle
    std::vector<OcrResult> last_res;  // final cycle's OCR results
    AlarmResult last_alarm;           // final cycle's alarm result
    double last_det_ms = 0;           // final cycle's detection time
    double last_rec_ms = 0;           // final cycle's recognition time
    double last_alarm_ms = 0;         // final cycle's alarm-detection time
    int last_n_crops = 0;             // final cycle's crop count
};

// Runs the pipeline `cycles` times via the same TextSystem::run() +
// AlarmDetector::detect() calls ocr_server.cpp uses, timing each stage and
// sampling CPU/memory each cycle.
CycleTotals run_cycles(const cv::Mat& img_orig, const TextSystem& text_system,
                        const AlarmDetector& alarm_detector, int cycles) {
    const unsigned int nproc = std::max(1u, std::thread::hardware_concurrency());

    CycleTotals totals;

    for (int cycle = 0; cycle < cycles; ++cycle) {
        double mem_before = get_rss_mb();
        auto [u0, s0] = read_cpu_times();
        long sys0 = read_system_cpu_total();
        auto wall_start = std::chrono::steady_clock::now();

        RunTiming run_timing;
        std::vector<OcrResult> results = text_system.run(img_orig, &run_timing);

        auto alarm_start = std::chrono::steady_clock::now();
        AlarmResult alarm = alarm_detector.detect(img_orig, results);
        double alarm_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alarm_start).count();

        auto wall_end = std::chrono::steady_clock::now();

        if (cycle == cycles - 1) {
            totals.last_det_ms = run_timing.det_ms;
            totals.last_rec_ms = run_timing.rec_ms;
            totals.last_alarm_ms = alarm_ms;
            totals.last_n_crops = run_timing.n_crops;
            totals.last_res = results;
            totals.last_alarm = alarm;
        }

        double mem_after = get_rss_mb();
        auto [u1, s1] = read_cpu_times();
        long sys1 = read_system_cpu_total();

        double elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        long proc_delta = (u1 - u0) + (s1 - s0);
        long sys_delta = sys1 - sys0;
        double cpu_pct = sys_delta > 0 ? (static_cast<double>(proc_delta) / sys_delta * 100.0 * nproc) : 0.0;

        totals.times_ms.push_back(elapsed_ms);
        totals.cpu_pcts.push_back(cpu_pct);
        totals.mem_mbs.push_back(std::max(mem_before, mem_after));
    }

    return totals;
}

}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <det_model.rknn> <rec_model.rknn> <char_dict.txt> <image.jpg> "
                "[cycles=1] [drop_score=0.4]\n",
                argv[0]);
        return 1;
    }
    const std::string det_model_path = argv[1];
    const std::string rec_model_path = argv[2];
    const std::string char_dict_path = argv[3];
    const std::string image_path = argv[4];
    const int cycles = argc > 5 ? std::atoi(argv[5]) : 1;
    const double drop_score = argc > 6 ? std::atof(argv[6]) : 0.4;

    printf("\n  OCR Benchmark  (%d cycle%s)\n", cycles, cycles == 1 ? "" : "s");
    printf("  Image      : %s\n", image_path.c_str());
    printf("  Det model  : %s\n", det_model_path.c_str());
    printf("  Rec model  : %s\n", rec_model_path.c_str());
    printf("  Drop score : %.2f\n", drop_score);
    printf("--\n");

    auto load_start = std::chrono::steady_clock::now();
    cv::Mat img = load_image(image_path);
    double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    if (img.empty()) {
        return 1;  // load_image already printed the failure reason
    }
    printf("Image load/decode : %.1f ms  (%zu bytes on disk)\n", load_ms,
           static_cast<size_t>(std::ifstream(image_path, std::ios::binary | std::ios::ate).tellg()));

    printf("Loading models (not included in timing)...\n");
    // Same construction as ocr_server.cpp, except drop_score is
    // CLI-configurable here instead of fixed at 0.4 -- it now actually
    // controls what TextSystem::run() returns, rather than being reapplied
    // as a second, separate filter after the fact.
    TextDetector detector(det_model_path, /*target=*/"", /*device_id=*/"",
                          /*det_thresh=*/0.3f, /*box_thresh=*/0.4f,
                          /*unclip_ratio=*/1.5f, /*max_candidates=*/3000);
    if (!detector.is_loaded()) {
        fprintf(stderr, "det model failed to load\n");
        return 1;
    }
    TextRecognizer recognizer(rec_model_path, /*target=*/"", /*device_id=*/"", char_dict_path);
    if (!recognizer.is_loaded()) {
        fprintf(stderr, "rec model failed to load\n");
        return 1;
    }
    TextSystem text_system(std::move(detector), std::move(recognizer),
                            drop_score, /*min_height=*/10.0, /*min_width=*/8.0);
    AlarmDetector alarm_detector;
    printf("Models loaded.\n");

    printf("\nRunning %d cycle%s...\n", cycles, cycles == 1 ? "" : "s");
    CycleTotals totals = run_cycles(img, text_system, alarm_detector, cycles);

    double avg_ms = mean(totals.times_ms);
    double std_ms = stddev(totals.times_ms);
    double avg_cpu = mean(totals.cpu_pcts);
    double avg_mem = mean(totals.mem_mbs);
    double other_ms = avg_ms - totals.last_det_ms - totals.last_rec_ms - totals.last_alarm_ms;

    printf("\n  RESULTS\n");
    printf("--\n");
    printf("  Avg time   : %.1f ms  (std %.1f)\n", avg_ms, std_ms);
    printf("    Det      : %.1f ms\n", totals.last_det_ms);
    printf("    Rec      : %.1f ms  (%d crops)\n", totals.last_rec_ms, totals.last_n_crops);
    printf("    Alarm    : %.1f ms\n", totals.last_alarm_ms);
    printf("    Other    : %.1f ms  (sort, NMS, crop extraction)\n", other_ms);
    printf("  Avg CPU    : %.1f%%\n", avg_cpu);
    printf("  Avg memory : %.1f MB\n", avg_mem);
    printf("  Recognized text:\n");
    if (totals.last_res.empty()) {
        printf("    (none)\n");
    } else {
        for (size_t i = 0; i < totals.last_res.size(); ++i) {
            const auto& r = totals.last_res[i];
            printf("    [%2zu] score=%.3f text=\"%s\"  box=", i, r.rec.score, r.rec.text.c_str());
            print_box(r.box);
            printf("\n");
        }
    }

    printf("\n  Alarm detector:\n");
    if (totals.last_alarm.alarm) {
        printf("    ALARM at (%d,%d) %dx%d", totals.last_alarm.bbox.x, totals.last_alarm.bbox.y,
               totals.last_alarm.bbox.width, totals.last_alarm.bbox.height);
        if (totals.last_alarm.text.has_value()) {
            printf(" text=\"%s\"", totals.last_alarm.text->c_str());
        }
        printf("\n");
    } else {
        printf("    no alarm detected\n");
    }

    return 0;
}
