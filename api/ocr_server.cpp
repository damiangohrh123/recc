// Entry point for the Internal OCR API server: loads the det/rec models
// once, then serves OCR + alarm-detection requests over HTTP until killed.
// See the "Usage > ocr_server (HTTP API)" section of the top-level
// README.md for the request/response contract this implements.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include "pipeline/ocr/alarm_detector.h"
#include "api/http_server.h"
#include "api/json_write.h"
#include "pipeline/ocr/ppocr_det.h"
#include "pipeline/ocr/ppocr_rec.h"
#include "pipeline/ocr/ppocr_system.h"

namespace {

std::string box_to_json(const Quad& box) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < box.size(); ++i) {
        if (i) out << ",";
        out << "[" << box[i].x << "," << box[i].y << "]";
    }
    out << "]";
    return out.str();
}

std::string results_to_json(const std::vector<OcrResult>& results, const AlarmResult& alarm,
                             double timing_ms) {
    std::ostringstream out;
    out << "{\"boxes\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) out << ",";
        out << "{\"text\":\"" << json_escape(r.rec.text) << "\","
            << "\"score\":" << r.rec.score << ","
            << "\"box\":" << box_to_json(r.box) << "}";
    }
    out << "],\"alarm\":{\"detected\":" << (alarm.alarm ? "true" : "false");
    if (alarm.alarm) {
        out << ",\"bbox\":[" << alarm.bbox.x << "," << alarm.bbox.y << ","
            << alarm.bbox.width << "," << alarm.bbox.height << "]";
        if (alarm.text.has_value()) {
            out << ",\"text\":\"" << json_escape(*alarm.text) << "\"";
        }
    }
    out << "}," << "\"timing_ms\":" << timing_ms << "}";
    return out.str();
}

// Runs the OCR + alarm-detection pipeline on an already-decoded BGR cv::Mat
// and times it. Only caller is /ocr, which builds the Mat directly over
// raw BGR888 bytes -- no JPEG/PNG decode step anywhere in this server, raw
// BGR888 is the only input format the pipeline accepts.
HttpResponse run_ocr(TextSystem& text_system, AlarmDetector& alarm_detector, const cv::Mat& img) {
    HttpResponse res;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<OcrResult> results = text_system.run(img);
    AlarmResult alarm = alarm_detector.detect(img, results);
    auto t1 = std::chrono::steady_clock::now();
    double timing_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    res.body = results_to_json(results, alarm, timing_ms);
    return res;
}

// Reads a big-endian uint32 out of a 4-byte buffer, matching the byte order
// kvmd's own raw channel already uses for its width/height/size fields (see
// recc_gen5_test_kit/test_ocr_continuous.py, which builds this same request),
// so a client that already speaks kvmd's raw protocol doesn't need a second
// convention.
uint32_t read_u32_be(const unsigned char* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <det_model.rknn> <rec_model.rknn> <char_dict.txt> <port>\n", argv[0]);
        return 1;
    }
    const std::string det_model_path = argv[1];
    const std::string rec_model_path = argv[2];
    const std::string char_dict_path = argv[3];
    const int port = std::atoi(argv[4]);

    printf("loading models...\n");
    // Empty target/device_id runs inference on this board's own NPU, same
    // as benchmark.cpp.
    // det_thresh lowered from 0.3 to 0.2: a 36-combination sweep against
    // 394 ground-truth fields across 5 real machine screens (see
    // documentation/Automation_Pipeline.docx, Appendix C) found 0.2 the
    // most accurate value (82.5% vs 80.7% at 0.3), at no cost to speed.
    TextDetector detector(det_model_path, /*target=*/"", /*device_id=*/"",
        /*det_thresh=*/0.2f, /*box_thresh=*/0.4f,
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
        /*drop_score=*/0.4, /*min_height=*/10.0, /*min_width=*/8.0);
    AlarmDetector alarm_detector;

    printf("models loaded, serving on port %d\n", port);

    HttpServer server(port);

    server.on("GET", "/health", [](const HttpRequest&) {
        HttpResponse res;
        res.body = "{\"status\":\"ok\"}";
        return res;
    });

    // Raw BGR888 input only -- no JPEG/PNG decode step anywhere in this
    // server (JPEG artifacts can distort small text -- see image_io.h).
    // Body layout: 4-byte big-endian width, 4-byte big-endian height, then
    // exactly width*height*3 bytes of tightly-packed BGR888 (no padding, no
    // per-row stride gap -- same layout kvmd's raw channel already
    // produces).
    server.on("POST", "/ocr", [&](const HttpRequest& req) {
        if (req.body.size() < 8) {
            HttpResponse res;
            res.status = 400;
            res.body = "{\"error\":\"body too short for width/height header\"}";
            return res;
        }
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(req.body.data());
        const uint32_t width = read_u32_be(bytes);
        const uint32_t height = read_u32_be(bytes + 4);
        // Sanity cap well above any real capture resolution, so a garbled
        // header can't make the size_t multiplication below wrap around
        // and slip past the body-size check.
        constexpr uint32_t kMaxDimension = 16384;

        if (width == 0 || height == 0 || width > kMaxDimension || height > kMaxDimension) {
            HttpResponse res;
            res.status = 400;
            res.body = "{\"error\":\"width/height missing or out of range\"}";
            return res;
        }

        const size_t expected_size = size_t(8) + size_t(width) * size_t(height) * 3;
        if (req.body.size() != expected_size) {
            HttpResponse res;
            res.status = 400;
            std::ostringstream msg;
            msg << "{\"error\":\"body size " << req.body.size() << " does not match 8 + width*height*3 "
                << "for width=" << width << " height=" << height << " (expected " << expected_size << ")\"}";
            res.status = 400;
            res.body = msg.str();
            return res;
        }

        // Own a copy of the pixel bytes: req.body (and therefore `bytes`)
        // only lives for the duration of this handler call, but cv::Mat's
        // pointer constructor doesn't copy by default.
        cv::Mat img(int(height), int(width), CV_8UC3, const_cast<unsigned char*>(bytes + 8));
        img = img.clone();
        return run_ocr(text_system, alarm_detector, img);
    });

    server.run();
    return 0;
}
