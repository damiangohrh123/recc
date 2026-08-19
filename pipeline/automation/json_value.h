#pragma once
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>

// Minimal JSON reader for this automation component. Not general-purpose,
// just enough to read the one shape this pipeline actually deals with in
// C++: a real POST /ocr response (an object with a "boxes" array).
// Mirrors the project's existing philosophy in api/json_write.h --
// hand-rolled, scoped to exactly what's needed, no third-party dependency.

enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    // Looks up a key in an Object value; returns nullptr if absent or if
    // this value isn't an Object.
    const JsonValue* find(const std::string& key) const {
        if (type != JsonType::Object) return nullptr;
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

// Reads a whole file and parses it as JSON. Throws std::runtime_error if
// the file can't be opened or doesn't parse.
JsonValue json_parse_file(const std::string& path);
