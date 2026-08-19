#include "json_value.h"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text), i_(0) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        return v;
    }

private:
    const std::string& s_;
    size_t i_;

    char peek() const {
        if (i_ >= s_.size()) throw std::runtime_error("json: unexpected end of input");
        return s_[i_];
    }

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }

    bool consume(char c) {
        skip_ws();
        if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) {
            throw std::runtime_error(std::string("json: expected '") + c + "' at position " + std::to_string(i_));
        }
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonType::Object;
        expect('{');
        skip_ws();
        if (consume('}')) return v;
        while (true) {
            skip_ws();
            std::string key = parse_raw_string();
            expect(':');
            JsonValue val = parse_value();
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (consume(',')) continue;
            expect('}');
            break;
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonType::Array;
        expect('[');
        skip_ws();
        if (consume(']')) return v;
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            if (consume(',')) continue;
            expect(']');
            break;
        }
        return v;
    }

    JsonValue parse_string_value() {
        JsonValue v;
        v.type = JsonType::String;
        v.str = parse_raw_string();
        return v;
    }

    std::string parse_raw_string() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) throw std::runtime_error("json: unterminated string");
            char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                if (i_ >= s_.size()) throw std::runtime_error("json: unterminated escape");
                char e = s_[i_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) throw std::runtime_error("json: bad \\u escape");
                        std::string hex = s_.substr(i_, 4);
                        i_ += 4;
                        unsigned int code = static_cast<unsigned int>(std::strtoul(hex.c_str(), nullptr, 16));
                        // Minimal UTF-8 encoding; surrogate pairs not handled,
                        // not needed for this pipeline's text (OCR labels).
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("json: unknown escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    JsonValue parse_number() {
        size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) ||
                                   s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                                   s_[i_] == '-' || s_[i_] == '+')) {
            ++i_;
        }
        if (i_ == start) throw std::runtime_error("json: expected number");
        JsonValue v;
        v.type = JsonType::Number;
        v.num = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonType::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v.b = true; i_ += 4; return v; }
        if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; return v; }
        throw std::runtime_error("json: expected true/false");
    }

    JsonValue parse_null() {
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return JsonValue(); }
        throw std::runtime_error("json: expected null");
    }
};

// Only json_parse_file is public; parsing from a string is an implementation
// detail of it.
JsonValue json_parse(const std::string& text) {
    Parser p(text);
    return p.parse();
}

}  // namespace

JsonValue json_parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("json: could not open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return json_parse(ss.str());
}
