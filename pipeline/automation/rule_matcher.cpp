#include "rule_matcher.h"
#include <algorithm>
#include <unordered_map>

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    return out;
}

struct Match {
    int a = 0;
    int b = 0;
    int size = 0;
};

// Longest contiguous run common to a[alo,ahi) and b[blo,bhi). Same
// dynamic-programming approach difflib.SequenceMatcher.find_longest_match
// uses internally.
Match find_longest_match(const std::string& a, int alo, int ahi,
                          const std::string& b, int blo, int bhi) {
    Match best{alo, blo, 0};
    std::unordered_map<int, int> j2len;
    for (int i = alo; i < ahi; ++i) {
        std::unordered_map<int, int> new_j2len;
        for (int j = blo; j < bhi; ++j) {
            if (a[i] != b[j]) continue;
            int k = 1;
            auto it = j2len.find(j - 1);
            if (it != j2len.end()) k = it->second + 1;
            new_j2len[j] = k;
            if (k > best.size) {
                best.a = i - k + 1;
                best.b = j - k + 1;
                best.size = k;
            }
        }
        j2len = std::move(new_j2len);
    }
    return best;
}

// Total matched characters between a[alo,ahi) and b[blo,bhi), recursing
// into the unmatched left/right remainders around each longest match.
int count_matches(const std::string& a, int alo, int ahi,
                   const std::string& b, int blo, int bhi) {
    if (alo >= ahi || blo >= bhi) return 0;
    Match m = find_longest_match(a, alo, ahi, b, blo, bhi);
    if (m.size == 0) return 0;
    int total = m.size;
    total += count_matches(a, alo, m.a, b, blo, m.b);
    total += count_matches(a, m.a + m.size, ahi, b, m.b + m.size, bhi);
    return total;
}

// Ratcliff/Obershelp similarity ratio, in [0, 1] -- the same algorithm
// Python's difflib.SequenceMatcher.ratio() uses (minus the "autojunk"
// heuristic, which only applies to sequences of 200+ elements and is
// irrelevant for short UI labels). Only find_by_keyword uses it.
double sequence_ratio(const std::string& a, const std::string& b) {
    int total_len = static_cast<int>(a.size() + b.size());
    if (total_len == 0) return 1.0;
    int matches = count_matches(a, 0, static_cast<int>(a.size()), b, 0, static_cast<int>(b.size()));
    return 2.0 * matches / total_len;
}

}  // namespace

std::pair<const DetectedBox*, double> find_by_keyword(const std::string& keyword,
                                                       const std::vector<DetectedBox>& boxes) {
    const DetectedBox* best = nullptr;
    double best_score = 0.0;
    std::string kw_lower = to_lower(keyword);
    for (const auto& b : boxes) {
        std::string text_lower = to_lower(b.text);
        double score = sequence_ratio(kw_lower, text_lower);
        if (text_lower.find(kw_lower) != std::string::npos) score += 0.3;
        if (score > best_score) {
            best_score = score;
            best = &b;
        }
    }
    return {best, best_score};
}

std::pair<int, int> box_center(const std::array<std::array<int, 2>, 4>& box) {
    int sx = 0, sy = 0;
    for (const auto& p : box) {
        sx += p[0];
        sy += p[1];
    }
    return {sx / 4, sy / 4};
}

std::pair<int, int> find_field_near_label(const DetectedBox& label_box,
                                           const std::vector<DetectedBox>& boxes) {
    int lx1 = label_box.box[1][0];
    int ly_center = (label_box.box[0][1] + label_box.box[2][1]) / 2;
    int candidate_x = lx1 + 20;
    for (const auto& b : boxes) {
        if (&b == &label_box) continue;
        int bx0 = b.box[0][0];
        int bx1 = b.box[1][0];
        int by0 = b.box[0][1];
        int by1 = b.box[2][1];
        if (bx0 <= candidate_x && candidate_x <= bx1 && by0 <= ly_center && ly_center <= by1) {
            candidate_x = bx1 + 20;
        }
    }
    return {candidate_x, ly_center};
}
