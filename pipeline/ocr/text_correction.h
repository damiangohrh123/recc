#pragma once
#include <string>

// Narrow, hand-rolled corrections for two recurring recognition mistakes
// documented from real testing (see documentation/OCR_Recognition_Fine_Tuning_Plan.docx,
// Background section): decimal points misread as dashes, and "l"/"I"/"1"
// confusion against a small fixed vocabulary of this project's actual UI
// labels. Deliberately narrow, this only rewrites text matching one of
// those two specific, well-evidenced failure modes, not general
// spell-checking. Pure string processing, no OpenCV dependency, so it's
// independent of the rest of the OCR pipeline and easy to test on its own.
std::string correct_known_ocr_errors(const std::string& text);
