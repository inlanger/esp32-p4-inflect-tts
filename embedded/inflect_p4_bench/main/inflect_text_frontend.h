#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inflect::text {

struct FrontendResult {
    std::string normalized_text;
    std::string phonemes;
    std::vector<uint16_t> tokens;
    int64_t normalize_us = 0;
    int64_t phonemize_us = 0;
    int64_t tokenize_us = 0;
};

enum class SegmentBoundary {
    End,
    Sentence,
    Clause,
    Whitespace,
};

struct FrontendSegment {
    std::string source_text;
    FrontendResult frontend;
    SegmentBoundary boundary = SegmentBoundary::End;
    uint32_t pause_after_ms = 0;
    bool boundary_punctuation_elided = false;
};

struct SegmentedFrontendResult {
    std::string normalized_text;
    std::vector<FrontendSegment> segments;
    int64_t normalize_us = 0;
    int64_t plan_us = 0;
};

const char *segment_boundary_name(SegmentBoundary boundary);

bool initialize(int64_t &elapsed_us, std::string &error);

bool tokenize(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    std::string &error);

bool plan_segments(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error);

bool plan_normalized_segments(
    const std::string &normalized_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error);

}  // namespace inflect::text
