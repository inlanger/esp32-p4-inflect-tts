#pragma once

#include <cstdint>
#include <string>

#include "inflect_text_frontend.h"

namespace inflect::text {

bool initialize_in_worker(int64_t &elapsed_us, std::string &error);

bool tokenize_in_worker(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    std::string &error);

bool plan_segments_in_worker(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error);

bool plan_normalized_segments_in_worker(
    const std::string &normalized_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error);

uint32_t last_worker_stack_high_water_bytes();

}  // namespace inflect::text
