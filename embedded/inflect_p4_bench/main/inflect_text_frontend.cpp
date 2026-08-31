#include "inflect_text_frontend.h"
#include "inflect_text_normalizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "esp_err.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"

extern "C" {
#include "espeak-ng/espeak_ng.h"
#include "espeak-ng/speak_lib.h"
}

namespace inflect::text {
namespace {

constexpr char kDataMount[] = "/g2p";
constexpr char kDataPartition[] = "g2p";
constexpr char kPunctuation[] = ";:,.!?¡¿—…\"«»“”(){}[]";
constexpr char kTokenLimitError[] = "text exceeds the model token limit";
constexpr size_t kMaximumPlannedSegments = 64;

// This order is part of the released Inflect-Nano-v2 checkpoint contract.
constexpr char kSymbols[] =
    "_"
    ";:,.!?¡¿—…\"«»“” "
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

bool g_initialized = false;
std::vector<uint32_t> g_symbol_codepoints;

bool decode_utf8(std::string_view text, size_t &offset, uint32_t &codepoint)
{
    if (offset >= text.size()) {
        return false;
    }
    const uint8_t first = static_cast<uint8_t>(text[offset++]);
    if (first < 0x80) {
        codepoint = first;
        return true;
    }
    unsigned continuation_count = 0;
    uint32_t value = 0;
    if ((first & 0xe0) == 0xc0) {
        continuation_count = 1;
        value = first & 0x1f;
    } else if ((first & 0xf0) == 0xe0) {
        continuation_count = 2;
        value = first & 0x0f;
    } else if ((first & 0xf8) == 0xf0) {
        continuation_count = 3;
        value = first & 0x07;
    } else {
        return false;
    }
    if (offset + continuation_count > text.size()) {
        return false;
    }
    for (unsigned index = 0; index < continuation_count; ++index) {
        const uint8_t next = static_cast<uint8_t>(text[offset++]);
        if ((next & 0xc0) != 0x80) {
            return false;
        }
        value = (value << 6) | (next & 0x3f);
    }
    codepoint = value;
    return true;
}

void append_utf8(std::string &output, uint32_t codepoint)
{
    if (codepoint < 0x80) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint < 0x10000) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

std::vector<uint32_t> codepoints_of(std::string_view text, bool &valid)
{
    std::vector<uint32_t> output;
    size_t offset = 0;
    uint32_t codepoint = 0;
    valid = true;
    while (offset < text.size()) {
        if (!decode_utf8(text, offset, codepoint)) {
            valid = false;
            output.clear();
            return output;
        }
        output.push_back(codepoint);
    }
    return output;
}

bool contains_codepoint(std::string_view text, uint32_t candidate)
{
    size_t offset = 0;
    uint32_t codepoint = 0;
    while (decode_utf8(text, offset, codepoint)) {
        if (codepoint == candidate) {
            return true;
        }
    }
    return false;
}

void replace_all(std::string &text, std::string_view source, std::string_view target)
{
    if (source.empty()) {
        return;
    }
    size_t offset = 0;
    while ((offset = text.find(source, offset)) != std::string::npos) {
        text.replace(offset, source.size(), target);
        offset += target.size();
    }
}

std::string trim_copy(std::string_view text)
{
    size_t begin = 0;
    while (begin < text.size()
           && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin
           && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string collapse_spaces(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    bool pending_space = false;
    for (const unsigned char byte : text) {
        if (std::isspace(byte)) {
            pending_space = !output.empty();
        } else {
            if (pending_space) {
                output.push_back(' ');
                pending_space = false;
            }
            output.push_back(static_cast<char>(byte));
        }
    }
    return output;
}

bool phonemize_chunk(std::string_view text, std::string &output)
{
    const std::string input = trim_copy(text);
    if (input.empty()) {
        return true;
    }
    const void *cursor = input.c_str();
    bool first = true;
    while (cursor != nullptr) {
        const char *phones = espeak_TextToPhonemes(
            &cursor,
            espeakCHARS_UTF8,
            0x02);
        if (phones == nullptr) {
            return false;
        }
        const std::string part = trim_copy(phones);
        if (!part.empty()) {
            if (!first) {
                output.push_back(' ');
            }
            output.append(part);
            first = false;
        }
    }
    return true;
}

bool phonemize_with_punctuation(std::string_view normalized, std::string &output)
{
    bool valid = false;
    const std::vector<uint32_t> input = codepoints_of(normalized, valid);
    if (!valid) {
        return false;
    }
    std::string chunk;
    bool chunk_had_leading_space = false;
    auto flush_chunk = [&]() -> bool {
        const std::string trimmed = trim_copy(chunk);
        if (!trimmed.empty()) {
            if (chunk_had_leading_space && !output.empty() && output.back() != ' ') {
                output.push_back(' ');
            }
            if (!phonemize_chunk(trimmed, output)) {
                return false;
            }
        }
        chunk.clear();
        chunk_had_leading_space = false;
        return true;
    };

    for (size_t index = 0; index < input.size(); ++index) {
        const uint32_t codepoint = input[index];
        const bool decimal_point = codepoint == '.' && index > 0
            && index + 1 < input.size() && input[index - 1] >= '0'
            && input[index - 1] <= '9' && input[index + 1] >= '0'
            && input[index + 1] <= '9';
        if (!decimal_point && contains_codepoint(kPunctuation, codepoint)) {
            if (!flush_chunk()) {
                return false;
            }
            while (!output.empty() && output.back() == ' ') {
                output.pop_back();
            }
            append_utf8(output, codepoint);
            continue;
        }
        if (codepoint < 128 && std::isspace(static_cast<unsigned char>(codepoint))) {
            if (chunk.empty()) {
                chunk_had_leading_space = true;
            } else if (chunk.back() != ' ') {
                chunk.push_back(' ');
            }
        } else {
            append_utf8(chunk, codepoint);
        }
    }
    if (!flush_chunk()) {
        return false;
    }
    replace_all(output, "sˈæskɐtʃˌuːən", "sɐskˈætʃəwən");
    replace_all(output, "flʊɹɹˈɛsənt", "flʊˈɹɛsənt");
    output = collapse_spaces(output);
    return !output.empty();
}

int symbol_id(uint32_t codepoint)
{
    int found = -1;
    for (size_t index = 0; index < g_symbol_codepoints.size(); ++index) {
        if (g_symbol_codepoints[index] == codepoint) {
            // The released table contains a duplicate apostrophe; Python's
            // dictionary keeps the final occurrence.
            found = static_cast<int>(index);
        }
    }
    return found;
}

bool tokenize_normalized(
    std::string_view normalized,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    std::string &error)
{
    error.clear();
    result = {};
    result.normalized_text.assign(normalized);
    if (result.normalized_text.empty()) {
        error = "text is empty after normalization";
        return false;
    }

    int64_t started = esp_timer_get_time();
    if (!phonemize_with_punctuation(result.normalized_text, result.phonemes)) {
        error = "eSpeak-NG phonemization failed";
        return false;
    }
    result.phonemize_us = esp_timer_get_time() - started;

    started = esp_timer_get_time();
    bool valid = false;
    const std::vector<uint32_t> phones = codepoints_of(result.phonemes, valid);
    if (!valid || phones.empty()) {
        error = "phoneme output is empty or invalid UTF-8";
        return false;
    }
    if (phones.size() * 2 + 1 > maximum_tokens) {
        error = kTokenLimitError;
        return false;
    }
    result.tokens.reserve(phones.size() * 2 + 1);
    result.tokens.push_back(0);
    for (const uint32_t phone : phones) {
        const int id = symbol_id(phone);
        if (id < 0 || static_cast<uint32_t>(id) >= vocabulary_size) {
            error = "phoneme is not present in the Inflect vocabulary";
            result.tokens.clear();
            return false;
        }
        result.tokens.push_back(static_cast<uint16_t>(id));
        result.tokens.push_back(0);
    }
    result.tokenize_us = esp_timer_get_time() - started;
    return true;
}

bool is_boundary_closer(uint32_t codepoint)
{
    switch (codepoint) {
    case '"':
    case ')':
    case ']':
    case '}':
    case 0x00bb:  // right guillemet
    case 0x201d:  // right double quotation mark
        return true;
    default:
        return false;
    }
}

bool is_closing_boundary_at(std::string_view text, size_t offset)
{
    uint32_t codepoint = 0;
    if (!decode_utf8(text, offset, codepoint) || !is_boundary_closer(codepoint)) {
        return false;
    }
    if (codepoint != '"') {
        return true;
    }
    uint32_t following = 0;
    if (!decode_utf8(text, offset, following)) {
        return true;
    }
    if (following < 128) {
        return !std::isalnum(static_cast<unsigned char>(following));
    }
    return contains_codepoint(kPunctuation, following);
}

uint32_t final_boundary_mark(std::string_view text)
{
    bool valid = false;
    const std::vector<uint32_t> codepoints = codepoints_of(text, valid);
    if (!valid) {
        return 0;
    }
    for (size_t reverse = codepoints.size(); reverse > 0; --reverse) {
        const uint32_t codepoint = codepoints[reverse - 1];
        if (
            !is_boundary_closer(codepoint)
            && !(codepoint < 128
                 && std::isspace(static_cast<unsigned char>(codepoint)))) {
            return codepoint;
        }
    }
    return 0;
}

SegmentBoundary classify_boundary(uint32_t mark)
{
    switch (mark) {
    case '.':
    case '!':
    case '?':
        return SegmentBoundary::Sentence;
    case ',':
    case ';':
    case ':':
    case '-':
    case 0x2014:  // em dash
        return SegmentBoundary::Clause;
    default:
        return SegmentBoundary::Whitespace;
    }
}

uint32_t boundary_pause_ms(uint32_t mark)
{
    switch (mark) {
    case '?':
        return 280;
    case '!':
        return 240;
    case '.':
        return 220;
    case ';':
        return 160;
    case ':':
        return 130;
    case ',':
        return 90;
    default:
        return 80;
    }
}

bool is_elidable_boundary_mark(uint32_t codepoint)
{
    switch (codepoint) {
    case '.':
    case '!':
    case '?':
    case ',':
    case ';':
    case ':':
        return true;
    default:
        return false;
    }
}

bool remove_terminal_boundary_mark(std::string &text)
{
    if (text.empty()) {
        return false;
    }
    size_t offset = text.size() - 1;
    while (
        offset > 0
        && (static_cast<uint8_t>(text[offset]) & 0xc0) == 0x80) {
        --offset;
    }
    size_t decoded_end = offset;
    uint32_t codepoint = 0;
    if (
        !decode_utf8(text, decoded_end, codepoint)
        || decoded_end != text.size()
        || !is_elidable_boundary_mark(codepoint)) {
        return false;
    }
    text.erase(offset);
    return !text.empty();
}

bool tokenize_planned_segment(
    std::string_view source_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    bool &boundary_punctuation_elided,
    std::string &error)
{
    boundary_punctuation_elided = false;
    if (tokenize_normalized(
            source_text,
            vocabulary_size,
            maximum_tokens,
            result,
            error)) {
        return true;
    }
    if (error != kTokenLimitError) {
        return false;
    }

    std::string synthesis_text(source_text);
    if (!remove_terminal_boundary_mark(synthesis_text)) {
        return false;
    }
    std::string reduced_error;
    if (!tokenize_normalized(
            synthesis_text,
            vocabulary_size,
            maximum_tokens,
            result,
            reduced_error)) {
        if (reduced_error != kTokenLimitError) {
            error = std::move(reduced_error);
        }
        return false;
    }
    boundary_punctuation_elided = true;
    error.clear();
    return true;
}

struct BoundaryCandidate {
    size_t end = 0;
    size_t next = 0;
    SegmentBoundary boundary = SegmentBoundary::End;
    uint32_t pause_after_ms = 0;
};

std::vector<BoundaryCandidate> collect_boundaries(const std::string &normalized)
{
    std::vector<BoundaryCandidate> boundaries;
    size_t offset = 0;
    while (offset < normalized.size()) {
        if (normalized[offset] != ' ') {
            ++offset;
            continue;
        }
        const size_t end = offset;
        while (offset < normalized.size() && normalized[offset] == ' ') {
            ++offset;
        }
        if (end != 0 && offset < normalized.size()) {
            if (is_closing_boundary_at(normalized, offset)) {
                continue;
            }
            const uint32_t mark = final_boundary_mark(
                std::string_view(normalized).substr(0, end));
            boundaries.push_back({
                .end = end,
                .next = offset,
                .boundary = classify_boundary(mark),
                .pause_after_ms = boundary_pause_ms(mark),
            });
        }
    }
    boundaries.push_back({
        .end = normalized.size(),
        .next = normalized.size(),
        .boundary = SegmentBoundary::End,
        .pause_after_ms = 0,
    });
    return boundaries;
}

}  // namespace

bool initialize(int64_t &elapsed_us, std::string &error)
{
    if (g_initialized) {
        elapsed_us = 0;
        return true;
    }
    const int64_t started = esp_timer_get_time();
    const esp_vfs_fat_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 8,
        .allocation_unit_size = 0,
        .disk_status_check_enable = false,
        .use_one_fat = true,
    };
    const esp_err_t mount_status = esp_vfs_fat_spiflash_mount_ro(
        kDataMount,
        kDataPartition,
        &mount_config);
    if (mount_status != ESP_OK) {
        error = std::string("g2p mount failed: ") + esp_err_to_name(mount_status);
        return false;
    }

    espeak_ng_InitializePath(kDataMount);
    espeak_ng_ERROR_CONTEXT context = nullptr;
    const espeak_ng_STATUS init_status = espeak_ng_Initialize(&context);
    if (init_status != ENS_OK) {
        error = "eSpeak-NG initialization failed";
        espeak_ng_ClearErrorContext(&context);
        return false;
    }
    const espeak_ng_STATUS voice_status = espeak_ng_SetVoiceByName("en-us");
    if (voice_status != ENS_OK) {
        error = "eSpeak-NG en-us voice not found";
        return false;
    }

    bool symbols_valid = false;
    g_symbol_codepoints = codepoints_of(kSymbols, symbols_valid);
    if (!symbols_valid || g_symbol_codepoints.empty()) {
        error = "invalid embedded Inflect symbol table";
        return false;
    }
    elapsed_us = esp_timer_get_time() - started;
    g_initialized = true;
    return true;
}

bool tokenize(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    std::string &error)
{
    if (!g_initialized) {
        error = "text frontend is not initialized";
        return false;
    }
    if (raw_text.empty()) {
        error = "text is empty";
        return false;
    }

    const int64_t started = esp_timer_get_time();
    const std::string normalized = normalizer::normalize(raw_text);
    const int64_t normalize_us = esp_timer_get_time() - started;
    const bool passed = tokenize_normalized(
        normalized, vocabulary_size, maximum_tokens, result, error);
    result.normalize_us = normalize_us;
    return passed;
}

const char *segment_boundary_name(SegmentBoundary boundary)
{
    switch (boundary) {
    case SegmentBoundary::End:
        return "end";
    case SegmentBoundary::Sentence:
        return "sentence";
    case SegmentBoundary::Clause:
        return "clause";
    case SegmentBoundary::Whitespace:
        return "whitespace";
    }
    return "unknown";
}

bool plan_normalized_segments(
    const std::string &normalized_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error)
{
    if (!g_initialized) {
        error = "text frontend is not initialized";
        return false;
    }
    if (normalized_text.empty()) {
        error = "text is empty";
        return false;
    }
    if (maximum_tokens < 3) {
        error = "model token limit is too small";
        return false;
    }

    result = {};
    result.normalized_text = normalized_text;

    const std::vector<BoundaryCandidate> boundaries =
        collect_boundaries(result.normalized_text);
    const int64_t plan_started = esp_timer_get_time();
    size_t begin = 0;
    while (begin < result.normalized_text.size()) {
        if (result.segments.size() >= kMaximumPlannedSegments) {
            error = "text requires too many segments";
            return false;
        }

        size_t target_index = boundaries.size();
        for (size_t index = 0; index < boundaries.size(); ++index) {
            if (boundaries[index].end <= begin) {
                continue;
            }
            if (
                boundaries[index].boundary == SegmentBoundary::Sentence
                || boundaries[index].boundary == SegmentBoundary::End) {
                target_index = index;
                break;
            }
        }
        if (target_index == boundaries.size()) {
            error = "could not find a segment boundary";
            return false;
        }

        const BoundaryCandidate *chosen = &boundaries[target_index];
        FrontendResult chosen_frontend;
        bool chosen_boundary_punctuation_elided = false;
        const std::string_view target_text(result.normalized_text.data() + begin,
                                           chosen->end - begin);
        if (!tokenize_planned_segment(
                target_text,
                vocabulary_size,
                maximum_tokens,
                chosen_frontend,
                chosen_boundary_punctuation_elided,
                error)) {
            if (error != kTokenLimitError) {
                return false;
            }

            const BoundaryCandidate *furthest = nullptr;
            const BoundaryCandidate *preferred_clause = nullptr;
            FrontendResult furthest_frontend;
            FrontendResult preferred_clause_frontend;
            bool furthest_boundary_punctuation_elided = false;
            bool preferred_clause_boundary_punctuation_elided = false;
            for (size_t index = 0; index < target_index; ++index) {
                const BoundaryCandidate &candidate = boundaries[index];
                if (candidate.end <= begin) {
                    continue;
                }
                FrontendResult candidate_frontend;
                bool candidate_boundary_punctuation_elided = false;
                std::string candidate_error;
                const std::string_view candidate_text(
                    result.normalized_text.data() + begin,
                    candidate.end - begin);
                if (!tokenize_planned_segment(
                        candidate_text,
                        vocabulary_size,
                        maximum_tokens,
                        candidate_frontend,
                        candidate_boundary_punctuation_elided,
                        candidate_error)) {
                    if (candidate_error != kTokenLimitError) {
                        error = std::move(candidate_error);
                        return false;
                    }
                    continue;
                }
                furthest = &candidate;
                furthest_frontend = std::move(candidate_frontend);
                furthest_boundary_punctuation_elided =
                    candidate_boundary_punctuation_elided;
                if (
                    candidate.boundary == SegmentBoundary::Clause
                    && furthest_frontend.tokens.size()
                        >= (maximum_tokens + 1) / 2) {
                    preferred_clause = &candidate;
                    preferred_clause_frontend = furthest_frontend;
                    preferred_clause_boundary_punctuation_elided =
                        furthest_boundary_punctuation_elided;
                }
            }
            if (furthest == nullptr) {
                error = "unsplittable_word exceeds the model token limit";
                return false;
            }
            if (preferred_clause != nullptr) {
                chosen = preferred_clause;
                chosen_frontend = std::move(preferred_clause_frontend);
                chosen_boundary_punctuation_elided =
                    preferred_clause_boundary_punctuation_elided;
            } else {
                chosen = furthest;
                chosen_frontend = std::move(furthest_frontend);
                chosen_boundary_punctuation_elided =
                    furthest_boundary_punctuation_elided;
            }
        }

        result.segments.push_back({
            .source_text = std::string(
                result.normalized_text.data() + begin,
                chosen->end - begin),
            .frontend = std::move(chosen_frontend),
            .boundary = chosen->boundary,
            .pause_after_ms = chosen->pause_after_ms,
            .boundary_punctuation_elided =
                chosen_boundary_punctuation_elided,
        });
        begin = chosen->next;
    }
    result.plan_us = esp_timer_get_time() - plan_started;
    error.clear();
    return true;
}

bool plan_segments(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error)
{
    if (!g_initialized) {
        error = "text frontend is not initialized";
        return false;
    }
    if (raw_text.empty()) {
        error = "text is empty";
        return false;
    }

    const int64_t started = esp_timer_get_time();
    const std::string normalized_text = normalizer::normalize(raw_text);
    const int64_t normalize_us = esp_timer_get_time() - started;
    if (normalized_text.empty()) {
        error = "text is empty after normalization";
        return false;
    }
    if (!plan_normalized_segments(
            normalized_text,
            vocabulary_size,
            maximum_tokens,
            result,
            error)) {
        return false;
    }
    result.normalize_us = normalize_us;
    return true;
}

}  // namespace inflect::text
