#include "inflect_text_normalizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace inflect::text::normalizer {
namespace {

struct Replacement {
    std::string_view source;
    std::string_view target;
};

constexpr std::array<Replacement, 10> kWordOverrides = {{
    {"Qwen3", "Qwen three"},
    {"Qwen", "Qwen"},
    {"PyTorch", "pie torch"},
    {"SQLite", "ess cue lite"},
    {"USB-C", "you ess bee see"},
    {"RTX 3060", "ar tee ex thirty sixty"},
    {"RTX 3090", "ar tee ex thirty ninety"},
    {"RTX 4090", "ar tee ex forty ninety"},
    {"RTX 5080", "ar tee ex fifty eighty"},
    {"RTX 5090", "ar tee ex fifty ninety"},
}};

constexpr std::array<Replacement, 10> kAbbreviations = {{
    {"Dr.", "doctor"},
    {"Mr.", "mister"},
    {"Mrs.", "missus"},
    {"Ms.", "miss"},
    {"Prof.", "professor"},
    {"St.", "saint"},
    {"vs.", "versus"},
    {"etc.", "et cetera"},
    {"e.g.", "for example"},
    {"i.e.", "that is"},
}};

constexpr std::array<std::string_view, 12> kMonths = {{
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}};

constexpr std::array<std::string_view, 20> kSmallNumbers = {{
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
}};

constexpr std::array<std::string_view, 10> kTens = {{
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
}};

constexpr std::array<std::string_view, 7> kScales = {{
    "", "thousand", "million", "billion", "trillion", "quadrillion",
    "quintillion",
}};

constexpr std::array<std::string_view, 13> kLabels = {{
    "apartment", "apt", "suite", "unit", "room", "flight", "extension",
    "order", "invoice", "locker", "aisle", "gate", "",
}};

bool ascii_is_word(char value)
{
    const unsigned char byte = static_cast<unsigned char>(value);
    return std::isalnum(byte) || value == '_';
}

bool ascii_is_digit(char value)
{
    return value >= '0' && value <= '9';
}

bool ascii_is_alpha(char value)
{
    const unsigned char byte = static_cast<unsigned char>(value);
    return std::isalpha(byte);
}

bool ascii_is_upper(char value)
{
    return value >= 'A' && value <= 'Z';
}

bool boundary_before(std::string_view text, size_t offset)
{
    return offset == 0 || !ascii_is_word(text[offset - 1]);
}

bool boundary_after(std::string_view text, size_t offset)
{
    return offset >= text.size() || !ascii_is_word(text[offset]);
}

bool word_boundary_at(std::string_view text, size_t offset)
{
    const bool before_is_word = offset > 0 && ascii_is_word(text[offset - 1]);
    const bool after_is_word = offset < text.size() && ascii_is_word(text[offset]);
    return before_is_word != after_is_word;
}

bool equals_ci(char left, char right)
{
    return std::tolower(static_cast<unsigned char>(left))
        == std::tolower(static_cast<unsigned char>(right));
}

bool match_at_ci(std::string_view text, size_t offset, std::string_view value)
{
    if (offset + value.size() > text.size()) {
        return false;
    }
    for (size_t index = 0; index < value.size(); ++index) {
        if (!equals_ci(text[offset + index], value[index])) {
            return false;
        }
    }
    return true;
}

void replace_all(std::string &text, std::string_view source, std::string_view target)
{
    size_t offset = 0;
    while ((offset = text.find(source, offset)) != std::string::npos) {
        text.replace(offset, source.size(), target);
        offset += target.size();
    }
}

std::string collapse_spaces(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    bool pending_space = false;
    for (const unsigned char byte : text) {
        if (std::isspace(byte)) {
            pending_space = !output.empty();
            continue;
        }
        if (pending_space) {
            output.push_back(' ');
            pending_space = false;
        }
        output.push_back(static_cast<char>(byte));
    }
    return output;
}

std::string replace_bounded(
    std::string text,
    std::string_view source,
    std::string_view target,
    bool case_insensitive,
    bool require_end_boundary)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        const bool matches = boundary_before(text, offset)
            && (case_insensitive ? match_at_ci(text, offset, source)
                                 : text.compare(offset, source.size(), source) == 0)
            && (!require_end_boundary
                || boundary_after(text, offset + source.size()));
        if (matches) {
            output.append(target);
            offset += source.size();
        } else {
            output.push_back(text[offset++]);
        }
    }
    return output;
}

std::string below_thousand(uint16_t value)
{
    std::string output;
    if (value >= 100) {
        output.append(kSmallNumbers[value / 100]);
        output.append(" hundred");
        value %= 100;
        if (value != 0) {
            output.append(" and ");
        }
    }
    if (value >= 20) {
        output.append(kTens[value / 10]);
        value %= 10;
        if (value != 0) {
            output.push_back(' ');
            output.append(kSmallNumbers[value]);
        }
    } else if (value != 0 || output.empty()) {
        output.append(kSmallNumbers[value]);
    }
    return output;
}

std::string number_words(uint64_t value)
{
    if (value == 0) {
        return "zero";
    }
    std::vector<uint16_t> groups;
    while (value != 0) {
        groups.push_back(static_cast<uint16_t>(value % 1000));
        value /= 1000;
    }

    std::string output;
    for (size_t reverse = groups.size(); reverse > 0; --reverse) {
        const size_t scale = reverse - 1;
        const uint16_t group = groups[scale];
        if (group == 0) {
            continue;
        }
        if (!output.empty()) {
            if (scale == 0 && group < 100) {
                output.append(" and ");
            } else {
                output.push_back(' ');
            }
        }
        output.append(below_thousand(group));
        if (scale != 0) {
            output.push_back(' ');
            output.append(kScales[scale]);
        }
    }
    return output;
}

std::string ordinalize_last_word(std::string cardinal)
{
    constexpr std::array<Replacement, 19> irregular = {{
        {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"four", "fourth"}, {"five", "fifth"}, {"six", "sixth"},
        {"seven", "seventh"}, {"eight", "eighth"}, {"nine", "ninth"},
        {"ten", "tenth"}, {"eleven", "eleventh"}, {"twelve", "twelfth"},
        {"thirteen", "thirteenth"}, {"fourteen", "fourteenth"},
        {"fifteen", "fifteenth"}, {"sixteen", "sixteenth"},
        {"seventeen", "seventeenth"}, {"eighteen", "eighteenth"},
        {"nineteen", "nineteenth"},
    }};
    const size_t start = cardinal.rfind(' ') == std::string::npos
        ? 0
        : cardinal.rfind(' ') + 1;
    const std::string last = cardinal.substr(start);
    for (const Replacement &entry : irregular) {
        if (last == entry.source) {
            cardinal.replace(start, last.size(), entry.target);
            return cardinal;
        }
    }
    if (last.size() >= 2 && last.ends_with("ty")) {
        cardinal.replace(start + last.size() - 1, 1, "ieth");
    } else {
        cardinal.append("th");
    }
    return cardinal;
}

std::string number_words_ordinal(uint64_t value)
{
    if (value == 0) {
        return "zeroth";
    }
    return ordinalize_last_word(number_words(value));
}

bool parse_uint64(std::string_view digits, uint64_t &value)
{
    value = 0;
    bool found = false;
    for (const char character : digits) {
        if (character == ',') {
            continue;
        }
        if (!ascii_is_digit(character)) {
            return false;
        }
        found = true;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    return found;
}

std::string digit_words(std::string_view digits, bool identifier = false)
{
    std::string output;
    size_t digit_index = 0;
    for (const char character : digits) {
        if (!ascii_is_digit(character)) {
            continue;
        }
        if (!output.empty()) {
            output.push_back(' ');
        }
        if (identifier && character == '0' && digit_index > 0) {
            output.append("oh");
        } else {
            output.append(kSmallNumbers[character - '0']);
        }
        ++digit_index;
    }
    return output;
}

std::string letter_name(char letter)
{
    constexpr std::array<std::string_view, 26> names = {{
        "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye",
        "jay", "kay", "ell", "em", "en", "oh", "pee", "cue", "ar",
        "ess", "tee", "you", "vee", "double you", "ex", "why", "zee",
    }};
    const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(letter)));
    return std::string(names[upper - 'A']);
}

std::string expand_identifier(std::string_view token)
{
    size_t begin = 0;
    size_t end = token.size();
    std::string output;
    if (begin < end && ascii_is_alpha(token[begin])) {
        output.append(letter_name(token[begin++]));
    }
    const size_t digit_begin = begin;
    while (begin < end && ascii_is_digit(token[begin])) {
        ++begin;
    }
    const std::string_view digits = token.substr(digit_begin, begin - digit_begin);
    if (!output.empty()) {
        output.push_back(' ');
    }
    if (digits.size() == 3 || (!digits.empty() && digits.front() == '0')) {
        output.append(digit_words(digits, true));
    } else {
        uint64_t value = 0;
        output.append(parse_uint64(digits, value) ? number_words(value)
                                                  : digit_words(digits));
    }
    if (begin < end && ascii_is_alpha(token[begin])) {
        output.push_back(' ');
        output.append(letter_name(token[begin]));
    }
    return output;
}

std::string expand_dotted_acronyms(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_upper(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t cursor = offset;
        std::string letters;
        while (cursor + 1 < text.size() && ascii_is_upper(text[cursor])
               && text[cursor + 1] == '.') {
            letters.push_back(text[cursor]);
            cursor += 2;
        }
        if (letters.size() < 2) {
            output.push_back(text[offset++]);
            continue;
        }
        for (size_t index = 0; index < letters.size(); ++index) {
            if (index != 0) {
                output.push_back(' ');
            }
            output.push_back(letters[index]);
        }
        offset = cursor;
    }
    return output;
}

bool parse_identifier_at(std::string_view text, size_t offset, size_t &end)
{
    size_t cursor = offset;
    if (cursor < text.size() && ascii_is_alpha(text[cursor])) {
        ++cursor;
    }
    const size_t digit_begin = cursor;
    while (cursor < text.size() && ascii_is_digit(text[cursor])
           && cursor - digit_begin < 4) {
        ++cursor;
    }
    const size_t digit_count = cursor - digit_begin;
    if (digit_count == 0 || (cursor < text.size() && ascii_is_digit(text[cursor]))) {
        return false;
    }
    if (cursor < text.size() && ascii_is_alpha(text[cursor])) {
        ++cursor;
    }
    if (!boundary_after(text, cursor)) {
        return false;
    }
    end = cursor;
    return true;
}

std::string expand_labeled_identifiers(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        bool replaced = false;
        if (boundary_before(text, offset)) {
            for (const std::string_view label : kLabels) {
                if (label.empty() || !match_at_ci(text, offset, label)) {
                    continue;
                }
                size_t label_end = offset + label.size();
                if (label == "apt" && label_end < text.size()
                    && text[label_end] == '.') {
                    ++label_end;
                }
                if (!boundary_after(text, label_end)) {
                    continue;
                }
                size_t token_begin = label_end;
                while (token_begin < text.size()
                       && std::isspace(static_cast<unsigned char>(text[token_begin]))) {
                    ++token_begin;
                }
                if (token_begin == label_end) {
                    continue;
                }
                size_t token_end = token_begin;
                if (!parse_identifier_at(text, token_begin, token_end)) {
                    continue;
                }
                output.append(text.substr(offset, label_end - offset));
                output.push_back(' ');
                output.append(expand_identifier(text.substr(token_begin, token_end - token_begin)));
                offset = token_end;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            output.push_back(text[offset++]);
        }
    }
    return output;
}

bool is_direction_at(std::string_view text, size_t offset)
{
    constexpr std::array<std::string_view, 4> directions = {{
        "North", "South", "East", "West",
    }};
    for (const std::string_view direction : directions) {
        if (match_at_ci(text, offset, direction)
            && boundary_after(text, offset + direction.size())) {
            return true;
        }
    }
    return false;
}

std::string expand_street_numbers(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (boundary_before(text, offset) && offset + 3 <= text.size()
            && ascii_is_digit(text[offset]) && ascii_is_digit(text[offset + 1])
            && ascii_is_digit(text[offset + 2])
            && (offset + 3 == text.size() || !ascii_is_digit(text[offset + 3]))) {
            size_t direction = offset + 3;
            while (direction < text.size()
                   && std::isspace(static_cast<unsigned char>(text[direction]))) {
                ++direction;
            }
            if (direction > offset + 3 && is_direction_at(text, direction)) {
                output.append(digit_words(text.substr(offset, 3), true));
                offset += 3;
                continue;
            }
        }
        output.push_back(text[offset++]);
    }
    return output;
}

std::string expand_money(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (text[offset] != '$' || offset + 1 >= text.size()
            || !ascii_is_digit(text[offset + 1])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t cursor = offset + 1;
        while (cursor < text.size()
               && (ascii_is_digit(text[cursor]) || text[cursor] == ',')) {
            ++cursor;
        }
        const size_t dollars_end = cursor;
        size_t cents_begin = cursor;
        size_t cents_end = cursor;
        if (cursor + 1 < text.size() && text[cursor] == '.'
            && ascii_is_digit(text[cursor + 1])) {
            cents_begin = cursor + 1;
            cents_end = cents_begin;
            while (cents_end < text.size() && ascii_is_digit(text[cents_end])
                   && cents_end - cents_begin < 2) {
                ++cents_end;
            }
            cursor = cents_end;
        }
        uint64_t dollars = 0;
        const std::string_view dollar_digits = text.substr(
            offset + 1, dollars_end - offset - 1);
        output.append(parse_uint64(dollar_digits, dollars)
                          ? number_words(dollars)
                          : digit_words(dollar_digits));
        output.append(dollars == 1 ? " dollar" : " dollars");
        if (cents_end > cents_begin) {
            std::string cents(text.substr(cents_begin, cents_end - cents_begin));
            if (cents.size() == 1) {
                cents.push_back('0');
            }
            uint64_t cent_value = 0;
            parse_uint64(cents, cent_value);
            if (cent_value != 0) {
                output.append(" and ");
                output.append(number_words(cent_value));
                output.append(cent_value == 1 ? " cent" : " cents");
            }
        }
        offset = cursor;
    }
    return output;
}

bool leap_year(unsigned year)
{
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

bool valid_date(unsigned month, unsigned day, unsigned year)
{
    constexpr std::array<unsigned, 12> days = {{
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    }};
    if (month < 1 || month > 12 || day < 1) {
        return false;
    }
    const unsigned maximum = days[month - 1]
        + ((month == 2 && leap_year(year)) ? 1 : 0);
    return day <= maximum;
}

bool parse_one_or_two_digits(
    std::string_view text,
    size_t offset,
    size_t &end,
    unsigned &value)
{
    if (offset >= text.size() || !ascii_is_digit(text[offset])) {
        return false;
    }
    end = offset + 1;
    if (end < text.size() && ascii_is_digit(text[end])) {
        ++end;
    }
    value = 0;
    for (size_t index = offset; index < end; ++index) {
        value = value * 10 + static_cast<unsigned>(text[index] - '0');
    }
    return true;
}

std::string expand_dates(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        size_t month_end = offset;
        size_t day_end = offset;
        unsigned month = 0;
        unsigned day = 0;
        if (boundary_before(text, offset)
            && parse_one_or_two_digits(text, offset, month_end, month)
            && month_end < text.size() && text[month_end] == '/'
            && parse_one_or_two_digits(text, month_end + 1, day_end, day)
            && day_end + 5 <= text.size() && text[day_end] == '/') {
            const size_t year_begin = day_end + 1;
            const bool four_digits = year_begin + 4 <= text.size()
                && std::all_of(
                    text.begin() + year_begin,
                    text.begin() + year_begin + 4,
                    ascii_is_digit);
            if (four_digits) {
                unsigned year = 0;
                for (size_t index = year_begin; index < year_begin + 4; ++index) {
                    year = year * 10 + static_cast<unsigned>(text[index] - '0');
                }
                const size_t end = year_begin + 4;
                if ((year / 100 == 19 || year / 100 == 20)
                    && boundary_after(text, end) && valid_date(month, day, year)) {
                    output.append(kMonths[month - 1]);
                    output.push_back(' ');
                    output.append(number_words_ordinal(day));
                    output.push_back(' ');
                    output.append(number_words(year));
                    offset = end;
                    continue;
                }
            }
        }
        output.push_back(text[offset++]);
    }
    return output;
}

bool parse_ampm(std::string_view text, size_t offset, size_t &end, char &kind)
{
    if (offset >= text.size() || (text[offset] != 'a' && text[offset] != 'A'
                                  && text[offset] != 'p' && text[offset] != 'P')) {
        return false;
    }
    kind = static_cast<char>(std::tolower(static_cast<unsigned char>(text[offset])));
    size_t cursor = offset + 1;
    if (cursor < text.size() && text[cursor] == '.') {
        ++cursor;
    }
    while (cursor < text.size()
           && std::isspace(static_cast<unsigned char>(text[cursor]))) {
        ++cursor;
    }
    if (cursor >= text.size() || (text[cursor] != 'm' && text[cursor] != 'M')) {
        return false;
    }
    ++cursor;
    if (cursor < text.size() && text[cursor] == '.'
        && word_boundary_at(text, cursor + 1)) {
        ++cursor;
    }
    if (!word_boundary_at(text, cursor)) {
        return false;
    }
    end = cursor;
    return true;
}

std::string expand_times(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        size_t hour_end = offset;
        unsigned hour = 0;
        if (boundary_before(text, offset)
            && parse_one_or_two_digits(text, offset, hour_end, hour)
            && hour_end + 3 <= text.size() && text[hour_end] == ':'
            && ascii_is_digit(text[hour_end + 1])
            && ascii_is_digit(text[hour_end + 2])) {
            const unsigned minute = static_cast<unsigned>(text[hour_end + 1] - '0') * 10
                + static_cast<unsigned>(text[hour_end + 2] - '0');
            size_t end = hour_end + 3;
            size_t suffix_begin = end;
            while (suffix_begin < text.size()
                   && std::isspace(static_cast<unsigned char>(text[suffix_begin]))) {
                ++suffix_begin;
            }
            size_t suffix_end = suffix_begin;
            char suffix = 0;
            const bool has_suffix = parse_ampm(text, suffix_begin, suffix_end, suffix);
            if ((has_suffix || boundary_after(text, end))) {
                output.append(number_words(hour));
                if (minute == 0) {
                    output.append(" o clock");
                } else if (minute < 10) {
                    output.append(" oh ");
                    output.append(number_words(minute));
                } else {
                    output.push_back(' ');
                    output.append(number_words(minute));
                }
                if (has_suffix) {
                    output.push_back(' ');
                    output.push_back(suffix);
                    output.append(" m");
                    end = suffix_end;
                }
                offset = end;
                continue;
            }
        }
        output.push_back(text[offset++]);
    }
    return output;
}

std::string expand_bare_hour_times(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        size_t hour_end = offset;
        unsigned hour = 0;
        if (boundary_before(text, offset)
            && parse_one_or_two_digits(text, offset, hour_end, hour)) {
            size_t suffix_begin = hour_end;
            while (suffix_begin < text.size()
                   && std::isspace(static_cast<unsigned char>(text[suffix_begin]))) {
                ++suffix_begin;
            }
            size_t suffix_end = suffix_begin;
            char suffix = 0;
            if (parse_ampm(text, suffix_begin, suffix_end, suffix)) {
                output.append(number_words(hour));
                output.push_back(' ');
                output.push_back(suffix);
                output.append(" m");
                offset = suffix_end;
                continue;
            }
        }
        output.push_back(text[offset++]);
    }
    return output;
}

std::string expand_phone_suffixes(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        const bool match = boundary_before(text, offset) && offset + 8 <= text.size()
            && std::all_of(text.begin() + offset, text.begin() + offset + 3, ascii_is_digit)
            && text[offset + 3] == '-'
            && std::all_of(text.begin() + offset + 4, text.begin() + offset + 8, ascii_is_digit)
            && boundary_after(text, offset + 8);
        if (match) {
            output.append(digit_words(text.substr(offset, 3)));
            output.append(", ");
            output.append(digit_words(text.substr(offset + 4, 4)));
            offset += 8;
        } else {
            output.push_back(text[offset++]);
        }
    }
    return output;
}

std::string expand_versions(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_digit(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t cursor = offset;
        std::vector<std::string_view> parts;
        while (cursor < text.size()) {
            const size_t begin = cursor;
            while (cursor < text.size() && ascii_is_digit(text[cursor])) {
                ++cursor;
            }
            if (cursor == begin) {
                break;
            }
            parts.push_back(text.substr(begin, cursor - begin));
            if (cursor >= text.size() || text[cursor] != '.'
                || cursor + 1 >= text.size() || !ascii_is_digit(text[cursor + 1])) {
                break;
            }
            ++cursor;
        }
        if (parts.size() < 3 || !boundary_after(text, cursor)) {
            output.push_back(text[offset++]);
            continue;
        }
        for (size_t index = 0; index < parts.size(); ++index) {
            if (index != 0) {
                output.append(" point ");
            }
            uint64_t value = 0;
            output.append(parse_uint64(parts[index], value) ? number_words(value)
                                                             : digit_words(parts[index]));
        }
        offset = cursor;
    }
    return output;
}

std::string expand_decimals(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_digit(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t whole_end = offset;
        while (whole_end < text.size() && ascii_is_digit(text[whole_end])) {
            ++whole_end;
        }
        if (whole_end >= text.size() || text[whole_end] != '.'
            || whole_end + 1 >= text.size() || !ascii_is_digit(text[whole_end + 1])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t end = whole_end + 1;
        while (end < text.size() && ascii_is_digit(text[end])) {
            ++end;
        }
        if (!boundary_after(text, end)) {
            output.push_back(text[offset++]);
            continue;
        }
        uint64_t whole = 0;
        const std::string_view whole_digits = text.substr(offset, whole_end - offset);
        output.append(parse_uint64(whole_digits, whole) ? number_words(whole)
                                                        : digit_words(whole_digits));
        output.append(" point ");
        output.append(digit_words(text.substr(whole_end + 1, end - whole_end - 1)));
        offset = end;
    }
    return output;
}

std::string expand_ordinals(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_digit(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t digits_end = offset;
        while (digits_end < text.size() && ascii_is_digit(text[digits_end])) {
            ++digits_end;
        }
        if (digits_end + 2 > text.size()) {
            output.push_back(text[offset++]);
            continue;
        }
        const std::string suffix = {
            static_cast<char>(std::tolower(static_cast<unsigned char>(text[digits_end]))),
            static_cast<char>(std::tolower(static_cast<unsigned char>(text[digits_end + 1]))),
        };
        const size_t end = digits_end + 2;
        if ((suffix != "st" && suffix != "nd" && suffix != "rd" && suffix != "th")
            || !boundary_after(text, end)) {
            output.push_back(text[offset++]);
            continue;
        }
        uint64_t value = 0;
        const std::string_view digits = text.substr(offset, digits_end - offset);
        output.append(parse_uint64(digits, value) ? number_words_ordinal(value)
                                                  : digit_words(digits));
        offset = end;
    }
    return output;
}

std::string expand_numbers(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_digit(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t end = offset;
        while (end < text.size()
               && (ascii_is_digit(text[end]) || text[end] == ',')) {
            ++end;
        }
        while (end > offset && text[end - 1] == ',') {
            --end;
        }
        if (!boundary_after(text, end)) {
            output.push_back(text[offset++]);
            continue;
        }
        std::string digits;
        for (size_t index = offset; index < end; ++index) {
            if (ascii_is_digit(text[index])) {
                digits.push_back(text[index]);
            }
        }
        uint64_t value = 0;
        if (digits.size() >= 5 && !digits.starts_with("20")) {
            output.append(digit_words(digits));
        } else if (parse_uint64(digits, value)) {
            output.append(number_words(value));
        } else {
            output.append(digit_words(digits));
        }
        offset = end;
    }
    return output;
}

std::string expand_acronyms(std::string_view text)
{
    std::string output;
    output.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (!boundary_before(text, offset) || !ascii_is_upper(text[offset])) {
            output.push_back(text[offset++]);
            continue;
        }
        size_t end = offset;
        while (end < text.size() && ascii_is_upper(text[end])) {
            ++end;
        }
        if (end - offset < 2 || !boundary_after(text, end)) {
            output.push_back(text[offset++]);
            continue;
        }
        for (size_t index = offset; index < end; ++index) {
            if (index != offset) {
                output.push_back(' ');
            }
            output.append(letter_name(text[index]));
        }
        offset = end;
    }
    return output;
}

std::string clean_punctuation(std::string_view text)
{
    std::string stage;
    stage.reserve(text.size());
    size_t offset = 0;
    while (offset < text.size()) {
        if (text[offset] != ',') {
            stage.push_back(text[offset++]);
            continue;
        }
        size_t cursor = offset + 1;
        bool repeated = false;
        while (cursor < text.size()) {
            size_t next = cursor;
            while (next < text.size()
                   && std::isspace(static_cast<unsigned char>(text[next]))) {
                ++next;
            }
            if (next >= text.size() || text[next] != ',') {
                break;
            }
            repeated = true;
            cursor = next + 1;
        }
        stage.push_back(',');
        offset = repeated ? cursor : offset + 1;
    }

    std::string without_comma_before_terminal;
    without_comma_before_terminal.reserve(stage.size());
    offset = 0;
    while (offset < stage.size()) {
        if (stage[offset] == ',') {
            size_t cursor = offset + 1;
            while (cursor < stage.size()
                   && std::isspace(static_cast<unsigned char>(stage[cursor]))) {
                ++cursor;
            }
            if (cursor < stage.size()
                && (stage[cursor] == '.' || stage[cursor] == '!'
                    || stage[cursor] == '?')) {
                offset = cursor;
                continue;
            }
        }
        without_comma_before_terminal.push_back(stage[offset++]);
    }

    std::string compact;
    compact.reserve(without_comma_before_terminal.size());
    for (const char character : without_comma_before_terminal) {
        const bool punctuation = character == ',' || character == ';'
            || character == ':' || character == '.' || character == '!'
            || character == '?';
        if (punctuation) {
            while (!compact.empty() && compact.back() == ' ') {
                compact.pop_back();
            }
        }
        compact.push_back(character);
    }

    std::string spaced;
    spaced.reserve(compact.size() + 8);
    for (size_t index = 0; index < compact.size(); ++index) {
        const char character = compact[index];
        spaced.push_back(character);
        const bool punctuation = character == ',' || character == ';'
            || character == ':' || character == '.' || character == '!'
            || character == '?';
        if (punctuation && index + 1 < compact.size()
            && !std::isspace(static_cast<unsigned char>(compact[index + 1]))) {
            spaced.push_back(' ');
        }
    }
    return collapse_spaces(spaced);
}

}  // namespace

std::string normalize(std::string text)
{
    replace_all(text, "‘", "'");
    replace_all(text, "’", "'");
    replace_all(text, "“", "\"");
    replace_all(text, "”", "\"");
    replace_all(text, "–", "-");
    replace_all(text, "—", ", ");
    replace_all(text, "…", "...");
    for (const std::string_view bracket : {"(", ")", "[", "]", "{", "}"}) {
        replace_all(text, bracket, ", ");
    }
    text = collapse_spaces(text);

    for (const Replacement &replacement : kWordOverrides) {
        text = replace_bounded(
            std::move(text), replacement.source, replacement.target, false, true);
    }
    for (const Replacement &replacement : kAbbreviations) {
        text = replace_bounded(
            std::move(text), replacement.source, replacement.target, true, false);
    }

    text = expand_dotted_acronyms(text);
    text = expand_labeled_identifiers(text);
    text = expand_street_numbers(text);
    text = expand_money(text);
    text = expand_dates(text);
    text = expand_times(text);
    text = expand_bare_hour_times(text);
    text = expand_phone_suffixes(text);
    text = expand_versions(text);
    text = expand_decimals(text);
    text = expand_ordinals(text);
    text = expand_numbers(text);
    text = expand_acronyms(text);
    return clean_punctuation(text);
}

}  // namespace inflect::text::normalizer
