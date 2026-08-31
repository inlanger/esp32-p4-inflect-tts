#include <algorithm>
#include <array>
#include <inttypes.h>
#include <iterator>
#include <map>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_private/esp_clk.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "dl_base_shape.hpp"
#include "dl_model_base.hpp"
#include "dl_module_creator.hpp"
#include "dl_tool.hpp"
#include "fbs_loader.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"
#include "inflect_text_frontend.h"
#include "inflect_text_frontend_worker.h"
#if INFLECT_BENCH_INTERLEAVE
#include "inflect_polyphase_interleave.h"
#endif
#if INFLECT_BENCH_PRIMITIVES
#include "inflect_decoder_primitives.h"
#endif
#if INFLECT_BENCH_STUDENTS
#include "inflect_student_benchmarks.h"
#endif
#include "sdkconfig.h"

namespace {

extern const uint8_t inflect_full_assets_start[]
    asm("_binary_inflect_full_assets_bin_start");
extern const uint8_t inflect_full_assets_end[]
    asm("_binary_inflect_full_assets_bin_end");

constexpr char kAcousticModel[] = "inflect-nano-acoustic-core-t49.espdl";
constexpr char kMainModelPartition[] = "model";
constexpr char kFlowFullModel[] = "inflect-nano-reverse-flow-t143.espdl";
constexpr char kFlowBucket48Model[] = "inflect-nano-reverse-flow-t48.espdl";
constexpr char kFlowBucket96Partition[] = "flow_short";
constexpr char kFlowBucket96Model[] = "inflect-nano-reverse-flow-t96.espdl";
constexpr char kDecoderBucket64Model[] = "inflect-decoder-bucket64-polyphase.espdl";
constexpr char kDecoderBucket96Model[] = "inflect-decoder-bucket96-polyphase.espdl";
constexpr char kDecoderFirst100Model[] = "inflect-decoder-first100-polyphase.espdl";
constexpr char kDecoderTail69Model[] = "inflect-decoder-tail69-polyphase.espdl";
constexpr char kDecoderTail65Model[] = "inflect-decoder-tail65-polyphase.espdl";
constexpr char kDecoderFullModel[] = "inflect-decoder-full-polyphase.espdl";
constexpr uint32_t kFlowBucket48Frames = 48;
constexpr uint32_t kDecoderBucket64Frames = 64;
constexpr uint32_t kDecoderBucket96Frames = 96;
constexpr uint32_t kDecoderFirst100Frames = 100;
constexpr uint32_t kDecoderTail69Frames = 69;
constexpr uint32_t kDecoderTail65Frames = 65;
constexpr uint32_t kDecoderRightContextFrames = 13;
constexpr uint32_t kDecoderTile10065MaximumFrames =
    kDecoderFirst100Frames - kDecoderRightContextFrames
    + kDecoderTail65Frames - kDecoderRightContextFrames;
#if INFLECT_DECODER_BRIDGE96_69
constexpr const char *kDecoderTiledFirstModel = kDecoderBucket96Model;
constexpr const char *kDecoderTiledSecondModel = kDecoderTail69Model;
constexpr const char *kDecoderTiledSecondStage = "decoder_tail69";
constexpr uint32_t kDecoderTiledFirstFrames = kDecoderBucket96Frames;
constexpr uint32_t kDecoderTiledSecondFrames = kDecoderTail69Frames;
constexpr uint32_t kDecoderTiledFirstPublishFrames =
    kDecoderBucket96Frames - kDecoderRightContextFrames + 4;
constexpr uint32_t kDecoderTiledMaximumFrames = 143;
#elif INFLECT_DECODER_TILE96
constexpr const char *kDecoderTiledFirstModel = kDecoderFirst100Model;
constexpr const char *kDecoderTiledSecondModel = kDecoderTail65Model;
constexpr const char *kDecoderTiledSecondStage = "decoder_tail65";
constexpr uint32_t kDecoderTiledFirstFrames = kDecoderFirst100Frames;
constexpr uint32_t kDecoderTiledSecondFrames = kDecoderTail65Frames;
constexpr uint32_t kDecoderTiledFirstPublishFrames =
    kDecoderFirst100Frames - kDecoderRightContextFrames;
constexpr uint32_t kDecoderTiledMaximumFrames =
    kDecoderTile10065MaximumFrames;
#else
constexpr const char *kDecoderTiledSecondModel = kDecoderTail65Model;
constexpr const char *kDecoderTiledSecondStage = "decoder_tail65";
constexpr uint32_t kDecoderTiledSecondFrames = kDecoderTail65Frames;
#endif
constexpr size_t kMaxLatentChannels = 128;
constexpr size_t kMaxTextTokens = 49;
constexpr int kSampleRate = 24000;
constexpr int kDecoderHop = 256;
constexpr uart_port_t kConsoleUart =
    static_cast<uart_port_t>(CONFIG_ESP_CONSOLE_UART_NUM);

class WhereFloat : public dl::module::Module {
public:
    explicit WhereFloat(const char *name) :
        Module(name, dl::MODULE_NON_INPLACE, dl::QUANT_TYPE_FLOAT32)
    {
    }

    std::vector<std::vector<int>> get_output_shape(
        std::vector<std::vector<int>> &input_shapes) override
    {
        std::vector<int> shape = dl::base::get_multidirectional_broadcasting_shape(
            input_shapes[0], input_shapes[1]);
        shape = dl::base::get_multidirectional_broadcasting_shape(shape, input_shapes[2]);
        return {shape};
    }

    void forward(dl::ModelContext *context, dl::runtime_mode_t) override
    {
        dl::TensorBase *condition = context->get_tensor(m_inputs_index[0]);
        dl::TensorBase *when_true = context->get_tensor(m_inputs_index[1]);
        dl::TensorBase *when_false = context->get_tensor(m_inputs_index[2]);
        dl::TensorBase *output = context->get_tensor(m_outputs_index[0]);
        if (
            condition->get_dtype() != dl::DATA_TYPE_BOOL
            || when_true->get_dtype() != dl::DATA_TYPE_FLOAT
            || when_false->get_dtype() != dl::DATA_TYPE_FLOAT
            || output->get_dtype() != dl::DATA_TYPE_FLOAT) {
            printf("{\"type\":\"full_pipeline_error\",\"stage\":\"where_dtype\"}\n");
            return;
        }

        const std::vector<int> output_shape = output->get_shape();
        const std::vector<int> condition_shape = condition->get_shape();
        const std::vector<int> true_shape = when_true->get_shape();
        const std::vector<int> false_shape = when_false->get_shape();
        const bool *condition_data = condition->get_element_ptr<bool>();
        const float *true_data = when_true->get_element_ptr<float>();
        const float *false_data = when_false->get_element_ptr<float>();
        float *output_data = output->get_element_ptr<float>();
        for (size_t index = 0; index < static_cast<size_t>(output->get_size()); ++index) {
            output_data[index] = condition_data[broadcast_index(index, output_shape, condition_shape)]
                ? true_data[broadcast_index(index, output_shape, true_shape)]
                : false_data[broadcast_index(index, output_shape, false_shape)];
        }
    }

    static dl::module::Module *deserialize(fbs::FbsModel *, std::string node_name)
    {
        return new WhereFloat(node_name.c_str());
    }

private:
    static size_t broadcast_index(
        size_t output_index,
        const std::vector<int> &output_shape,
        const std::vector<int> &input_shape)
    {
        size_t input_index = 0;
        size_t input_stride = 1;
        for (size_t offset = 0; offset < output_shape.size(); ++offset) {
            const size_t output_axis = output_shape.size() - 1 - offset;
            const size_t coordinate = output_index % output_shape[output_axis];
            output_index /= output_shape[output_axis];
            if (offset < input_shape.size()) {
                const size_t input_axis = input_shape.size() - 1 - offset;
                if (input_shape[input_axis] != 1) {
                    input_index += coordinate * input_stride;
                }
                input_stride *= input_shape[input_axis];
            }
        }
        return input_index;
    }
};

void register_custom_modules()
{
    dl::module::ModuleCreator *creator = dl::module::ModuleCreator::get_instance();
    creator->register_dl_modules();
    creator->register_module("Where", WhereFloat::deserialize);
#if INFLECT_BENCH_INTERLEAVE
    creator->register_module(
        "PolyphaseInterleave",
        inflect::polyphase::PolyphaseInterleave::deserialize);
#endif
}

#pragma pack(push, 1)
struct FullPipelineAssetsHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab_size;
    uint32_t embedding_channels;
    int32_t embedding_exponent;
    int32_t mask_exponent;
    uint32_t max_text_tokens;
    uint32_t max_latent_frames;
    uint32_t latent_channels;
    uint32_t token_count;
    uint32_t expected_latent_frames;
    float noise_scale;
    uint32_t embedding_count;
    uint32_t noise_count;
    uint32_t token_fnv1a;
};
#pragma pack(pop)

static_assert(sizeof(FullPipelineAssetsHeader) == 68);

struct AssetsView {
    const FullPipelineAssetsHeader *header;
    const int8_t *embedding;
    const float *noise;
    const uint16_t *tokens;
};

struct PipelineRequest {
    const uint16_t *tokens;
    uint32_t token_count;
    uint32_t expected_latent_frames;
    uint32_t token_fnv1a;
    bool validate_reference;
};

enum class PipelineFailure {
    None,
    Other,
    DurationOverflow,
};

struct PipelineOutcome {
    PipelineFailure failure = PipelineFailure::Other;
    uint32_t required_latent_frames = 0;
    uint32_t generated_pcm_samples = 0;
};

struct StageMetrics {
    int64_t load_us = 0;
    int64_t run_us = 0;
    size_t internal_bytes = 0;
    size_t psram_bytes = 0;
    size_t flash_bytes = 0;
    bool cached = false;
};

uint32_t runtime_max_latent_frames(const FullPipelineAssetsHeader &spec)
{
#if INFLECT_SHORT_VALUE64
    return kDecoderBucket96Frames - kDecoderRightContextFrames;
#else
    return spec.max_latent_frames;
#endif
}

uint32_t fnv1a(const uint8_t *data, size_t size)
{
    uint32_t value = 0x811c9dc5;
    for (size_t index = 0; index < size; ++index) {
        value ^= data[index];
        value *= 0x01000193;
    }
    return value;
}

void print_heap_snapshot(const char *stage)
{
    printf(
        "{\"type\":\"heap\",\"stage\":\"%s\"," 
        "\"internal_free\":%u,\"internal_largest\":%u," 
        "\"psram_free\":%u,\"psram_largest\":%u," 
        "\"stack_high_water_bytes\":%u}\n",
        stage,
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
        static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
        static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)),
        static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
}

bool parse_assets(AssetsView &view)
{
    const size_t bytes = inflect_full_assets_end - inflect_full_assets_start;
    if (bytes < sizeof(FullPipelineAssetsHeader)) {
        return false;
    }
    const auto *header = reinterpret_cast<const FullPipelineAssetsHeader *>(
        inflect_full_assets_start);
    if (
        memcmp(header->magic, "INFLFULL", 8) != 0
        || header->version != 1
        || header->header_bytes != sizeof(FullPipelineAssetsHeader)
        || header->embedding_count
            != header->vocab_size * header->embedding_channels
        || header->noise_count
            != header->latent_channels * header->max_latent_frames
        || header->latent_channels == 0
        || header->latent_channels > kMaxLatentChannels
        || header->max_text_tokens != kMaxTextTokens
        || header->token_count > header->max_text_tokens) {
        return false;
    }
    const size_t required = header->header_bytes
        + header->embedding_count * sizeof(int8_t)
        + header->noise_count * sizeof(float)
        + header->token_count * sizeof(uint16_t);
    if (required != bytes) {
        return false;
    }
    const uint8_t *cursor = inflect_full_assets_start + header->header_bytes;
    view.header = header;
    view.embedding = reinterpret_cast<const int8_t *>(cursor);
    cursor += header->embedding_count * sizeof(int8_t);
    view.noise = reinterpret_cast<const float *>(cursor);
    cursor += header->noise_count * sizeof(float);
    view.tokens = reinterpret_cast<const uint16_t *>(cursor);
    return reinterpret_cast<uintptr_t>(view.noise) % alignof(float) == 0;
}

bool verify_model_package(
    const char *partition,
    int expected_models,
    const char *expected_version = nullptr)
{
    const int64_t started = esp_timer_get_time();
    fbs::FbsLoader loader(partition, fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    char version[16] = {};
    const bool verified = loader.verify_package_sha256();
    const esp_err_t version_result = loader.get_package_version(version, sizeof(version));
    const bool version_ok = expected_version == nullptr
        || (version_result == ESP_OK && strcmp(version, expected_version) == 0);
    printf(
        "{\"type\":\"model_package\",\"partition\":\"%s\",\"format\":%d,"
        "\"models\":%d,\"bytes\":%" PRIu32 ",\"version\":\"%s\","
        "\"verify_us\":%" PRId64 ",\"verified\":%s,\"accepted\":%s}\n",
        partition,
        static_cast<int>(loader.get_model_format()),
        loader.get_model_num(),
        loader.get_package_size(),
        version_result == ESP_OK ? version : "",
        esp_timer_get_time() - started,
        verified ? "true" : "false",
        verified && loader.get_model_num() == expected_models && version_ok
            ? "true"
            : "false");
    return verified && loader.get_model_num() == expected_models && version_ok;
}

bool is_decoder_model(const char *name)
{
    return strcmp(name, kDecoderBucket64Model) == 0
        || strcmp(name, kDecoderBucket96Model) == 0
        || strcmp(name, kDecoderFirst100Model) == 0
        || strcmp(name, kDecoderTail69Model) == 0
        || strcmp(name, kDecoderTail65Model) == 0
        || strcmp(name, kDecoderFullModel) == 0;
}

dl::Model *load_model(
    const char *partition,
    const char *name,
    StageMetrics &metrics)
{
#if INFLECT_PERSIST_MODELS
    static std::map<std::string, dl::Model *> models;
    const std::string cache_key = std::string(partition) + ":" + name;
    const auto cached = models.find(cache_key);
    if (cached != models.end()) {
        metrics.cached = true;
        const auto memory = cached->second->get_memory_info().at("total");
        metrics.internal_bytes = memory.internal;
        metrics.psram_bytes = memory.psram;
        metrics.flash_bytes = memory.flash;
        return cached->second;
    }
#endif
    static void *decoder_full_psram_pad = nullptr;
    if (
        strcmp(name, kDecoderFullModel) == 0
        && INFLECT_DECODER_FULL_PSRAM_PAD_BYTES > 0
        && decoder_full_psram_pad == nullptr) {
        const size_t free_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        const size_t largest_before =
            heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
        decoder_full_psram_pad = dl::tool::calloc_aligned(
            INFLECT_DECODER_FULL_PSRAM_PAD_BYTES,
            1,
            MALLOC_CAP_SPIRAM);
        if (decoder_full_psram_pad == nullptr) {
            printf(
                "{\"type\":\"full_pipeline_error\","
                "\"stage\":\"decoder_psram_pad\",\"bytes\":%u}\n",
                static_cast<unsigned>(INFLECT_DECODER_FULL_PSRAM_PAD_BYTES));
            abort();
        }
        printf(
            "{\"type\":\"decoder_psram_pad\",\"model\":\"%s\","
            "\"bytes\":%u,\"address\":\"%p\","
            "\"free_before\":%u,\"free_after\":%u,"
            "\"largest_before\":%u,\"largest_after\":%u}\n",
            name,
            static_cast<unsigned>(INFLECT_DECODER_FULL_PSRAM_PAD_BYTES),
            decoder_full_psram_pad,
            static_cast<unsigned>(free_before),
            static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
            static_cast<unsigned>(largest_before),
            static_cast<unsigned>(
                heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)));
    }
    const int64_t started = esp_timer_get_time();
    const int internal_bytes = is_decoder_model(name)
        ? INFLECT_DECODER_INTERNAL_BYTES
        : 0;
    auto *model = new dl::Model(
        partition,
        name,
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
        internal_bytes,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        true);
    metrics.load_us = esp_timer_get_time() - started;
    const auto memory = model->get_memory_info().at("total");
    metrics.internal_bytes = memory.internal;
    metrics.psram_bytes = memory.psram;
    metrics.flash_bytes = memory.flash;
#if INFLECT_PERSIST_MODELS
    models.emplace(cache_key, model);
#endif
    printf(
        "{\"type\":\"stage_load\",\"partition\":\"%s\",\"model\":\"%s\","
        "\"load_us\":%" PRId64 ",\"internal_bytes\":%u,"
        "\"psram_bytes\":%u,\"flash_bytes\":%u}\n",
        partition,
        name,
        metrics.load_us,
        static_cast<unsigned>(metrics.internal_bytes),
        static_cast<unsigned>(metrics.psram_bytes),
        static_cast<unsigned>(metrics.flash_bytes));
    return model;
}

void release_model(dl::Model *model)
{
#if INFLECT_PERSIST_MODELS
    (void)model;
#else
    delete model;
#endif
}

#if INFLECT_VALIDATE_CANARIES
uint32_t compare_tensor(dl::TensorBase *actual, dl::TensorBase *expected)
{
    if (
        actual == nullptr || expected == nullptr
        || actual->get_dtype() != expected->get_dtype()
        || actual->get_size() != expected->get_size()) {
        return UINT32_MAX;
    }
    const auto *lhs = actual->get_element_ptr<int8_t>();
    const auto *rhs = expected->get_element_ptr<int8_t>();
    uint32_t mismatches = 0;
    for (size_t index = 0; index < actual->get_size(); ++index) {
        mismatches += lhs[index] != rhs[index];
    }
    return mismatches;
}
#endif

void print_tensor_map(
    const char *model,
    const char *direction,
    std::map<std::string, dl::TensorBase *> &tensors)
{
    for (const auto &entry : tensors) {
        printf(
            "{\"type\":\"tensor_map\",\"model\":\"%s\","
            "\"direction\":\"%s\",\"key\":\"%s\","
            "\"size\":%d,\"bytes\":%d,\"exponent\":%d,\"dtype\":\"%s\"}\n",
            model,
            direction,
            entry.first.c_str(),
            entry.second->get_size(),
            entry.second->get_bytes(),
            entry.second->get_exponent(),
            entry.second->get_dtype_string());
    }
}

dl::TensorBase *find_tensor(
    std::map<std::string, dl::TensorBase *> &tensors,
    const char *key_fragment,
    size_t element_count,
    dl::TensorBase *excluded = nullptr)
{
    for (const auto &entry : tensors) {
        if (
            entry.second != excluded
            && entry.first.find(key_fragment) != std::string::npos
            && static_cast<size_t>(entry.second->get_size()) == element_count) {
            return entry.second;
        }
    }
    for (const auto &entry : tensors) {
        if (
            entry.second != excluded
            && static_cast<size_t>(entry.second->get_size()) == element_count) {
            return entry.second;
        }
    }
    return nullptr;
}

#if INFLECT_VALIDATE_CANARIES
bool check_acoustic_canary(dl::Model *model, bool check_inputs)
{
    fbs::FbsLoader loader("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    fbs::FbsModel *reference = loader.load(kAcousticModel, nullptr, false);
    if (reference == nullptr) {
        return false;
    }
    reference->load_map();
    uint32_t mismatches = 0;
    auto &tensors = check_inputs ? model->get_inputs() : model->get_outputs();
    for (const auto &entry : tensors) {
        dl::TensorBase *expected = check_inputs
            ? reference->get_test_input_tensor(entry.first)
            : reference->get_test_output_tensor(entry.first);
        const uint32_t tensor_mismatches = compare_tensor(entry.second, expected);
        mismatches = tensor_mismatches == UINT32_MAX
            ? UINT32_MAX
            : mismatches + tensor_mismatches;
        delete expected;
    }
    reference->clear_map();
    delete reference;
    printf(
        "{\"type\":\"acoustic_canary\",\"direction\":\"%s\","
        "\"mismatches\":%" PRIu32 ",\"passed\":%s}\n",
        check_inputs ? "input" : "output",
        mismatches,
        mismatches == 0 ? "true" : "false");
    return mismatches == 0;
}
#endif

#if INFLECT_EMIT_PCM_HEX
void emit_pcm_hex(const int16_t *samples, size_t sample_count, uint32_t checksum)
{
    constexpr size_t kChunkBytes = 96;
    const auto *bytes = reinterpret_cast<const uint8_t *>(samples);
    const size_t byte_count = sample_count * sizeof(int16_t);
    printf(
        "PCM_BEGIN samples=%u bytes=%u sample_rate=%d channels=1 "
        "sample_width=2 fnv1a=%08" PRIx32 "\n",
        static_cast<unsigned>(sample_count),
        static_cast<unsigned>(byte_count),
        kSampleRate,
        checksum);
    for (size_t offset = 0; offset < byte_count; offset += kChunkBytes) {
        const size_t chunk = std::min(kChunkBytes, byte_count - offset);
        printf("PCM_HEX offset=%08x data=", static_cast<unsigned>(offset));
        for (size_t index = 0; index < chunk; ++index) {
            printf("%02x", bytes[offset + index]);
        }
        printf("\n");
    }
    printf(
        "PCM_END bytes=%u fnv1a=%08" PRIx32 "\n",
        static_cast<unsigned>(byte_count),
        checksum);
    fflush(stdout);
}
#endif

bool emit_pcm_binary(const int16_t *samples, size_t sample_count, uint32_t checksum)
{
    constexpr size_t kTransportChunkBytes = 2048;
    const size_t byte_count = sample_count * sizeof(int16_t);
    printf(
        "PCM_BINARY_BEGIN samples=%u bytes=%u sample_rate=%d channels=1 "
        "sample_width=2 fnv1a=%08" PRIx32 "\n",
        static_cast<unsigned>(sample_count),
        static_cast<unsigned>(byte_count),
        kSampleRate,
        checksum);
    fflush(stdout);
    const auto *bytes = reinterpret_cast<const uint8_t *>(samples);
    size_t written = 0;
    bool transport_ok = true;
    while (written < byte_count) {
        const size_t chunk = std::min(kTransportChunkBytes, byte_count - written);
        const int chunk_written = uart_write_bytes(
            kConsoleUart,
            bytes + written,
            chunk);
        if (
            chunk_written != static_cast<int>(chunk)
            || uart_wait_tx_done(kConsoleUart, portMAX_DELAY) != ESP_OK) {
            transport_ok = false;
            break;
        }
        written += chunk;
    }
    printf(
        "\nPCM_BINARY_END bytes=%u written=%d fnv1a=%08" PRIx32
        " passed=%s\n",
        static_cast<unsigned>(byte_count),
        static_cast<int>(written),
        checksum,
        transport_ok && written == byte_count ? "true" : "false");
    fflush(stdout);
    return transport_ok && written == byte_count;
}

bool run_full_pipeline(
    const AssetsView &assets,
    const PipelineRequest &request,
    bool emit_binary_pcm = false,
    bool detailed_diagnostics = true,
    PipelineOutcome *outcome = nullptr,
    bool fade_pcm_edges = false,
    uint32_t pause_after_ms = 0)
{
    if (outcome != nullptr) {
        *outcome = {};
    }
    const auto &spec = *assets.header;
    if (request.token_count == 0 || request.token_count > spec.max_text_tokens) {
        printf(
            "{\"type\":\"full_pipeline_error\",\"stage\":\"token_count\","
            "\"tokens\":%u,\"maximum\":%u}\n",
            static_cast<unsigned>(request.token_count),
            static_cast<unsigned>(spec.max_text_tokens));
        return false;
    }
    const int64_t pipeline_started = esp_timer_get_time();
#if INFLECT_PRELOAD_FULL_DECODER
    StageMetrics decoder_preload_metrics;
    if (detailed_diagnostics) {
        print_heap_snapshot("before_decoder_preload");
    }
    load_model(
        kMainModelPartition,
        kDecoderFullModel,
        decoder_preload_metrics);
    if (detailed_diagnostics) {
        print_heap_snapshot("after_decoder_preload");
    }
#endif
    StageMetrics acoustic_metrics;
    if (detailed_diagnostics) {
        print_heap_snapshot("before_acoustic");
    }
    dl::Model *acoustic = load_model(
        kMainModelPartition, kAcousticModel, acoustic_metrics);
    auto &acoustic_inputs = acoustic->get_inputs();
    auto &acoustic_outputs = acoustic->get_outputs();
    if (!acoustic_metrics.cached) {
        print_tensor_map("acoustic", "input", acoustic_inputs);
        print_tensor_map("acoustic", "output", acoustic_outputs);
    }
    dl::TensorBase *embedded = find_tensor(
        acoustic_inputs, "Mul_0", spec.embedding_channels * spec.max_text_tokens);
    dl::TensorBase *text_mask = find_tensor(
        acoustic_inputs, "Mul_1", spec.max_text_tokens);
    dl::TensorBase *means = find_tensor(
        acoustic_outputs, "1650", spec.latent_channels * spec.max_text_tokens);
    dl::TensorBase *log_scales = find_tensor(
        acoustic_outputs,
        "1651",
        spec.latent_channels * spec.max_text_tokens,
        means);
    dl::TensorBase *log_durations = find_tensor(
        acoustic_outputs, "1666", spec.max_text_tokens);
    if (
        embedded == nullptr || text_mask == nullptr || means == nullptr
        || log_scales == nullptr || log_durations == nullptr
        || embedded->get_dtype() != dl::DATA_TYPE_INT8
        || embedded->get_exponent() != spec.embedding_exponent
        || text_mask->get_exponent() != spec.mask_exponent) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"acoustic_io\"}\n");
        release_model(acoustic);
        return false;
    }

    auto *embedded_data = embedded->get_element_ptr<int8_t>();
    auto *mask_data = text_mask->get_element_ptr<int8_t>();
    memset(embedded_data, 0, embedded->get_bytes());
    memset(mask_data, 0, text_mask->get_bytes());
    for (size_t token_index = 0; token_index < request.token_count; ++token_index) {
        const uint16_t token = request.tokens[token_index];
        if (token >= spec.vocab_size) {
            printf("{\"type\":\"full_pipeline_error\",\"stage\":\"token_range\"}\n");
            release_model(acoustic);
            return false;
        }
        for (size_t channel = 0; channel < spec.embedding_channels; ++channel) {
            embedded_data[channel * spec.max_text_tokens + token_index]
                = assets.embedding[token * spec.embedding_channels + channel];
        }
        mask_data[token_index] = 127;
    }
    const uint32_t token_hash = fnv1a(
        reinterpret_cast<const uint8_t *>(request.tokens),
        request.token_count * sizeof(uint16_t));
    if (token_hash != request.token_fnv1a) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"token_hash\"}\n");
        release_model(acoustic);
        return false;
    }

#if INFLECT_VALIDATE_CANARIES
    const bool acoustic_input_parity = !request.validate_reference
        || check_acoustic_canary(acoustic, true);
#else
    const bool acoustic_input_parity = true;
#endif
    if (!acoustic_input_parity) {
        release_model(acoustic);
        return false;
    }

    int64_t started = esp_timer_get_time();
    acoustic->run(dl::RUNTIME_MODE_MULTI_CORE);
    acoustic_metrics.run_us = esp_timer_get_time() - started;
    vTaskDelay(1);
#if INFLECT_VALIDATE_CANARIES
    const bool acoustic_output_parity = !request.validate_reference
        || check_acoustic_canary(acoustic, false);
#else
    const bool acoustic_output_parity = true;
#endif
    if (!acoustic_output_parity) {
        release_model(acoustic);
        return false;
    }

    const int8_t *means_data = means->get_element_ptr<int8_t>();
    const int8_t *logs_data = log_scales->get_element_ptr<int8_t>();
    const int8_t *durations_data = log_durations->get_element_ptr<int8_t>();
    const float means_scale = ldexpf(1.0f, means->get_exponent());
    const float logs_scale = ldexpf(1.0f, log_scales->get_exponent());
    const float duration_scale = ldexpf(1.0f, log_durations->get_exponent());
    std::array<uint32_t, kMaxTextTokens> token_durations = {};
    uint32_t latent_frames = 0;
    for (size_t token_index = 0; token_index < request.token_count; ++token_index) {
        const uint32_t duration = static_cast<uint32_t>(std::max(
            1,
            static_cast<int>(ceilf(
                expf(durations_data[token_index] * duration_scale)))));
        token_durations[token_index] = duration;
        latent_frames += duration;
    }
    const uint32_t maximum_latent_frames = runtime_max_latent_frames(spec);
    if (latent_frames > maximum_latent_frames) {
        printf(
            "{\"type\":\"full_pipeline_error\","
            "\"stage\":\"duration_overflow\",\"required_frames\":%u,"
            "\"maximum_frames\":%u}\n",
            static_cast<unsigned>(latent_frames),
            static_cast<unsigned>(maximum_latent_frames));
        if (outcome != nullptr) {
            outcome->failure = PipelineFailure::DurationOverflow;
            outcome->required_latent_frames = latent_frames;
        }
        release_model(acoustic);
        return false;
    }
    auto *flow_input_buffer = static_cast<int8_t *>(heap_caps_aligned_alloc(
        16,
        spec.latent_channels * spec.max_latent_frames,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (flow_input_buffer == nullptr) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"latent_alloc\"}\n");
        release_model(acoustic);
        return false;
    }
    memset(
        flow_input_buffer,
        0,
        spec.latent_channels * spec.max_latent_frames);

    latent_frames = 0;
    uint32_t clipped_values = 0;
    std::array<float, kMaxLatentChannels> token_means;
    std::array<float, kMaxLatentChannels> token_standard_deviations;
    for (size_t token_index = 0; token_index < request.token_count; ++token_index) {
        const uint32_t duration = token_durations[token_index];
        for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
            const size_t acoustic_index = token_index * spec.latent_channels + channel;
            token_means[channel] = means_data[acoustic_index] * means_scale;
            token_standard_deviations[channel] = expf(
                logs_data[acoustic_index] * logs_scale);
        }
        for (uint32_t repeat = 0; repeat < duration; ++repeat) {
            for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
                const float noise = assets.noise[
                    channel * spec.max_latent_frames + latent_frames];
                const float value = token_means[channel]
                    + noise * token_standard_deviations[channel] * spec.noise_scale;
                int quantized = static_cast<int>(lroundf(value / 0.125f));
                if (quantized < -128 || quantized > 127) {
                    ++clipped_values;
                }
                quantized = std::clamp(quantized, -128, 127);
                flow_input_buffer[channel * spec.max_latent_frames + latent_frames]
                    = static_cast<int8_t>(quantized);
            }
            ++latent_frames;
        }
    }
    if (detailed_diagnostics) {
        const uint32_t acoustic_output_hash = fnv1a(
            reinterpret_cast<const uint8_t *>(means_data), means->get_bytes())
            ^ fnv1a(
                reinterpret_cast<const uint8_t *>(logs_data),
                log_scales->get_bytes())
            ^ fnv1a(
                reinterpret_cast<const uint8_t *>(durations_data),
                log_durations->get_bytes());
        printf(
            "{\"type\":\"acoustic_result\",\"run_us\":%" PRId64 ","
            "\"latent_frames\":%u,\"fp32_reference_frames\":%u,"
            "\"clipped_latent_values\":%u,"
            "\"output_fnv1a_xor\":\"%08" PRIx32 "\"}\n",
            acoustic_metrics.run_us,
            static_cast<unsigned>(latent_frames),
            static_cast<unsigned>(request.expected_latent_frames),
            static_cast<unsigned>(clipped_values),
            acoustic_output_hash);
    }
    release_model(acoustic);
    if (detailed_diagnostics) {
        print_heap_snapshot("after_acoustic_free");
    }

#if INFLECT_SHORT_VALUE64
    const bool use_decoder_bucket64 = latent_frames + kDecoderRightContextFrames
        <= kDecoderBucket64Frames;
    const bool use_flow_bucket48 = latent_frames <= kFlowBucket48Frames;
#else
    const bool use_bucket96 = latent_frames + kDecoderRightContextFrames
        <= kDecoderBucket96Frames;
#endif
    const char *flow_partition = kMainModelPartition;
    const char *flow_model = kFlowFullModel;
    size_t flow_bucket_frames = spec.max_latent_frames;
#if INFLECT_SHORT_VALUE64
    if (use_flow_bucket48) {
        flow_model = kFlowBucket48Model;
        flow_bucket_frames = kFlowBucket48Frames;
    } else {
        flow_partition = kFlowBucket96Partition;
        flow_model = kFlowBucket96Model;
        flow_bucket_frames = kDecoderBucket96Frames;
    }
#elif INFLECT_FLOW96
    if (use_bucket96) {
        flow_partition = kFlowBucket96Partition;
        flow_model = kFlowBucket96Model;
        flow_bucket_frames = kDecoderBucket96Frames;
    }
#endif
    StageMetrics flow_metrics;
    dl::Model *flow = load_model(flow_partition, flow_model, flow_metrics);
    auto &flow_inputs = flow->get_inputs();
    auto &flow_outputs = flow->get_outputs();
    if (!flow_metrics.cached) {
        print_tensor_map("flow", "input", flow_inputs);
        print_tensor_map("flow", "output", flow_outputs);
    }
    dl::TensorBase *flow_latent = find_tensor(
        flow_inputs, "Slice_0", spec.latent_channels * flow_bucket_frames);
    dl::TensorBase *flow_mask = find_tensor(
        flow_inputs, "Mul_1", flow_bucket_frames);
    dl::TensorBase *flow_output =
        flow_outputs.size() == 1 ? flow->get_output() : nullptr;
    if (
        flow_latent == nullptr || flow_mask == nullptr || flow_output == nullptr
        || latent_frames > flow_bucket_frames
        || flow_latent->get_exponent() != -3 || flow_mask->get_exponent() != -7
        || static_cast<size_t>(flow_output->get_size())
            != spec.latent_channels * flow_bucket_frames
        || flow_output->get_dtype() != dl::DATA_TYPE_INT8) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"flow_io\"}\n");
        heap_caps_free(flow_input_buffer);
        release_model(flow);
        return false;
    }
    const int flow_output_exponent = flow_output->get_exponent();
    auto *flow_latent_data = flow_latent->get_element_ptr<int8_t>();
    for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
        memcpy(
            flow_latent_data + channel * flow_bucket_frames,
            flow_input_buffer + channel * spec.max_latent_frames,
            flow_bucket_frames);
    }
    auto *flow_mask_data = flow_mask->get_element_ptr<int8_t>();
    memset(flow_mask_data, 0, flow_mask->get_bytes());
    memset(flow_mask_data, 127, latent_frames);
    started = esp_timer_get_time();
    flow->run(dl::RUNTIME_MODE_MULTI_CORE);
    flow_metrics.run_us = esp_timer_get_time() - started;
    vTaskDelay(1);
    memset(
        flow_input_buffer,
        0,
        spec.latent_channels * spec.max_latent_frames);
    const auto *flow_output_data = flow_output->get_element_ptr<int8_t>();
    for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
        memcpy(
            flow_input_buffer + channel * spec.max_latent_frames,
            flow_output_data + channel * flow_bucket_frames,
            flow_bucket_frames);
    }
    if (detailed_diagnostics) {
        const uint32_t flow_hash = fnv1a(
            reinterpret_cast<const uint8_t *>(flow_input_buffer),
            spec.latent_channels * spec.max_latent_frames);
        printf(
            "{\"type\":\"flow_result\",\"partition\":\"%s\","
            "\"model\":\"%s\",\"flow_bucket_frames\":%u,"
            "\"output_exponent\":%d,"
            "\"run_us\":%" PRId64 ","
            "\"output_fnv1a\":\"%08" PRIx32 "\"}\n",
            flow_partition,
            flow_model,
            static_cast<unsigned>(flow_bucket_frames),
            flow_output_exponent,
            flow_metrics.run_us,
            flow_hash);
    }
    release_model(flow);
    if (detailed_diagnostics) {
        print_heap_snapshot("after_flow_free");
    }

#if INFLECT_SHORT_VALUE64
    const bool use_decoder_tiles = false;
    const char *decoder_model = use_decoder_bucket64
        ? kDecoderBucket64Model
        : kDecoderBucket96Model;
#elif INFLECT_DECODER_BRIDGE96_69 || INFLECT_DECODER_TILE96
    const bool use_decoder_tiles = latent_frames
        > kDecoderTiledFirstPublishFrames;
    const char *decoder_model = kDecoderTiledFirstModel;
#else
    const bool use_decoder_tiles = false;
    const char *decoder_model = use_bucket96
        ? kDecoderBucket96Model
        : kDecoderFullModel;
#endif
    StageMetrics decoder_metrics;
    dl::Model *decoder = load_model(
        kMainModelPartition, decoder_model, decoder_metrics);
    auto &decoder_inputs = decoder->get_inputs();
    auto &decoder_outputs = decoder->get_outputs();
    if (!decoder_metrics.cached) {
        print_tensor_map("decoder", "input", decoder_inputs);
        print_tensor_map("decoder", "output", decoder_outputs);
    }
    dl::TensorBase *decoder_input = decoder->get_input();
    dl::TensorBase *decoder_output = decoder->get_output();
    auto decoder_input_dtype =
        decoder_input != nullptr ? decoder_input->get_dtype() : dl::DATA_TYPE_UNDEFINED;
    auto decoder_output_dtype = decoder_output != nullptr
        ? decoder_output->get_dtype()
        : dl::DATA_TYPE_UNDEFINED;
    if (
        decoder_input == nullptr || decoder_output == nullptr
        || (decoder_input_dtype != dl::DATA_TYPE_INT8
            && decoder_input_dtype != dl::DATA_TYPE_INT16)
        || decoder_input->get_size() % spec.latent_channels != 0
        || (decoder_output_dtype != dl::DATA_TYPE_INT8
            && decoder_output_dtype != dl::DATA_TYPE_INT16)) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"decoder_io\"}\n");
        heap_caps_free(flow_input_buffer);
        release_model(decoder);
        return false;
    }
    size_t decoder_bucket_frames =
        decoder_input->get_size() / spec.latent_channels;
#if INFLECT_DECODER_BRIDGE96_69 || INFLECT_DECODER_TILE96
    const bool decoder_shape_ok = use_decoder_tiles
        ? decoder_bucket_frames == kDecoderTiledFirstFrames
            && latent_frames > kDecoderTiledFirstPublishFrames
            && latent_frames <= kDecoderTiledMaximumFrames
        : latent_frames <= decoder_bucket_frames;
#else
    const bool decoder_shape_ok = latent_frames <= decoder_bucket_frames;
#endif
    if (
        !decoder_shape_ok
        || decoder_output->get_size() < decoder_bucket_frames * kDecoderHop) {
        printf(
            "{\"type\":\"full_pipeline_error\",\"stage\":\"decoder_shape\"," 
            "\"latent_frames\":%u,\"decoder_bucket_frames\":%u}\n",
            static_cast<unsigned>(latent_frames),
            static_cast<unsigned>(decoder_bucket_frames));
        heap_caps_free(flow_input_buffer);
        release_model(decoder);
        return false;
    }
    int decoder_input_shift = flow_output_exponent
        - decoder_input->get_exponent();
    if (decoder_input_shift < 0 || decoder_input_shift > 8) {
        printf(
            "{\"type\":\"full_pipeline_error\",\"stage\":\"decoder_input_scale\"," 
            "\"dtype\":\"%s\",\"exponent\":%d}\n",
            dl::dtype_to_string(decoder_input_dtype),
            decoder_input->get_exponent());
        heap_caps_free(flow_input_buffer);
        release_model(decoder);
        return false;
    }
    const size_t sample_count = static_cast<size_t>(latent_frames) * kDecoderHop;
    const size_t pause_sample_count =
        static_cast<size_t>(pause_after_ms) * kSampleRate / 1000;
    const size_t emitted_sample_count = sample_count + pause_sample_count;
    auto *pcm = static_cast<int16_t *>(heap_caps_aligned_alloc(
        16,
        emitted_sample_count * sizeof(int16_t),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (pcm == nullptr) {
        printf("{\"type\":\"full_pipeline_error\",\"stage\":\"pcm_alloc\"}\n");
        heap_caps_free(flow_input_buffer);
        release_model(decoder);
        return false;
    }

    auto fill_decoder_input = [&](size_t source_frame_start) {
        decoder->reset();
        if (decoder_input_dtype == dl::DATA_TYPE_INT8) {
            auto *decoder_input_data = decoder_input->get_element_ptr<int8_t>();
            for (size_t frame = 0; frame < decoder_bucket_frames; ++frame) {
                const size_t source_frame = source_frame_start + frame;
                for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
                    const int8_t source = source_frame < latent_frames
                        ? flow_input_buffer[
                            channel * spec.max_latent_frames + source_frame]
                        : 0;
                    const int32_t value = static_cast<int32_t>(source)
                        << decoder_input_shift;
                    decoder_input_data[frame * spec.latent_channels + channel]
                        = static_cast<int8_t>(
                            std::clamp<int32_t>(value, -128, 127));
                }
            }
        } else {
            auto *decoder_input_data = decoder_input->get_element_ptr<int16_t>();
            for (size_t frame = 0; frame < decoder_bucket_frames; ++frame) {
                const size_t source_frame = source_frame_start + frame;
                for (size_t channel = 0; channel < spec.latent_channels; ++channel) {
                    const int8_t source = source_frame < latent_frames
                        ? flow_input_buffer[
                            channel * spec.max_latent_frames + source_frame]
                        : 0;
                    const int32_t value = static_cast<int32_t>(source)
                        << decoder_input_shift;
                    decoder_input_data[frame * spec.latent_channels + channel]
                        = static_cast<int16_t>(
                            std::clamp<int32_t>(value, -32768, 32767));
                }
            }
        }
    };

    float output_scale = ldexpf(1.0f, decoder_output->get_exponent());
    const size_t decoder_first_bucket_frames = decoder_bucket_frames;
    const auto decoder_first_input_dtype = decoder_input_dtype;
    const auto decoder_first_output_dtype = decoder_output_dtype;
    const int decoder_first_input_exponent = decoder_input->get_exponent();
    const int decoder_first_output_exponent = decoder_output->get_exponent();
    size_t decoder_second_bucket_frames = 0;
    auto decoder_second_input_dtype = dl::DATA_TYPE_UNDEFINED;
    auto decoder_second_output_dtype = dl::DATA_TYPE_UNDEFINED;
    int decoder_second_input_exponent = 0;
    int decoder_second_output_exponent = 0;
    StageMetrics decoder_second_metrics;
    int16_t pcm_min = INT16_MAX;
    int16_t pcm_max = INT16_MIN;
    uint64_t pcm_square_sum = 0;
    bool use_int16_pcm_fast_path =
        decoder_output_dtype == dl::DATA_TYPE_INT16
        && decoder_output->get_exponent() == -16;
    auto copy_decoder_output = [&](size_t output_frame_start,
                                   size_t frame_count,
                                   size_t pcm_frame_start) {
        const size_t source_sample_start = output_frame_start * kDecoderHop;
        const size_t destination_sample_start = pcm_frame_start * kDecoderHop;
        const size_t copy_samples = frame_count * kDecoderHop;
        for (size_t offset = 0; offset < copy_samples; ++offset) {
            const size_t source_index = source_sample_start + offset;
            const size_t destination_index = destination_sample_start + offset;
            const int32_t raw_value = decoder_output_dtype == dl::DATA_TYPE_INT8
                ? decoder_output->get_element_ptr<int8_t>()[source_index]
                : decoder_output->get_element_ptr<int16_t>()[source_index];
            int32_t pcm_value = 0;
            if (use_int16_pcm_fast_path) {
                const int64_t scaled = static_cast<int64_t>(raw_value) * 32767;
                const int64_t magnitude = scaled >= 0 ? scaled : -scaled;
                const int32_t rounded = static_cast<int32_t>(
                    (magnitude + 32768) / 65536);
                pcm_value = scaled >= 0 ? rounded : -rounded;
            } else {
                const float value = std::clamp(
                    raw_value * output_scale, -1.0f, 1.0f);
                pcm_value = static_cast<int32_t>(lroundf(value * 32767.0f));
            }
            pcm[destination_index] = static_cast<int16_t>(pcm_value);
            pcm_min = std::min(pcm_min, pcm[destination_index]);
            pcm_max = std::max(pcm_max, pcm[destination_index]);
            pcm_square_sum += static_cast<uint64_t>(
                static_cast<int64_t>(pcm_value) * pcm_value);
        }
    };

#if INFLECT_DECODER_BRIDGE96_69 || INFLECT_DECODER_TILE96
    const size_t first_pcm_frames = use_decoder_tiles
        ? kDecoderTiledFirstPublishFrames
        : latent_frames;
#else
    const size_t first_pcm_frames = latent_frames;
#endif
    fill_decoder_input(0);
    started = esp_timer_get_time();
    decoder->run(dl::RUNTIME_MODE_MULTI_CORE);
    const int64_t decoder_first_run_us = esp_timer_get_time() - started;
    copy_decoder_output(0, first_pcm_frames, 0);
    const int64_t first_pcm_ready_at = esp_timer_get_time();
    const int64_t first_pcm_ready_us = first_pcm_ready_at - pipeline_started;

    int64_t decoder_second_run_us = 0;
    int64_t continuation_ready_us = 0;
    if (use_decoder_tiles) {
        vTaskDelay(1);
        release_model(decoder);
        decoder = load_model(
            kMainModelPartition, kDecoderTiledSecondModel, decoder_second_metrics);
        if (!decoder_second_metrics.cached) {
            print_tensor_map(
                kDecoderTiledSecondStage, "input", decoder->get_inputs());
            print_tensor_map(
                kDecoderTiledSecondStage, "output", decoder->get_outputs());
        }
        decoder_input = decoder->get_input();
        decoder_output = decoder->get_output();
        decoder_input_dtype = decoder_input != nullptr
            ? decoder_input->get_dtype()
            : dl::DATA_TYPE_UNDEFINED;
        decoder_output_dtype = decoder_output != nullptr
            ? decoder_output->get_dtype()
            : dl::DATA_TYPE_UNDEFINED;
        if (
            decoder_input == nullptr || decoder_output == nullptr
            || (decoder_input_dtype != dl::DATA_TYPE_INT8
                && decoder_input_dtype != dl::DATA_TYPE_INT16)
            || decoder_input->get_size() % spec.latent_channels != 0
            || (decoder_output_dtype != dl::DATA_TYPE_INT8
                && decoder_output_dtype != dl::DATA_TYPE_INT16)) {
            printf(
                "{\"type\":\"full_pipeline_error\"," 
                "\"stage\":\"%s_io\"}\n",
                kDecoderTiledSecondStage);
            heap_caps_free(flow_input_buffer);
            heap_caps_free(pcm);
            release_model(decoder);
            return false;
        }
        decoder_bucket_frames =
            decoder_input->get_size() / spec.latent_channels;
        if (
            decoder_bucket_frames != kDecoderTiledSecondFrames
            || decoder_output->get_size()
                < decoder_bucket_frames * kDecoderHop) {
            printf(
                "{\"type\":\"full_pipeline_error\"," 
                "\"stage\":\"%s_shape\"," 
                "\"decoder_bucket_frames\":%u}\n",
                kDecoderTiledSecondStage,
                static_cast<unsigned>(decoder_bucket_frames));
            heap_caps_free(flow_input_buffer);
            heap_caps_free(pcm);
            release_model(decoder);
            return false;
        }
        decoder_input_shift = -3 - decoder_input->get_exponent();
        if (decoder_input_shift < 0 || decoder_input_shift > 8) {
            printf(
                "{\"type\":\"full_pipeline_error\"," 
                "\"stage\":\"%s_input_scale\"," 
                "\"dtype\":\"%s\",\"exponent\":%d}\n",
                kDecoderTiledSecondStage,
                dl::dtype_to_string(decoder_input_dtype),
                decoder_input->get_exponent());
            heap_caps_free(flow_input_buffer);
            heap_caps_free(pcm);
            release_model(decoder);
            return false;
        }
        output_scale = ldexpf(1.0f, decoder_output->get_exponent());
        use_int16_pcm_fast_path =
            decoder_output_dtype == dl::DATA_TYPE_INT16
            && decoder_output->get_exponent() == -16;
        decoder_second_bucket_frames = decoder_bucket_frames;
        decoder_second_input_dtype = decoder_input_dtype;
        decoder_second_output_dtype = decoder_output_dtype;
        decoder_second_input_exponent = decoder_input->get_exponent();
        decoder_second_output_exponent = decoder_output->get_exponent();
#if INFLECT_DECODER_BRIDGE96_69
        const size_t second_source_start =
            latent_frames - kDecoderTiledSecondFrames;
        const size_t second_output_frame_start =
            first_pcm_frames - second_source_start;
#else
        const size_t second_source_start =
            first_pcm_frames - kDecoderRightContextFrames;
        const size_t second_output_frame_start = kDecoderRightContextFrames;
#endif
        const size_t continuation_frames = latent_frames - first_pcm_frames;
        if (
            second_source_start > first_pcm_frames
            || second_output_frame_start > kDecoderTiledSecondFrames
            || continuation_frames
                > kDecoderTiledSecondFrames - second_output_frame_start) {
            printf(
                "{\"type\":\"full_pipeline_error\"," 
                "\"stage\":\"decoder_tile_coverage\"}\n");
            heap_caps_free(flow_input_buffer);
            heap_caps_free(pcm);
            release_model(decoder);
            return false;
        }
        fill_decoder_input(second_source_start);
        started = esp_timer_get_time();
        decoder->run(dl::RUNTIME_MODE_MULTI_CORE);
        decoder_second_run_us = esp_timer_get_time() - started;
        copy_decoder_output(
            second_output_frame_start,
            continuation_frames,
            first_pcm_frames);
        continuation_ready_us = esp_timer_get_time() - first_pcm_ready_at;
    }
    decoder_metrics.run_us = decoder_first_run_us + decoder_second_run_us;
    heap_caps_free(flow_input_buffer);
    vTaskDelay(1);

    size_t edge_fade_samples = 0;
    if (fade_pcm_edges) {
        edge_fade_samples = std::min<size_t>(
            static_cast<size_t>(kSampleRate) * 5 / 1000,
            sample_count / 2);
        if (edge_fade_samples > 1) {
            const float denominator = static_cast<float>(edge_fade_samples - 1);
            for (size_t index = 0; index < edge_fade_samples; ++index) {
                const float ramp = static_cast<float>(index) / denominator;
                pcm[index] = static_cast<int16_t>(lroundf(pcm[index] * ramp));
                const size_t tail_index = sample_count - 1 - index;
                pcm[tail_index] = static_cast<int16_t>(
                    lroundf(pcm[tail_index] * ramp));
            }
            pcm_min = INT16_MAX;
            pcm_max = INT16_MIN;
            pcm_square_sum = 0;
            for (size_t index = 0; index < sample_count; ++index) {
                const int32_t value = pcm[index];
                pcm_min = std::min(pcm_min, pcm[index]);
                pcm_max = std::max(pcm_max, pcm[index]);
                pcm_square_sum += static_cast<uint64_t>(
                    static_cast<int64_t>(value) * value);
            }
        }
    }
    if (pause_sample_count != 0) {
        memset(
            pcm + sample_count,
            0,
            pause_sample_count * sizeof(int16_t));
    }

    const uint32_t pcm_hash = fnv1a(
        reinterpret_cast<const uint8_t *>(pcm),
        emitted_sample_count * sizeof(int16_t));
    const int64_t pipeline_us = esp_timer_get_time() - pipeline_started;
    const int64_t learned_run_us = acoustic_metrics.run_us
        + flow_metrics.run_us + decoder_metrics.run_us;
    const int64_t learned_first_pcm_us = acoustic_metrics.run_us
        + flow_metrics.run_us + decoder_first_run_us;
    const int64_t first_pcm_audio_us = static_cast<int64_t>(
        first_pcm_frames * kDecoderHop * 1000000ULL / kSampleRate);
    const int64_t playback_headroom_us = use_decoder_tiles
        ? first_pcm_audio_us - continuation_ready_us
        : 0;
    const double audio_seconds = static_cast<double>(sample_count) / kSampleRate;
    printf(
        "{\"type\":\"decoder_audio\",\"mode\":\"full_text_to_wave\"," 
        "\"model\":\"Inflect-Nano-v2\",\"token_count\":%u,"
        "\"token_fnv1a\":\"%08" PRIx32 "\","
        "\"latent_frames\":%u,\"decoder_bucket_frames\":%u,"
        "\"decoder_first_model\":\"%s\","
        "\"decoder_second_model\":\"%s\","
        "\"decoder_first_bucket_frames\":%u,"
        "\"decoder_second_bucket_frames\":%u,"
        "\"decoder_input_dtype\":\"%s\",\"decoder_input_exponent\":%d,"
        "\"decoder_output_dtype\":\"%s\",\"decoder_output_exponent\":%d,"
        "\"decoder_second_input_dtype\":\"%s\","
        "\"decoder_second_input_exponent\":%d,"
        "\"decoder_second_output_dtype\":\"%s\","
        "\"decoder_second_output_exponent\":%d,"
        "\"decoder_tiled\":%s,\"decoder_runs\":%u,"
        "\"decoder_first_run_us\":%" PRId64 ","
        "\"decoder_second_load_us\":%" PRId64 ","
        "\"decoder_second_run_us\":%" PRId64 ","
        "\"first_pcm_samples\":%u,\"first_pcm_audio_us\":%" PRId64 ","
        "\"first_pcm_ready_us\":%" PRId64 ","
        "\"learned_first_pcm_us\":%" PRId64 ","
        "\"continuation_ready_us\":%" PRId64 ","
        "\"playback_headroom_us\":%" PRId64 ","
        "\"samples\":%u,\"emitted_samples\":%u,"
        "\"edge_fade_samples\":%u,\"pause_after_samples\":%u,"
        "\"sample_rate\":%d,"
        "\"audio_seconds\":%.6f,\"acoustic_run_us\":%" PRId64 ","
        "\"flow_run_us\":%" PRId64 ",\"decoder_run_us\":%" PRId64 ","
        "\"learned_run_us\":%" PRId64 ",\"pipeline_us\":%" PRId64 ","
        "\"rtf\":%.6f,\"x_realtime\":%.3f,\"pcm_min\":%d,"
        "\"pcm_max\":%d,\"pcm_rms\":%.6f,"
        "\"pcm_fnv1a\":\"%08" PRIx32 "\",\"passed\":true}\n",
        static_cast<unsigned>(request.token_count),
        token_hash,
        static_cast<unsigned>(latent_frames),
        static_cast<unsigned>(decoder_first_bucket_frames),
        decoder_model,
        use_decoder_tiles ? kDecoderTiledSecondModel : "",
        static_cast<unsigned>(decoder_first_bucket_frames),
        static_cast<unsigned>(decoder_second_bucket_frames),
        dl::dtype_to_string(decoder_first_input_dtype),
        decoder_first_input_exponent,
        dl::dtype_to_string(decoder_first_output_dtype),
        decoder_first_output_exponent,
        dl::dtype_to_string(decoder_second_input_dtype),
        decoder_second_input_exponent,
        dl::dtype_to_string(decoder_second_output_dtype),
        decoder_second_output_exponent,
        use_decoder_tiles ? "true" : "false",
        use_decoder_tiles ? 2U : 1U,
        decoder_first_run_us,
        decoder_second_metrics.load_us,
        decoder_second_run_us,
        static_cast<unsigned>(first_pcm_frames * kDecoderHop),
        first_pcm_audio_us,
        first_pcm_ready_us,
        learned_first_pcm_us,
        continuation_ready_us,
        playback_headroom_us,
        static_cast<unsigned>(sample_count),
        static_cast<unsigned>(emitted_sample_count),
        static_cast<unsigned>(edge_fade_samples),
        static_cast<unsigned>(pause_sample_count),
        kSampleRate,
        audio_seconds,
        acoustic_metrics.run_us,
        flow_metrics.run_us,
        decoder_metrics.run_us,
        learned_run_us,
        pipeline_us,
        pipeline_us / (audio_seconds * 1000000.0),
        audio_seconds * 1000000.0 / pipeline_us,
        pcm_min,
        pcm_max,
        sqrt(static_cast<double>(pcm_square_sum) / sample_count) / 32767.0,
        pcm_hash);
#if INFLECT_PROFILE_DECODER
    printf("{\"type\":\"module_profile_start\",\"mode\":\"multi_core\"}\n");
    const int64_t profile_started = esp_timer_get_time();
    const auto module_info = decoder->get_module_info(dl::RUNTIME_MODE_MULTI_CORE);
    const int64_t profile_wall_us = esp_timer_get_time() - profile_started;
    decoder->print_module_info(module_info, true);
    printf(
        "{\"type\":\"module_profile_complete\",\"mode\":\"multi_core\"," 
        "\"wall_us\":%" PRId64 ",\"module_sum_us\":%" PRIu32 "}\n",
        profile_wall_us,
        module_info.at("total").latency);
#endif
#if INFLECT_EMIT_PCM_HEX
    emit_pcm_hex(pcm, emitted_sample_count, pcm_hash);
#endif
    const bool transport_ok = !emit_binary_pcm
        || emit_pcm_binary(pcm, emitted_sample_count, pcm_hash);
    heap_caps_free(pcm);
    release_model(decoder);
    if (detailed_diagnostics) {
        print_heap_snapshot("after_decoder_free");
    }
    if (outcome != nullptr) {
        outcome->failure = transport_ok
            ? PipelineFailure::None
            : PipelineFailure::Other;
        outcome->required_latent_frames = latent_frames;
        outcome->generated_pcm_samples = static_cast<uint32_t>(
            emitted_sample_count);
    }
    return transport_ok;
}

void print_hardware_inventory()
{
    esp_chip_info_t chip_info = {};
    uint32_t flash_size = 0;
    esp_chip_info(&chip_info);
    esp_flash_get_size(nullptr, &flash_size);
    printf(
        "{\"type\":\"hardware\",\"target\":\"%s\",\"idf\":\"%s\","
        "\"revision_major\":%u,\"revision_minor\":%u,\"cores\":%u,"
        "\"cpu_mhz\":%d,\"flash_bytes\":%" PRIu32 ",\"psram_bytes\":%u,"
        "\"psram_mhz\":%d,\"l2_cache_bytes\":%u}\n",
        CONFIG_IDF_TARGET,
        esp_get_idf_version(),
        static_cast<unsigned>(chip_info.revision / 100),
        static_cast<unsigned>(chip_info.revision % 100),
        static_cast<unsigned>(chip_info.cores),
        esp_clk_cpu_freq() / 1000000,
        flash_size,
        static_cast<unsigned>(esp_psram_get_size()),
        CONFIG_SPIRAM_SPEED,
        static_cast<unsigned>(CONFIG_CACHE_L2_CACHE_SIZE));
}

bool run_request(
    const AssetsView &assets,
    const PipelineRequest &request,
    bool prerequisites_ok,
    unsigned request_id,
    bool emit_binary_pcm)
{
    static bool detailed_request_completed = false;
    const bool detailed_diagnostics = !detailed_request_completed;
    if (detailed_diagnostics) {
        printf(
            "{\"type\":\"run_start\","
            "\"suite\":\"inflect_p4_full_pipeline_v1\"}\n");
        printf(
            "{\"type\":\"run_iteration\",\"iteration\":%u,\"total\":0}\n",
            request_id);
        print_heap_snapshot("boot");
    }
    const bool passed = prerequisites_ok
        && run_full_pipeline(
            assets, request, emit_binary_pcm, detailed_diagnostics);
    if (detailed_diagnostics) {
        print_heap_snapshot("complete");
        detailed_request_completed = passed;
    }
    printf(
        "{\"type\":\"run_complete\",\"suite\":\"inflect_p4_full_pipeline_v1\","
        "\"iteration\":%u,\"passed\":%s}\n",
        request_id,
        passed ? "true" : "false");
    fflush(stdout);
    return passed;
}

#if INFLECT_COMMAND_LOOP
bool parse_say_command(
    char *command,
    const FullPipelineAssetsHeader &spec,
    std::vector<uint16_t> &tokens,
    uint32_t &token_hash)
{
    char *save = nullptr;
    const char *verb = strtok_r(command, " ", &save);
    const char *count_text = strtok_r(nullptr, " ", &save);
    const char *hash_text = strtok_r(nullptr, " ", &save);
    char *token_text = strtok_r(nullptr, " ", &save);
    if (
        verb == nullptr || strcmp(verb, "SAY") != 0 || count_text == nullptr
        || hash_text == nullptr || token_text == nullptr
        || strtok_r(nullptr, " ", &save) != nullptr) {
        printf("{\"type\":\"command_error\",\"stage\":\"say_syntax\"}\n");
        return false;
    }

    char *end = nullptr;
    const unsigned long requested_count = strtoul(count_text, &end, 10);
    if (
        *count_text == '\0' || *end != '\0' || requested_count == 0
        || requested_count > spec.max_text_tokens) {
        printf(
            "{\"type\":\"command_error\",\"stage\":\"say_token_count\","
            "\"maximum\":%u}\n",
            static_cast<unsigned>(spec.max_text_tokens));
        return false;
    }
    const unsigned long requested_hash = strtoul(hash_text, &end, 16);
    if (*hash_text == '\0' || *end != '\0' || requested_hash > UINT32_MAX) {
        printf("{\"type\":\"command_error\",\"stage\":\"say_hash\"}\n");
        return false;
    }

    tokens.clear();
    tokens.reserve(requested_count);
    char *list_save = nullptr;
    for (
        char *token = strtok_r(token_text, ",", &list_save);
        token != nullptr;
        token = strtok_r(nullptr, ",", &list_save)) {
        const unsigned long value = strtoul(token, &end, 10);
        if (
            *token == '\0' || *end != '\0' || value >= spec.vocab_size
            || tokens.size() >= requested_count) {
            printf("{\"type\":\"command_error\",\"stage\":\"say_token\"}\n");
            return false;
        }
        tokens.push_back(static_cast<uint16_t>(value));
    }
    token_hash = fnv1a(
        reinterpret_cast<const uint8_t *>(tokens.data()),
        tokens.size() * sizeof(uint16_t));
    if (tokens.size() != requested_count || token_hash != requested_hash) {
        printf(
            "{\"type\":\"command_error\",\"stage\":\"say_integrity\","
            "\"tokens\":%u,\"fnv1a\":\"%08" PRIx32 "\"}\n",
            static_cast<unsigned>(tokens.size()),
            token_hash);
        return false;
    }
    return true;
}

bool initialize_command_console()
{
    fflush(stdout);
    if (!uart_is_driver_installed(kConsoleUart)) {
        const uart_config_t config = {
            .baud_rate = CONFIG_ESP_CONSOLE_UART_BAUDRATE,
            .data_bits = UART_DATA_8_BITS,
            .parity = UART_PARITY_DISABLE,
            .stop_bits = UART_STOP_BITS_1,
            .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
            .rx_flow_ctrl_thresh = 0,
            .source_clk = UART_SCLK_DEFAULT,
            .flags = {},
        };
        if (
            uart_driver_install(
                kConsoleUart, 1024, 0, 0, nullptr, 0) != ESP_OK
            || uart_param_config(kConsoleUart, &config) != ESP_OK) {
            return false;
        }
    }
    uart_vfs_dev_port_set_rx_line_endings(
        CONFIG_ESP_CONSOLE_UART_NUM,
        ESP_LINE_ENDINGS_LF);
    uart_vfs_dev_port_set_tx_line_endings(
        CONFIG_ESP_CONSOLE_UART_NUM,
        ESP_LINE_ENDINGS_LF);
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
    setvbuf(stdin, nullptr, _IONBF, 0);
    return true;
}

void print_json_string(const std::string &value)
{
    putchar('"');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            fputs("\\\"", stdout);
            break;
        case '\\':
            fputs("\\\\", stdout);
            break;
        case '\b':
            fputs("\\b", stdout);
            break;
        case '\f':
            fputs("\\f", stdout);
            break;
        case '\n':
            fputs("\\n", stdout);
            break;
        case '\r':
            fputs("\\r", stdout);
            break;
        case '\t':
            fputs("\\t", stdout);
            break;
        default:
            if (byte < 0x20) {
                printf("\\u%04x", static_cast<unsigned>(byte));
            } else {
                putchar(byte);
            }
        }
    }
    putchar('"');
}

void print_text_frontend_result(
    const std::string &raw_text,
    const inflect::text::FrontendResult &frontend,
    uint32_t token_hash)
{
    printf(
        "{\"type\":\"text_frontend\",\"engine\":\"espeak-ng-1.52.0\"," 
        "\"input_bytes\":%u,\"normalized_text\":",
        static_cast<unsigned>(raw_text.size()));
    print_json_string(frontend.normalized_text);
    fputs(",\"phonemes\":", stdout);
    print_json_string(frontend.phonemes);
    printf(
        ",\"token_count\":%u,\"token_fnv1a\":\"%08" PRIx32 "\"," 
        "\"normalize_us\":%" PRId64 ",\"phonemize_us\":%" PRId64 ","
        "\"tokenize_us\":%" PRId64 ",\"tokens\":[",
        static_cast<unsigned>(frontend.tokens.size()),
        token_hash,
        frontend.normalize_us,
        frontend.phonemize_us,
        frontend.tokenize_us);
    for (size_t index = 0; index < frontend.tokens.size(); ++index) {
        printf("%s%u", index == 0 ? "" : ",", frontend.tokens[index]);
    }
    fputs("]}\n", stdout);
}

void print_text_segment_plan(
    const std::string &raw_text,
    const inflect::text::SegmentedFrontendResult &plan)
{
    printf(
        "{\"type\":\"text_segment_plan\",\"engine\":\"espeak-ng-1.52.0\","
        "\"input_bytes\":%u,\"normalized_text\":",
        static_cast<unsigned>(raw_text.size()));
    print_json_string(plan.normalized_text);
    printf(
        ",\"normalize_us\":%" PRId64 ",\"plan_us\":%" PRId64 ","
        "\"segment_count\":%u,\"worker_stack_high_water_bytes\":%u,"
        "\"segments\":[",
        plan.normalize_us,
        plan.plan_us,
        static_cast<unsigned>(plan.segments.size()),
        static_cast<unsigned>(
            inflect::text::last_worker_stack_high_water_bytes()));
    for (size_t index = 0; index < plan.segments.size(); ++index) {
        const auto &segment = plan.segments[index];
        const uint32_t token_hash = fnv1a(
            reinterpret_cast<const uint8_t *>(segment.frontend.tokens.data()),
            segment.frontend.tokens.size() * sizeof(uint16_t));
        printf("%s{\"index\":%u,\"normalized_text\":", index == 0 ? "" : ",",
               static_cast<unsigned>(index));
        print_json_string(segment.source_text);
        fputs(",\"synthesis_text\":", stdout);
        print_json_string(segment.frontend.normalized_text);
        fputs(",\"phonemes\":", stdout);
        print_json_string(segment.frontend.phonemes);
        printf(
            ",\"token_count\":%u,\"token_fnv1a\":\"%08" PRIx32 "\","
            "\"boundary\":\"%s\",\"pause_after_ms\":%u,"
            "\"boundary_punctuation_elided\":%s}",
            static_cast<unsigned>(segment.frontend.tokens.size()),
            token_hash,
            inflect::text::segment_boundary_name(segment.boundary),
            static_cast<unsigned>(segment.pause_after_ms),
            segment.boundary_punctuation_elided ? "true" : "false");
    }
    fputs("]}\n", stdout);
}

bool run_segmented_text_request(
    const AssetsView &assets,
    const std::string &raw_text,
    unsigned request_id)
{
    constexpr size_t kMaximumRuntimeSegments = 64;
    constexpr size_t kMaximumDurationReplans = 32;
    inflect::text::SegmentedFrontendResult plan;
    std::string error;
    if (!inflect::text::plan_segments_in_worker(
            raw_text,
            assets.header->vocab_size,
            assets.header->max_text_tokens,
            plan,
            error)) {
        printf(
            "{\"type\":\"command_error\","
            "\"stage\":\"long_text_plan\",\"error\":");
        print_json_string(error);
        fputs("}\n", stdout);
        return false;
    }

    print_text_segment_plan(raw_text, plan);
    printf(
        "{\"type\":\"long_text_start\",\"request_id\":%u,"
        "\"planned_segments\":%u}\n",
        request_id,
        static_cast<unsigned>(plan.segments.size()));
    const int64_t started = esp_timer_get_time();
    size_t segment_index = 0;
    size_t replan_count = 0;
    uint64_t emitted_samples = 0;
    while (segment_index < plan.segments.size()) {
        auto &segment = plan.segments[segment_index];
        const uint32_t token_hash = fnv1a(
            reinterpret_cast<const uint8_t *>(segment.frontend.tokens.data()),
            segment.frontend.tokens.size() * sizeof(uint16_t));
        printf(
            "{\"type\":\"long_text_segment_start\","
            "\"request_id\":%u,\"index\":%u,\"current_segments\":%u,"
            "\"token_count\":%u,\"token_fnv1a\":\"%08" PRIx32 "\","
            "\"boundary\":\"%s\",\"pause_after_ms\":%u,"
            "\"boundary_punctuation_elided\":%s,\"text\":",
            request_id,
            static_cast<unsigned>(segment_index),
            static_cast<unsigned>(plan.segments.size()),
            static_cast<unsigned>(segment.frontend.tokens.size()),
            token_hash,
            inflect::text::segment_boundary_name(segment.boundary),
            static_cast<unsigned>(segment.pause_after_ms),
            segment.boundary_punctuation_elided ? "true" : "false");
        print_json_string(segment.source_text);
        fputs(",\"synthesis_text\":", stdout);
        print_json_string(segment.frontend.normalized_text);
        fputs("}\n", stdout);

        const PipelineRequest request = {
            .tokens = segment.frontend.tokens.data(),
            .token_count = static_cast<uint32_t>(
                segment.frontend.tokens.size()),
            .expected_latent_frames = 0,
            .token_fnv1a = token_hash,
            .validate_reference = false,
        };
        PipelineOutcome outcome;
        if (run_full_pipeline(
                assets,
                request,
                true,
                false,
                &outcome,
                true,
                segment.pause_after_ms)) {
            emitted_samples += outcome.generated_pcm_samples;
            printf(
                "{\"type\":\"long_text_segment_complete\","
                "\"request_id\":%u,\"index\":%u,"
                "\"latent_frames\":%u,\"emitted_samples\":%u}\n",
                request_id,
                static_cast<unsigned>(segment_index),
                static_cast<unsigned>(outcome.required_latent_frames),
                static_cast<unsigned>(outcome.generated_pcm_samples));
            ++segment_index;
            continue;
        }
        if (outcome.failure != PipelineFailure::DurationOverflow) {
            printf(
                "{\"type\":\"long_text_complete\","
                "\"request_id\":%u,\"passed\":false,"
                "\"failed_segment\":%u}\n",
                request_id,
                static_cast<unsigned>(segment_index));
            return false;
        }
        if (
            replan_count >= kMaximumDurationReplans
            || segment.frontend.tokens.size() <= 3) {
            printf(
                "{\"type\":\"command_error\","
                "\"stage\":\"duration_replan_limit\","
                "\"required_frames\":%u}\n",
                static_cast<unsigned>(outcome.required_latent_frames));
            return false;
        }

        const uint32_t scaled_limit = static_cast<uint32_t>(
            segment.frontend.tokens.size()
            * static_cast<uint64_t>(runtime_max_latent_frames(*assets.header))
            / outcome.required_latent_frames);
        const uint32_t reduced_limit = std::min<uint32_t>(
            static_cast<uint32_t>(segment.frontend.tokens.size() - 2),
            scaled_limit);
        inflect::text::SegmentedFrontendResult replacement;
        if (
            reduced_limit < 3
            || !inflect::text::plan_normalized_segments_in_worker(
                segment.source_text,
                assets.header->vocab_size,
                reduced_limit,
                replacement,
                error)
            || replacement.segments.size() < 2) {
            printf(
                "{\"type\":\"command_error\","
                "\"stage\":\"duration_unsplittable\","
                "\"required_frames\":%u,\"token_limit\":%u,\"error\":",
                static_cast<unsigned>(outcome.required_latent_frames),
                static_cast<unsigned>(reduced_limit));
            print_json_string(error);
            fputs("}\n", stdout);
            return false;
        }
        if (
            plan.segments.size() - 1 + replacement.segments.size()
            > kMaximumRuntimeSegments) {
            printf(
                "{\"type\":\"command_error\","
                "\"stage\":\"too_many_runtime_segments\"}\n");
            return false;
        }

        const auto inherited_boundary = segment.boundary;
        const uint32_t inherited_pause_ms = segment.pause_after_ms;
        const size_t replacement_count = replacement.segments.size();
        replacement.segments.back().boundary = inherited_boundary;
        replacement.segments.back().pause_after_ms = inherited_pause_ms;
        printf(
            "{\"type\":\"long_text_segment_replan\","
            "\"request_id\":%u,\"index\":%u,"
            "\"required_frames\":%u,\"previous_tokens\":%u,"
            "\"new_token_limit\":%u,\"replacement_segments\":%u}\n",
            request_id,
            static_cast<unsigned>(segment_index),
            static_cast<unsigned>(outcome.required_latent_frames),
            static_cast<unsigned>(segment.frontend.tokens.size()),
            static_cast<unsigned>(reduced_limit),
            static_cast<unsigned>(replacement_count));
        plan.segments.erase(plan.segments.begin() + segment_index);
        plan.segments.insert(
            plan.segments.begin() + segment_index,
            std::make_move_iterator(replacement.segments.begin()),
            std::make_move_iterator(replacement.segments.end()));
        ++replan_count;
    }

    printf(
        "{\"type\":\"long_text_complete\",\"request_id\":%u,"
        "\"passed\":true,\"segments\":%u,\"duration_replans\":%u,"
        "\"emitted_samples\":%" PRIu64 ",\"elapsed_us\":%" PRId64 "}\n",
        request_id,
        static_cast<unsigned>(plan.segments.size()),
        static_cast<unsigned>(replan_count),
        emitted_samples,
        esp_timer_get_time() - started);
    return true;
}
#endif

}  // namespace

void inflect_service_task(void *)
{
    register_custom_modules();
#if INFLECT_SHORT_VALUE64
    bool package_ok = verify_model_package(
        kMainModelPartition, 4, "short64-flow48");
#elif INFLECT_DECODER_BRIDGE96_69
    bool package_ok = verify_model_package(
        kMainModelPartition, 4, "tile69-v1");
#elif INFLECT_DECODER_TILE96
    bool package_ok = verify_model_package(
        kMainModelPartition, 4, "tile10065-v1");
#else
    bool package_ok = verify_model_package(kMainModelPartition, 4);
#endif
#if INFLECT_FLOW96
    package_ok = verify_model_package(
        kFlowBucket96Partition, 1, "flow96-v1") && package_ok;
#endif
    AssetsView assets = {};
    const bool assets_ok = parse_assets(assets);
    const PipelineRequest compiled_request = {
        .tokens = assets.tokens,
        .token_count = assets_ok ? assets.header->token_count : 0,
        .expected_latent_frames = assets_ok
            ? assets.header->expected_latent_frames
            : 0,
        .token_fnv1a = assets_ok ? assets.header->token_fnv1a : 0,
        .validate_reference = true,
    };
    print_hardware_inventory();
    printf(
        "{\"type\":\"assets\",\"bytes\":%u,\"parsed\":%s,"
        "\"token_count\":%u,\"expected_latent_frames\":%u}\n",
        static_cast<unsigned>(inflect_full_assets_end - inflect_full_assets_start),
        assets_ok ? "true" : "false",
        assets_ok ? static_cast<unsigned>(assets.header->token_count) : 0,
        assets_ok ? static_cast<unsigned>(assets.header->expected_latent_frames) : 0);

#if INFLECT_COMMAND_LOOP
    const bool console_ok = initialize_command_console();
    int64_t frontend_init_us = 0;
    std::string frontend_error;
    const bool frontend_ok = console_ok && inflect::text::initialize_in_worker(
        frontend_init_us,
        frontend_error);
    printf(
        "{\"type\":\"text_frontend_init\",\"engine\":\"espeak-ng-1.52.0\"," 
        "\"ready\":%s,\"init_us\":%" PRId64 ",\"error\":",
        frontend_ok ? "true" : "false",
        frontend_init_us);
    print_json_string(frontend_error);
    fputs("}\n", stdout);
    printf(
        "{\"type\":\"service_ready\",\"ready\":%s,"
        "\"commands\":[\"RUN\",\"WAV\",\"TEXT\",\"TEXT_RUN\","
        "\"TEXT_LONG\","
        "\"FRONTEND\",\"PLAN_TEXT\",\"SAY\",\"STATUS\""
#if INFLECT_BENCH_INTERLEAVE
        ",\"BENCH_INTERLEAVE\""
#endif
#if INFLECT_BENCH_PRIMITIVES
        ",\"BENCH_PRIMITIVES\""
#endif
#if INFLECT_BENCH_STUDENTS
        ",\"BENCH_STUDENTS\",\"BENCH_STUDENT_BRANCH\","
        "\"PROFILE_STUDENT_BRANCH\",\"INSPECT_STUDENT_LUTS\","
        "\"INSPECT_DECODER_LUTS\",\"INSPECT_DECODER_CONVS\","
        "\"INSPECT_DECODER_ACTIVATION_EDGES\","
        "\"TEST_DECODER_LUT_KERNELS\""
#endif
        "]}\n",
        package_ok && assets_ok && console_ok && frontend_ok ? "true" : "false");
    fflush(stdout);
    char command[512] = {};
    std::vector<uint16_t> request_tokens;
    unsigned request_id = 0;
    while (console_ok && fgets(command, sizeof(command), stdin) != nullptr) {
        command[strcspn(command, "\r\n")] = '\0';
        if (strcmp(command, "STATUS") == 0) {
            print_heap_snapshot("service_status");
#if INFLECT_BENCH_INTERLEAVE
        } else if (strcmp(command, "BENCH_INTERLEAVE") == 0) {
            inflect::polyphase::run_interleave_benchmarks();
#endif
#if INFLECT_BENCH_PRIMITIVES
        } else if (strcmp(command, "BENCH_PRIMITIVES") == 0) {
            inflect::bench::run_decoder_primitive_benchmarks();
#endif
#if INFLECT_BENCH_STUDENTS
        } else if (strcmp(command, "BENCH_STUDENTS") == 0) {
            inflect::bench::run_student_decoder_benchmarks();
        } else if (strcmp(command, "BENCH_STUDENT_BRANCH") == 0) {
            inflect::bench::run_student_branch_benchmark();
        } else if (strcmp(command, "PROFILE_STUDENT_BRANCH") == 0) {
            inflect::bench::profile_student_branch();
        } else if (strcmp(command, "INSPECT_STUDENT_LUTS") == 0) {
            inflect::bench::inspect_student_branch_luts();
        } else if (strcmp(command, "INSPECT_DECODER_LUTS") == 0) {
            inflect::bench::inspect_exact_decoder_luts();
        } else if (strcmp(command, "INSPECT_DECODER_CONVS") == 0) {
            inflect::bench::inspect_exact_decoder_convs();
        } else if (strcmp(command, "INSPECT_DECODER_ACTIVATION_EDGES") == 0) {
            inflect::bench::inspect_exact_decoder_activation_edges();
        } else if (strcmp(command, "TEST_DECODER_LUT_KERNELS") == 0) {
            inflect::bench::test_exact_decoder_lut_kernels();
#endif
        } else if (strcmp(command, "RUN") == 0 || strcmp(command, "WAV") == 0) {
            ++request_id;
            run_request(
                assets,
                compiled_request,
                package_ok && assets_ok,
                request_id,
                strcmp(command, "WAV") == 0);
        } else if (strncmp(command, "SAY ", 4) == 0) {
            uint32_t token_hash = 0;
            if (assets_ok && parse_say_command(
                    command, *assets.header, request_tokens, token_hash)) {
                ++request_id;
                const PipelineRequest text_request = {
                    .tokens = request_tokens.data(),
                    .token_count = static_cast<uint32_t>(request_tokens.size()),
                    .expected_latent_frames = 0,
                    .token_fnv1a = token_hash,
                    .validate_reference = false,
                };
                printf(
                    "{\"type\":\"text_request\",\"token_count\":%u,"
                    "\"token_fnv1a\":\"%08" PRIx32 "\"}\n",
                    static_cast<unsigned>(request_tokens.size()),
                    token_hash);
                run_request(
                    assets,
                    text_request,
                    package_ok,
                    request_id,
                    true);
            }
        } else if (strncmp(command, "PLAN_TEXT ", 10) == 0) {
            inflect::text::SegmentedFrontendResult plan;
            std::string error;
            const std::string raw_text(command + 10);
            if (!frontend_ok || !assets_ok) {
                printf(
                    "{\"type\":\"command_error\","
                    "\"stage\":\"text_frontend_unavailable\"}\n");
            } else if (!inflect::text::plan_segments_in_worker(
                    raw_text,
                    assets.header->vocab_size,
                    assets.header->max_text_tokens,
                    plan,
                    error)) {
                printf(
                    "{\"type\":\"command_error\","
                    "\"stage\":\"text_segment_plan\",\"error\":");
                print_json_string(error);
                fputs("}\n", stdout);
            } else {
                print_text_segment_plan(raw_text, plan);
            }
        } else if (strncmp(command, "TEXT_LONG ", 10) == 0) {
            const std::string raw_text(command + 10);
            if (!frontend_ok || !assets_ok || !package_ok) {
                printf(
                    "{\"type\":\"command_error\","
                    "\"stage\":\"long_text_unavailable\"}\n");
            } else {
                ++request_id;
                run_segmented_text_request(assets, raw_text, request_id);
            }
        } else if (strncmp(command, "FRONTEND ", 9) == 0) {
            inflect::text::FrontendResult frontend;
            std::string error;
            const std::string raw_text(command + 9);
            if (!frontend_ok || !assets_ok) {
                printf(
                    "{\"type\":\"command_error\"," 
                    "\"stage\":\"text_frontend_unavailable\"}\n");
            } else if (!inflect::text::tokenize_in_worker(
                    raw_text,
                    assets.header->vocab_size,
                    1023,
                    frontend,
                    error)) {
                printf(
                    "{\"type\":\"command_error\"," 
                    "\"stage\":\"text_frontend\",\"error\":");
                print_json_string(error);
                fputs("}\n", stdout);
            } else {
                const uint32_t token_hash = fnv1a(
                    reinterpret_cast<const uint8_t *>(frontend.tokens.data()),
                    frontend.tokens.size() * sizeof(uint16_t));
                print_text_frontend_result(raw_text, frontend, token_hash);
            }
        } else if (
            strncmp(command, "TEXT ", 5) == 0
            || strncmp(command, "TEXT_RUN ", 9) == 0
        ) {
            inflect::text::FrontendResult frontend;
            std::string error;
            const bool emit_binary_pcm = strncmp(command, "TEXT ", 5) == 0;
            const std::string raw_text(command + (emit_binary_pcm ? 5 : 9));
            if (!frontend_ok || !assets_ok) {
                printf(
                    "{\"type\":\"command_error\","
                    "\"stage\":\"text_frontend_unavailable\"}\n");
            } else if (!inflect::text::tokenize_in_worker(
                    raw_text,
                    assets.header->vocab_size,
                    assets.header->max_text_tokens,
                    frontend,
                    error)) {
                printf(
                    "{\"type\":\"command_error\","
                    "\"stage\":\"text_frontend\",\"error\":");
                print_json_string(error);
                fputs("}\n", stdout);
            } else {
                const uint32_t token_hash = fnv1a(
                    reinterpret_cast<const uint8_t *>(frontend.tokens.data()),
                    frontend.tokens.size() * sizeof(uint16_t));
                print_text_frontend_result(raw_text, frontend, token_hash);
                ++request_id;
                const PipelineRequest text_request = {
                    .tokens = frontend.tokens.data(),
                    .token_count = static_cast<uint32_t>(frontend.tokens.size()),
                    .expected_latent_frames = 0,
                    .token_fnv1a = token_hash,
                    .validate_reference = false,
                };
                run_request(
                    assets,
                    text_request,
                    package_ok,
                    request_id,
                    emit_binary_pcm);
            }
        } else if (command[0] != '\0') {
            printf(
                "{\"type\":\"command_error\",\"command\":\"%s\"}\n",
                command);
        }
        printf(
            "{\"type\":\"service_ready\",\"ready\":true,"
            "\"request_count\":%u}\n",
            request_id);
        fflush(stdout);
    }
#else
    static_assert(INFLECT_REPEAT_RUNS >= 1, "INFLECT_REPEAT_RUNS must be positive");
    for (int iteration = 1; iteration <= INFLECT_REPEAT_RUNS; ++iteration) {
        printf("{\"type\":\"run_start\",\"suite\":\"inflect_p4_full_pipeline_v1\"}\n");
        printf(
            "{\"type\":\"run_iteration\",\"iteration\":%d,\"total\":%d}\n",
            iteration,
            INFLECT_REPEAT_RUNS);
        print_heap_snapshot("boot");
        const bool passed = assets_ok && package_ok
            && run_full_pipeline(assets, compiled_request);
        print_heap_snapshot("complete");
        printf(
            "{\"type\":\"run_complete\",\"suite\":\"inflect_p4_full_pipeline_v1\","
            "\"iteration\":%d,\"passed\":%s}\n",
            iteration,
            passed ? "true" : "false");
        fflush(stdout);
        if (!passed) {
            break;
        }
    }
#endif
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" void app_main(void)
{
    TaskHandle_t service_task_handle = nullptr;
    const BaseType_t created = xTaskCreatePinnedToCoreWithCaps(
        inflect_service_task,
        "inflect_service",
        16 * 1024,
        nullptr,
        uxTaskPriorityGet(nullptr),
        &service_task_handle,
        0,
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (created != pdPASS) {
        fputs(
            "{\"type\":\"boot_error\","
            "\"stage\":\"create_service_task\"}\n",
            stdout);
        abort();
    }
}
