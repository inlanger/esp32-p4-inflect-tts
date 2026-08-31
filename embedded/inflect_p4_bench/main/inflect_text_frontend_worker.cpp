#include "inflect_text_frontend_worker.h"

#include <algorithm>
#include <cstdint>

#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"

namespace inflect::text {
namespace {

constexpr uint32_t kFrontendStackBytes = 48 * 1024;

enum class Operation {
    Initialize,
    Tokenize,
    PlanSegments,
    PlanNormalizedSegments,
};

struct FrontendJob {
    Operation operation;
    TaskHandle_t caller;
    const std::string *raw_text = nullptr;
    uint32_t vocabulary_size = 0;
    uint32_t maximum_tokens = 0;
    FrontendResult *result = nullptr;
    SegmentedFrontendResult *segmented_result = nullptr;
    int64_t *elapsed_us = nullptr;
    std::string *error = nullptr;
    bool passed = false;
    uint32_t stack_high_water_bytes = 0;
};

uint32_t g_last_stack_high_water_bytes = 0;

void frontend_task(void *context)
{
    FrontendJob &job = *static_cast<FrontendJob *>(context);
    if (job.operation == Operation::Initialize) {
        job.passed = initialize(*job.elapsed_us, *job.error);
    } else if (job.operation == Operation::PlanSegments) {
        job.passed = plan_segments(
            *job.raw_text,
            job.vocabulary_size,
            job.maximum_tokens,
            *job.segmented_result,
            *job.error);
    } else if (job.operation == Operation::PlanNormalizedSegments) {
        job.passed = plan_normalized_segments(
            *job.raw_text,
            job.vocabulary_size,
            job.maximum_tokens,
            *job.segmented_result,
            *job.error);
    } else {
        job.passed = tokenize(
            *job.raw_text,
            job.vocabulary_size,
            job.maximum_tokens,
            *job.result,
            *job.error);
    }
    job.stack_high_water_bytes = uxTaskGetStackHighWaterMark(nullptr);
    xTaskNotifyGive(job.caller);
    vTaskSuspend(nullptr);
}

bool execute(FrontendJob &job)
{
    while (ulTaskNotifyTake(pdTRUE, 0) != 0) {
    }
    job.caller = xTaskGetCurrentTaskHandle();
    StackType_t *stack = static_cast<StackType_t *>(heap_caps_malloc(
        kFrontendStackBytes * sizeof(StackType_t),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    StaticTask_t *task_buffer = static_cast<StaticTask_t *>(heap_caps_calloc(
        1,
        sizeof(StaticTask_t),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (stack == nullptr || task_buffer == nullptr) {
        heap_caps_free(task_buffer);
        heap_caps_free(stack);
        *job.error = "could not allocate temporary eSpeak task";
        return false;
    }
    const UBaseType_t priority = std::min<UBaseType_t>(
        uxTaskPriorityGet(nullptr) + 1,
        configMAX_PRIORITIES - 1);
    TaskHandle_t worker = xTaskCreateStaticPinnedToCore(
        frontend_task,
        "inflect_frontend",
        kFrontendStackBytes,
        &job,
        priority,
        stack,
        task_buffer,
        0);
    if (worker == nullptr) {
        heap_caps_free(task_buffer);
        heap_caps_free(stack);
        *job.error = "could not allocate temporary eSpeak task";
        return false;
    }
    if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == 0) {
        vTaskDelete(worker);
        heap_caps_free(task_buffer);
        heap_caps_free(stack);
        *job.error = "temporary eSpeak task did not complete";
        return false;
    }

    g_last_stack_high_water_bytes = job.stack_high_water_bytes;
    vTaskDelete(worker);
    heap_caps_free(task_buffer);
    heap_caps_free(stack);
    return job.passed;
}

}  // namespace

bool initialize_in_worker(int64_t &elapsed_us, std::string &error)
{
    FrontendJob job = {
        .operation = Operation::Initialize,
        .caller = nullptr,
        .elapsed_us = &elapsed_us,
        .error = &error,
    };
    return execute(job);
}

bool tokenize_in_worker(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    FrontendResult &result,
    std::string &error)
{
    FrontendJob job = {
        .operation = Operation::Tokenize,
        .caller = nullptr,
        .raw_text = &raw_text,
        .vocabulary_size = vocabulary_size,
        .maximum_tokens = maximum_tokens,
        .result = &result,
        .error = &error,
    };
    return execute(job);
}

bool plan_segments_in_worker(
    const std::string &raw_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error)
{
    FrontendJob job = {
        .operation = Operation::PlanSegments,
        .caller = nullptr,
        .raw_text = &raw_text,
        .vocabulary_size = vocabulary_size,
        .maximum_tokens = maximum_tokens,
        .segmented_result = &result,
        .error = &error,
    };
    return execute(job);
}

bool plan_normalized_segments_in_worker(
    const std::string &normalized_text,
    uint32_t vocabulary_size,
    uint32_t maximum_tokens,
    SegmentedFrontendResult &result,
    std::string &error)
{
    FrontendJob job = {
        .operation = Operation::PlanNormalizedSegments,
        .caller = nullptr,
        .raw_text = &normalized_text,
        .vocabulary_size = vocabulary_size,
        .maximum_tokens = maximum_tokens,
        .segmented_result = &result,
        .error = &error,
    };
    return execute(job);
}

uint32_t last_worker_stack_high_water_bytes()
{
    return g_last_stack_high_water_bytes;
}

}  // namespace inflect::text
