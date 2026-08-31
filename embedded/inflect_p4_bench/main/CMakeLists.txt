set(INFLECT_ASSETS_BIN "${CMAKE_CURRENT_LIST_DIR}/inflect_full_assets.bin")
set(INFLECT_ASSETS_ASM "${CMAKE_BINARY_DIR}/inflect_full_assets_aligned.S")
set(
    INFLECT_ESPDL_TEXT_PAD_BYTES
    0
    CACHE STRING
    "Unexecuted flash-text padding before ESP-DL for cache-index experiments")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/inflect_full_assets_aligned.S.in"
    "${INFLECT_ASSETS_ASM}"
    @ONLY)

idf_component_register(
    SRCS
        "inflect_full_pipeline.cpp"
        "inflect_text_frontend.cpp"
        "inflect_text_frontend_worker.cpp"
        "inflect_text_normalizer.cpp"
        "${INFLECT_ASSETS_ASM}"
    INCLUDE_DIRS "."
    REQUIRES espeak_ng esp_driver_uart esp_hw_support esp_psram esp_timer espressif__esp-dl fatfs spi_flash vfs
)

option(INFLECT_EMIT_PCM_HEX "Emit the full PCM payload over UART" ON)
option(INFLECT_PROFILE_DECODER "Run the ESP-DL module profiler after timing" OFF)
option(INFLECT_BENCH_INTERLEAVE "Enable the fused polyphase interleave microbenchmark command" OFF)
option(INFLECT_BENCH_PRIMITIVES "Enable decoder architecture primitive benchmarks" OFF)
option(INFLECT_BENCH_STUDENTS "Enable complete student decoder architecture benchmarks" OFF)
option(INFLECT_PERSIST_MODELS "Keep all loaded models resident between runs" OFF)
option(INFLECT_VALIDATE_CANARIES "Validate packaged acoustic canary tensors" ON)
option(INFLECT_COMMAND_LOOP "Wait for RUN/WAV commands on the console UART" OFF)
option(INFLECT_PRELOAD_FULL_DECODER "Load the Full143 decoder before acoustic and flow" OFF)
option(
    INFLECT_SHORT_VALUE64
    "Use T64/T96 decoders and segment requests that exceed 83 latent frames"
    OFF)
option(
    INFLECT_DECODER_TILE96
    "Use the context-cropped first/tail decoder package for long requests"
    OFF)
option(
    INFLECT_DECODER_BRIDGE96_69
    "Publish four reduced-context T96 frames before a suffix-aligned T69 tail"
    OFF)
if(INFLECT_DECODER_TILE96 AND INFLECT_DECODER_BRIDGE96_69)
    message(FATAL_ERROR
        "INFLECT_DECODER_TILE96 and INFLECT_DECODER_BRIDGE96_69 are exclusive")
endif()
if(INFLECT_SHORT_VALUE64 AND (INFLECT_DECODER_TILE96 OR INFLECT_DECODER_BRIDGE96_69))
    message(FATAL_ERROR
        "INFLECT_SHORT_VALUE64 is exclusive with tiled decoder modes")
endif()
if(INFLECT_SHORT_VALUE64 AND INFLECT_PRELOAD_FULL_DECODER)
    message(FATAL_ERROR
        "INFLECT_SHORT_VALUE64 does not package the Full143 decoder")
endif()
if(INFLECT_SHORT_VALUE64 AND NOT INFLECT_FLOW96)
    message(FATAL_ERROR
        "INFLECT_SHORT_VALUE64 requires INFLECT_FLOW96 for the short route")
endif()
set(INFLECT_REPEAT_RUNS 1 CACHE STRING "Full pipeline iterations per boot")
set(INFLECT_DECODER_INTERNAL_BYTES 0 CACHE STRING "Internal SRAM budget for decoder tensors")
set(
    INFLECT_DECODER_FULL_PSRAM_PAD_BYTES
    0
    CACHE STRING
    "Persistent PSRAM placement pad allocated before the Full143 decoder")
if(INFLECT_BENCH_INTERLEAVE)
    target_sources(${COMPONENT_LIB} PRIVATE "inflect_polyphase_interleave.cpp")
endif()
if(INFLECT_BENCH_PRIMITIVES)
    target_sources(${COMPONENT_LIB} PRIVATE "inflect_decoder_primitives.cpp")
endif()
if(INFLECT_BENCH_STUDENTS)
    target_sources(${COMPONENT_LIB} PRIVATE "inflect_student_benchmarks.cpp")
endif()
target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE
        INFLECT_EMIT_PCM_HEX=$<BOOL:${INFLECT_EMIT_PCM_HEX}>
        INFLECT_PROFILE_DECODER=$<BOOL:${INFLECT_PROFILE_DECODER}>
        INFLECT_BENCH_INTERLEAVE=$<BOOL:${INFLECT_BENCH_INTERLEAVE}>
        INFLECT_BENCH_PRIMITIVES=$<BOOL:${INFLECT_BENCH_PRIMITIVES}>
        INFLECT_BENCH_STUDENTS=$<BOOL:${INFLECT_BENCH_STUDENTS}>
        INFLECT_PERSIST_MODELS=$<BOOL:${INFLECT_PERSIST_MODELS}>
        INFLECT_VALIDATE_CANARIES=$<BOOL:${INFLECT_VALIDATE_CANARIES}>
        INFLECT_COMMAND_LOOP=$<BOOL:${INFLECT_COMMAND_LOOP}>
        INFLECT_PRELOAD_FULL_DECODER=$<BOOL:${INFLECT_PRELOAD_FULL_DECODER}>
        INFLECT_SHORT_VALUE64=$<BOOL:${INFLECT_SHORT_VALUE64}>
        INFLECT_DECODER_TILE96=$<BOOL:${INFLECT_DECODER_TILE96}>
        INFLECT_DECODER_BRIDGE96_69=$<BOOL:${INFLECT_DECODER_BRIDGE96_69}>
        INFLECT_FLOW96=$<BOOL:${INFLECT_FLOW96}>
        INFLECT_REPEAT_RUNS=${INFLECT_REPEAT_RUNS}
        INFLECT_DECODER_INTERNAL_BYTES=${INFLECT_DECODER_INTERNAL_BYTES}
        INFLECT_DECODER_FULL_PSRAM_PAD_BYTES=${INFLECT_DECODER_FULL_PSRAM_PAD_BYTES}
)

if(INFLECT_SHORT_VALUE64)
    set(tts_models_source
        "${CMAKE_CURRENT_LIST_DIR}/inflect_tts_models_short64.espdl")
elseif(INFLECT_DECODER_BRIDGE96_69)
    set(tts_models_source
        "${CMAKE_CURRENT_LIST_DIR}/inflect_tts_models_tile69.espdl")
elseif(INFLECT_DECODER_TILE96)
    set(tts_models_source
        "${CMAKE_CURRENT_LIST_DIR}/inflect_tts_models_tile100_65.espdl")
else()
    set(tts_models_source
        "${CMAKE_CURRENT_LIST_DIR}/inflect_tts_models.espdl")
endif()
set(tts_models "${CMAKE_BINARY_DIR}/inflect_tts_models.espdl")
configure_file(
    "${tts_models_source}"
    "${tts_models}"
    COPYONLY
)
esptool_py_flash_to_partition(flash "model" "${tts_models}")

set(flow96_models "${CMAKE_BINARY_DIR}/inflect_flow96_models.espdl")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/inflect_flow96_models.espdl"
    "${flow96_models}"
    COPYONLY
)
esptool_py_flash_to_partition(flash "flow_short" "${flow96_models}")
