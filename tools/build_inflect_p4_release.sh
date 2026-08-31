#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/embedded/inflect_p4_bench"
IDF_DIR="${IDF_DIR:-$HOME/esp/esp-idf-v6.0.2}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-release}"
SDKCONFIG="${SDKCONFIG:-$BUILD_DIR/sdkconfig}"

"$ROOT_DIR/tools/prepare_esp_dl.sh"
source "$IDF_DIR/export.sh" >/dev/null 2>&1

sha256_print() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1"
    else
        sha256sum "$1"
    fi
}

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G "Unix Makefiles" \
  -DESP_PLATFORM=1 \
  -DIDF_TARGET=esp32p4 \
  -DSDKCONFIG="$SDKCONFIG" \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.cache512.defaults;sdkconfig.service.defaults" \
  -DINFLECT_COMMAND_LOOP=ON \
  -DINFLECT_PERSIST_MODELS=ON \
  -DINFLECT_VALIDATE_CANARIES=OFF \
  -DINFLECT_PRELOAD_FULL_DECODER=OFF \
  -DINFLECT_DECODER_TILE96=OFF \
  -DINFLECT_DECODER_BRIDGE96_69=OFF \
  -DINFLECT_EMIT_PCM_HEX=OFF \
  -DINFLECT_PROFILE_DECODER=OFF \
  -DINFLECT_BENCH_INTERLEAVE=OFF \
  -DINFLECT_BENCH_PRIMITIVES=OFF \
  -DINFLECT_BENCH_STUDENTS=OFF \
  -DINFLECT_RUNTIME_PRUNE_K3=OFF \
  -DINFLECT_RUNTIME_COPY_K3_MERGES=OFF \
  -DINFLECT_RUNTIME_DIV2_REQUANT=OFF \
  -DINFLECT_RUNTIME_PRUNE_K7_TAIL=OFF \
  -DINFLECT_RUNTIME_PRUNE_K7_SECOND=OFF \
  -DINFLECT_RUNTIME_BYPASS_K7_MERGES=OFF \
  -DINFLECT_FLOW96=ON \
  -DINFLECT_SHORT_VALUE64=ON \
  -DINFLECT_DECODER_INTERNAL_BYTES=0 \
  -DINFLECT_DECODER_FULL_PSRAM_PAD_BYTES=0 \
  -DINFLECT_ESPDL_TEXT_PAD_BYTES=0

cmake --build "$BUILD_DIR" -j

echo "Build outputs:"
for file in \
  "$BUILD_DIR/inflect_p4_bench.bin" \
  "$BUILD_DIR/inflect_tts_models.espdl" \
  "$BUILD_DIR/inflect_flow96_models.espdl" \
  "$BUILD_DIR/g2p.bin"; do
    printf '%s bytes  %s\n' "$(wc -c < "$file" | tr -d ' ')" "$file"
    sha256_print "$file"
done
