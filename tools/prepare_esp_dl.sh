#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/third_party/esp-dl"
PATCH="$ROOT_DIR/patches/esp-dl-3.3.9-inflect.patch"
HASHES="$ROOT_DIR/patches/esp-dl-3.3.9-patched-files.sha256"
ARCHIVE_URL="https://components-file.espressif.com/components/espressif/esp-dl/3.3.9/espressif__esp-dl-v3.3.9.zip"
ARCHIVE_SHA256="aad5bcc10c34ee638003aa50e41d19f66106cc067c72393d542c8f8ca9155910"

sha256_check() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "${1:--}"
    else
        sha256sum -c "${1:--}"
    fi
}

verify_tree() {
    (cd "$DEST" && sha256_check "$HASHES")
}

if [[ -d "$DEST" ]]; then
    if verify_tree >/dev/null 2>&1; then
        echo "ESP-DL 3.3.9 Inflect patch is ready: $DEST"
        exit 0
    fi
    echo "Existing $DEST does not match the release patch." >&2
    echo "Remove that directory and run this command again." >&2
    exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/inflect-esp-dl.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
ARCHIVE="$TMP_DIR/esp-dl-3.3.9.zip"

curl --fail --location --silent --show-error "$ARCHIVE_URL" --output "$ARCHIVE"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256_check
unzip -q "$ARCHIVE" -d "$TMP_DIR/unpacked"

mkdir -p "$(dirname "$DEST")"
mv "$TMP_DIR/unpacked" "$DEST"
patch -d "$DEST" -p1 < "$PATCH"
verify_tree
echo "Prepared patched ESP-DL 3.3.9 in $DEST"
