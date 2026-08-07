#!/bin/sh
set -eu

paper_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:-/tmp/esp32-p4-inflect-arxiv.tar.gz}
staging=$(mktemp -d "${TMPDIR:-/tmp}/esp32-p4-inflect-arxiv.XXXXXX")
trap 'rm -rf "$staging"' EXIT HUP INT TERM

cp "$paper_dir/esp32_p4_inflect_tts.tex" "$staging/main.tex"
cp "$paper_dir/references.bib" "$staging/references.bib"

tar -C "$staging" -czf "$output" main.tex references.bib
printf '%s\n' "$output"
