# Complete Raw-Text Neural TTS on ESP32-P4

This repository contains the paper, firmware, runtime patch, model packages,
host capture tool, and listening evidence for a complete English
text-to-waveform deployment of
[Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) on an
ESP32-P4.

The ESP32-P4 receives raw UTF-8 text and performs all speech work locally:

- normalization and eSpeak-NG phonemization;
- Inflect tokenization and VITS blank insertion;
- acoustic inference, duration/path construction, and latent sampling;
- inverse normalizing flow;
- neural waveform decoding and PCM16 conversion.

USB provides power, sends text, and captures diagnostic PCM. It does not
provide phonemes, tokens, latent tensors, or waveform samples.

## Result

The retained no-retraining route uses the released Inflect weights. On the
tested ESP32-P4 revision 1.3 at 360 MHz, the `Ready.` request produced 11,264
samples, or 0.469333 seconds of 24 kHz mono speech. The learned stages took
1.656448 seconds:

| Stage | Time |
|---|---:|
| Acoustic model | 86.827 ms |
| Flow48 | 234.609 ms |
| T64 decoder | 1,335.012 ms |
| Complete learned path | 1,656.448 ms |

The decoder returns a complete buffer and is slower than real time. This is a
local short-phrase/status-announcement runtime, not a conversational streaming
claim.

The exact captured result is in `evidence/ready-flow48/`:

- WAV SHA-256: `add6b5be3e21278a46837c5a0269cee9392bbf6dc6081327082797963fd381b8`
- JSON SHA-256: `17e17cb633b26863ae7c990cae0e0529f4fcec9c5917ffdca7eaa8b21500ec31`
- PCM FNV-1a: `51971333`

## Repository Layout

- `embedded/inflect_p4_bench/`: final ESP-IDF project and on-device frontend.
- `patches/`: Inflect-specific changes over official ESP-DL 3.3.9.
- `tools/prepare_esp_dl.sh`: verified ESP-DL download and patch application.
- `tools/build_inflect_p4_release.sh`: board-independent release build.
- `tools/build_flash_inflect_p4_service.sh`: build and flash helper.
- `tools/run_inflect_p4_service.py`: raw-text request and verified WAV capture.
- `evidence/`: retained board output and machine-readable telemetry.
- `paper/`: immutable publication snapshot.

Experimental G5-G18 decoders, rejected pruning routes, training corpora, and
diagnostic builds are intentionally absent.

## Requirements

- ESP32-P4 revision 1.x with 32 MiB PSRAM and 16 MiB flash.
- ESP-IDF 6.0.2 at commit
  `7101770dc6db2667b3c477cc31365dd1acd6db4e`.
- `curl`, `unzip`, `patch`, and either `shasum` or `sha256sum` on the build
  host.
- Python 3 with `pyserial` for WAV capture.

The checked-in defaults target ESP32-P4 revision 1.x. They are not a firmware
image for revision 3.x hardware.

## Build

Install the pinned ESP-IDF environment once:

```sh
git clone --branch v6.0.2 --recursive \
  https://github.com/espressif/esp-idf.git "$HOME/esp/esp-idf-v6.0.2"
cd "$HOME/esp/esp-idf-v6.0.2"
./install.sh esp32p4
```

Then build from this repository:

```sh
./tools/build_inflect_p4_release.sh
```

The preparation script downloads the official ESP-DL 3.3.9 registry archive,
verifies archive SHA-256
`aad5bcc10c34ee638003aa50e41d19f66106cc067c72393d542c8f8ca9155910`,
applies the checked-in 11-file runtime patch, and verifies every patched file.
No full modified ESP-DL vendor tree is hidden in the repository.

## Flash And Capture

With a compatible board attached:

```sh
PORT=/dev/cu.usbserial-3110 ./tools/build_flash_inflect_p4_service.sh

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tools/run_inflect_p4_service.py \
  --port /dev/cu.usbserial-3110 \
  --text "Ready." \
  --output captures/ready.wav
```

The host tool checks the declared byte count and board FNV-1a before writing a
24 kHz mono PCM16 WAV. Short requests use the T64/Flow48 route. Requests that
need more latent frames use T96/Flow96, subject to the static limits documented
in the paper.

## Provenance

The runtime source and packages were exported from research commit
`f14bf17d91eb5835dee93e8dc6f5b80ef8b57a09`. The original publication-only
snapshot remains commit `336e7670a4e6a540d8841df35f8d9b3da3fae23f`.
See `PROVENANCE.md` and `artifact-manifest.json` for the complete boundary.

## Paper

`paper/esp32_p4_inflect_tts.pdf` is the rendered report, *Complete Raw-Text
Neural Speech Synthesis on an ESP32-P4*. Its source and arXiv archive are in
the same directory.

## Licensing

The combined firmware source is released under GPL-3.0-or-later because it
embeds eSpeak-NG. Inflect-derived model packages retain the upstream
Apache-2.0 terms. ESP-DL and the patch over it retain MIT terms. The manuscript
has a separate publication license boundary. See `LICENSES.md`.
