# Reproducibility Boundary

## Publication

- Public repository: `https://github.com/inlanger/esp32-p4-inflect-tts`
- Original paper snapshot: `336e7670a4e6a540d8841df35f8d9b3da3fae23f`
- Exact runtime/artifact release: `f5701f9f92e92522086233d6c10879c07740b10f`
- Supplemental decoder evidence: `d73897194ae034cf8a13d87add01ca15f99799f8`
- Hugging Face model revision: `282a7eaf953e8a9e972b86d844d1d2823a48df21`
- Runtime source boundary: `f14bf17d91eb5835dee93e8dc6f5b80ef8b57a09`
- Source boundary message: `feat: deploy exact Flow48 short-value route`

The runtime boundary precedes the G5/G6 decoder-training work. No trained,
distilled, flow-absorbing, causal, PTQ, or experimental student decoder is part
of this release.

## Platform

- ESP32-P4 revision 1.3
- Two high-performance RISC-V cores at 360 MHz
- 32 MiB in-package PSRAM at 200 MHz
- 16 MiB flash
- ESP-IDF 6.0.2, commit `7101770dc6db2667b3c477cc31365dd1acd6db4e`
- ESP-DL 3.3.9 plus `patches/esp-dl-3.3.9-inflect.patch`
- eSpeak-NG 1.52.0, commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`

## Included Routes

The main package contains the T49 acoustic graph, T64 decoder, T96 decoder,
and exact CSE Flow48 graph. Flow96 is stored in a second flash partition for
requests that do not fit Flow48. The released model weights are not retrained.

Static routing and complete-buffer output are deliberate limits. The release
does not claim native decoder streaming, faster-than-real-time synthesis,
multilingual speech, voice cloning, or quality equivalence outside the retained
listening evidence.

## Evidence

`evidence/ready-flow48/ready-flow48.wav` and its JSON sidecar are copied byte
for byte from the accepted connected-board gate. The JSON preserves its
original local output path so its publication SHA remains identical to the
one printed in the paper.

`evidence/decoder-surface/decoder-surface.json` is the public supplemental
record for the decoder structure, nominal compute, activation statistics,
parameter groups, and receptive-field calculations. Only two workstation
paths were removed from the retained research artifact. Its public SHA-256 is
`ad3fe556db6ea0ca54115ad7413ff1d7a88d7df18ef0f1c81e26419180c254bb`;
the original pre-sanitization SHA-256 is retained inside the file.

The original application binary SHA-256 reported by the paper is
`50462fa0add9f59b68ca9b8e7adc8525e4eb4427d839405694652a101b9d1634`.
Build paths and ESP-IDF generated metadata can affect a rebuilt application
binary; model package and evidence hashes are the stable cross-build checks.
