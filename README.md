# Complete Raw-Text Neural TTS on ESP32-P4

This repository is the public publication snapshot for the technical report
"Complete Raw-Text Neural Speech Synthesis on an ESP32-P4" by Volodymyr
Sokolovskyi.

The reported system accepts raw UTF-8 text and executes the complete English
text-to-waveform path on an ESP32-P4 revision 1.3:

- text normalization and eSpeak-NG phonemization;
- model-specific tokenization and VITS blank insertion;
- acoustic inference, duration and monotonic-path construction;
- latent sampling and inverse normalizing flow;
- neural waveform decoding and PCM16 conversion.

The development computer supplies USB power and raw text. It is also used for
diagnostic PCM capture. It does not compute phonemes, tokens, latent tensors,
or waveform samples.

## Publication files

- `paper/esp32_p4_inflect_tts.pdf`: rendered technical report.
- `paper/esp32_p4_inflect_tts.tex`: complete LaTeX source.
- `paper/references.bib`: BibTeX references.
- `paper/esp32_p4_inflect_arxiv_source.tar.gz`: arXiv upload archive.
- `paper/arxiv_metadata.txt`: proposed arXiv metadata.
- `paper/build_arxiv_source.sh`: deterministic source-archive builder.

## Implementation status

The implementation and experiment history are still being consolidated while
decoder-training work continues. This initial public snapshot does not yet
contain firmware, ESP-DL modifications, model packages, or the listening
corpus. Those artifacts will be published as a versioned release after their
reproducibility and third-party licensing boundaries are frozen.

The report already records the exact hardware revision, dependency versions,
model/checkpoint hashes, package hashes, measured memory, board timings, and
accepted or rejected experiment results. Later artifact releases will preserve
the publication snapshot rather than rewrite its history.

## Reported result

The final exact-weight short-value route uses Flow48 and T64 for a 44-frame
request. It executes all learned stages in 1.656448 seconds and produces
0.469333 seconds of 24 kHz mono speech. The current decoder is slower than real
time and returns complete buffers; the report makes no native-streaming claim.

## Author

Volodymyr Sokolovskyi, Go Wombat
vladimir.s@gowombat.team

## Licensing

See `LICENSES.md`. No repository-wide software license is implied by this
publication-only snapshot.
