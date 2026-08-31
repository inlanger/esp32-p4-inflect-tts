# eSpeak-NG phonemizer component

This component vendors the eSpeak-NG 1.52.0 C sources from upstream commit
`4870adfa25b1a32b4361592f1be8a40337c58d6c`. Optional audio, async, MBROLA,
Klatt, Sonic, and SpeechPlayer backends are disabled. The Inflect firmware uses
only the synchronous English text-to-IPA API.

The read-only runtime data is generated from the exact `espeakng-loader`
1.52.0 data used by the pinned host reference. It contains only `en_dict`, the
shared phoneme tables, intonation data, and the `en`/`en-US` language files.
Upstream GPL-3.0 and Unicode data licenses are retained as `COPYING` and
`COPYING.UCD`.

