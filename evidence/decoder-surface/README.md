# Decoder Surface Evidence

`decoder-surface.json` is the public supplemental record behind the paper's
decoder-shape, parameter, nominal-MAC, activation-statistics, and receptive-
field calculations. It is evidence for the analysis; it is not required by
the firmware or model runtime.

The public file was derived mechanically from the retained research artifact:

- Original SHA-256: `3572eb3b963bdc98c2460957457665f5f538bfb6cd715a5017b67651a12a9b17`
- Public SHA-256: `ad3fe556db6ea0ca54115ad7413ff1d7a88d7df18ef0f1c81e26419180c254bb`

Only two workstation-local path fields were removed. They were replaced by
the pinned upstream model repository/revision and a stable input artifact ID.
All numeric evidence and source hashes are unchanged. The transformation is
described in the JSON's `publication` object.
