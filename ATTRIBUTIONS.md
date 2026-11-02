# Attributions

No third-party code is bundled in this repository (checked with fosslic, 2026-11-01).

- `experiments/kvmove/kvmove_test.py` builds its test document at run time from the Python standard library sources
  installed on the machine (`/usr/lib/python3*/*.py`, PSF License). Those sources and the tokenized documents are not
  included here.
- The experiments drive [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and
  [llama-swap](https://github.com/mostlygeek/llama-swap) (MIT) over HTTP; neither is vendored.
