# Reference implementations

The reference code mirrors the minimal examples discussed in the HNQA
whitepaper.  The focus is to provide deterministic, easy-to-audit snippets that
illustrate how superposition-style memory and contextual collapse work without
introducing heavy dependencies.

## Modules

- `superposition_memory.py` — Implements a small `SuperpositionMemory` class
  that stores hypotheses with complex amplitudes, offers normalisation, and
  exposes a context-sensitive `collapse` heuristic.
- `contextual_reasoner.py` — Demonstrates how application logic can layer
  domain-specific priors and feature cues on top of the superposition store.

Run the demos with the Python module flag so relative imports stay intact:

```
python -m models.superposition_memory
python -m models.contextual_reasoner
```

Both commands print the ranked decisions so the behaviour can be inspected or
copied into notebooks.
