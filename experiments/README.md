# Experiment & evaluation setup

The whitepaper outlines three pillars for validating Hybrid Neural Quantum
Architecture concepts: ambiguity retention, contextual collapse quality, and
resource efficiency compared with classical baselines.  This directory provides
stubs and documentation so collaborators can reproduce those studies once data
and infrastructure become available.

## Directory structure

```
experiments/
├── README.md            # This file
├── data/                # Placeholder for preprocessed benchmark artefacts
├── notebooks/           # Jupyter notebooks for exploratory analysis
└── pipelines/           # Reusable scripts for automated evaluation runs
```

Create the sub-folders as needed; they are intentionally empty in the public
repository to avoid distributing potentially licensed datasets.

## Proposed evaluation tracks

1. **Ambiguity retention benchmark**
   - *Goal:* Measure how well the superposition memory maintains alternate
     hypotheses compared with a classical softmax attention model.
   - *Inputs:* Synthetic sequences containing forced ambiguity events.
   - *Metrics:* Entropy of the stored hypothesis distribution, recovery rate
     after delayed disambiguation cues.

2. **Contextual collapse quality**
   - *Goal:* Quantify whether contextual features steer the collapse toward the
     correct interpretation without over-fitting.
   - *Inputs:* Scenario datasets with labelled context cues (e.g. sensor logs,
     operator annotations).
   - *Metrics:* Accuracy, calibration (Brier score), qualitative inspection via
     notebooks.

3. **Resource footprint comparison**
   - *Goal:* Compare runtime and memory footprint of HNQA-inspired modules
     against classical baselines on identical hardware.
   - *Inputs:* Same datasets as above plus profiling hooks.
   - *Metrics:* Wall-clock time, peak memory, throughput measured with standard
     Python tooling (`time.perf_counter`, `tracemalloc`).

## Getting started

1. Generate or obtain the benchmark data and place it under `experiments/data/`.
2. Use the reference implementations in `models/` as building blocks for the
   first experimental pipelines.
3. Document assumptions and results in notebooks saved under
   `experiments/notebooks/`.  Include links from the notebook markdown cells
   back to the relevant whitepaper sections for traceability.
4. When pipelines stabilise, promote them into version-controlled Python scripts
   under `experiments/pipelines/` and annotate them with the parameters used to
   reproduce published figures.

## Data governance

- Keep raw proprietary data outside the repository; only derived statistics or
  synthetic samples should be committed.
- Note the DOI of the whitepaper in reports to maintain citation continuity.
- Record software versions (Python, library dependencies) inside notebooks or
  pipeline metadata for reproducibility.
