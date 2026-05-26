# Gap Analysis

This document summarizes the design gaps addressed in the current Causpera
codebase and the boundaries that remain open for future research.

Causpera is positioned as an experimental framework for combining
causal-sensitive sparse learning, adaptive memory consolidation, and
uncertainty-gated computation. The current version focuses on making these ideas
testable in a clean PyTorch package, while keeping the causal claims explicit
and conservative.

## Design Goals

The current implementation was organized around four goals:

1. provide a clean and reusable PyTorch package;
2. expose a practical model API for experimentation;
3. make causal-sensitive masking inspectable without overstating causal proof;
4. support adaptive inference and parameter-importance tracking in one model.

These goals shaped the package structure, model interface, benchmark design,
test coverage, and documentation.

## Useful Capabilities Retained

The current codebase keeps several capabilities that are important for a public
research prototype.

### Model usability

Causpera provides a central model interface with configuration-driven
initialization, model-level loss computation, inference helpers, adaptive
routing statistics, and consolidation utilities. This makes the project easier
to use in experiments without requiring users to manually connect each internal
component.

### Modular architecture

The implementation keeps a clear separation between attention, configuration,
data generation, loss functions, metrics, model logic, and plasticity. This
separation makes the code easier to inspect, test, and extend.

### Conservative causal positioning

The causal component is implemented as a temporal causal-mask proxy. It uses
temporal order, directional compatibility, and perturbation sensitivity as
trainable signals. The mask is treated as an experimental approximation, not as
formal causal identification.

### Temporal toy benchmark

The package includes a toy temporal dataset with known lagged causal features
and correlated distractors. This benchmark is useful for smoke-testing whether
the model can focus on relevant temporal signals without claiming real-world
causal discovery.

### Adaptive inference path

The model supports uncertainty-gated inference, where lower-uncertainty inputs
can use a lighter computation path and higher-uncertainty inputs can use deeper
processing. This makes compute allocation part of the model behavior rather
than a fixed assumption.

### Parameter-importance tracking

The plasticity module tracks parameter importance and penalizes destructive
drift. This supports early experimentation with continual-learning behavior,
although it should not be interpreted as a complete solution to catastrophic
forgetting.

## Main Gaps Addressed

The current version addresses several engineering and research-positioning
gaps.

### Packaging and repository structure

The project now uses a standard `src/` package layout with `pyproject.toml`.
This makes the package installable in editable mode and suitable for testing,
linting, and future distribution.

### Public-facing naming

The final package uses `causpera` as the public name. The name avoids making
the acronym `CSPN` the primary identity, because similar acronyms already exist
in other machine learning contexts. Compatibility aliases can still be provided
where useful, but the public-facing brand remains Causpera.

### Documentation clarity

The documentation avoids encoding artifacts, overly broad claims, and internal
implementation history that would not be useful to repository users. It focuses
on what the package does, what assumptions it makes, and what limitations remain.

### Causal-claim control

The project avoids implying that attention weights or learned masks prove
causality. The causal mask is described as a proxy for experimentation and
diagnostics, not as a substitute for formal causal identification or
intervention-based validation.

### Adaptive inference behavior

Adaptive inference is implemented at the model-computation level, not only at
the classifier-head level. Lower-uncertainty inputs can take a lighter path,
while more uncertain inputs can trigger deeper processing.

### Test organization

Tests are separated by concern, including attention, data, loss, model behavior,
routing, and plasticity. This makes failures easier to diagnose and keeps the
test suite maintainable as the framework evolves.

### Repository hygiene

Generated files, caches, and environment artifacts are excluded through
`.gitignore`. This keeps the repository cleaner and more suitable for public
release.

## Remaining Gaps

Several important gaps remain and should be treated as future work.

### Formal causal identification

Causpera does not establish Pearl-style causal identification from observational
data alone. The current mask is a differentiable approximation and should be
validated against stronger causal-discovery methods and intervention-based
datasets.

### Real-world benchmark coverage

The current toy benchmark is useful for controlled testing, but it does not
capture the complexity of real-world causal structure. Additional benchmarks
are needed for time-series forecasting, healthcare risk modeling, fraud
detection, operational decisioning, and edge inference.

### Efficiency measurement

The current routing statistics estimate compute use based on model path
selection and layer usage. They are not a profiler-level FLOP, latency, memory,
or energy measurement. Proper efficiency claims require hardware-aware
benchmarking.

### Continual-learning validation

The consolidation mechanism penalizes movement of important parameters, but it
has not yet been validated across a broad continual-learning benchmark suite.
Further testing is needed against EWC-style methods, Synaptic Intelligence,
replay-based approaches, and adapter-based methods.

### Calibration and uncertainty

The current uncertainty signal is sufficient for routing experiments, but it
should be evaluated with stronger calibration metrics before being used in
high-stakes settings. Future work may include expected calibration error,
selective prediction, abstention behavior, and uncertainty decomposition.

### Interpretability

Causal masks and routing statistics are inspectable, but inspectability is not
the same as validated interpretability. Additional tooling is needed to compare
learned masks with known causal structure, intervention results, or expert
annotations.

## Intentional Non-Goals

The current version intentionally avoids several claims and features.

- It does not claim to solve causal discovery in deep learning.
- It does not claim that attention weights prove causality.
- It does not claim broad real-world robustness based on the toy benchmark.
- It does not claim measured energy reduction without hardware profiling.
- It does not replace established causal inference methods.
- It does not eliminate catastrophic forgetting in all continual-learning
  settings.
- It does not target production deployment for high-stakes decision systems in
  its current form.

## Recommended Evaluation Path

Future evaluation should proceed in stages.

1. Compare Causpera against fixed-depth transformer baselines on the toy temporal
   benchmark.
2. Run ablation studies for causal-sensitive masking, consolidation, and
   uncertainty-gated routing.
3. Measure routing behavior against accuracy, calibration, and compute estimates.
4. Evaluate continual-learning behavior across sequential tasks.
5. Test on realistic temporal datasets with known or partially known causal
   structure.
6. Add profiler-level latency, memory, and FLOP measurements.
7. Validate interpretability claims against ground-truth edges or expert review.

## Summary

Causpera closes several practical gaps needed for a public research prototype:
standard packaging, a cleaner API, explicit causal assumptions, adaptive
inference, parameter-importance tracking, test coverage, and research
limitations.

The remaining work is primarily empirical. The framework is ready for controlled
experimentation, but broader claims about causality, robustness, interpretability,
and energy efficiency should depend on future benchmarks and ablation studies.