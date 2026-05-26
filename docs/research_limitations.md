# Research Limitations

Causpera is an experimental research framework. It is designed to make several
ideas testable in code: causal-sensitive sparse learning, adaptive memory
consolidation, and uncertainty-gated computation.

The current implementation should be understood as a prototype for controlled
experimentation, not as a completed method for causal discovery, continual
learning, or production-grade adaptive inference.

## Scope of the Current Implementation

The current version focuses on sequence classification with temporal structure.
It provides:

- a causal-sensitive attention mask based on temporal and perturbation signals;
- parameter-importance tracking for consolidation experiments;
- uncertainty-gated routing between lighter and deeper computation paths;
- a temporal toy benchmark with known lagged causal features;
- diagnostics for masks, routing behavior, uncertainty, and estimated compute use.

These components are intended to support experimentation and ablation studies.
They should not be interpreted as evidence of broad real-world performance until
the model has been evaluated on stronger benchmarks.

## Causal Interpretation

The causal mask in Causpera is a learned approximation. It is designed to act as
an inductive bias that encourages the model to focus on temporally plausible and
perturbation-sensitive relationships.

It should not be interpreted as formal proof of causality.

A high mask value means that the model has learned to treat a relationship as
useful under the implemented proxy. It does not prove that the source variable
causes the target variable in the underlying data-generating process.

Real causal claims require stronger evidence, such as:

- explicit causal assumptions;
- intervention data;
- randomized or quasi-experimental designs;
- domain-specific validation;
- comparison with established causal discovery methods;
- robustness checks against confounding and selection bias.

## Observational Data

Causpera can be trained on observational temporal data, but observational data
has inherent limitations for causal inference.

Temporal order can rule out some impossible relationships. For example, a future
token should not cause a past token. However, temporal order alone does not solve
the broader causal identification problem.

The model may still be affected by:

- hidden confounders;
- selection bias;
- measurement error;
- data leakage;
- non-stationary data;
- correlated but non-causal features;
- spurious relationships that hold only in the training distribution.

For this reason, Causpera should be evaluated carefully when used on real-world
datasets where causal interpretation matters.

## Perturbation-Based Proxy

The current implementation uses perturbation sensitivity as part of the
causal-mask signal. This means the model observes how internal representations
change under small simulated perturbations.

This is useful for experimentation, but it is not equivalent to intervening on
the real system.

A hidden-state perturbation does not necessarily correspond to a real-world
intervention on the data-generating process. For example, perturbing a latent
representation of a medical feature is not the same as changing a patient’s
clinical condition.

The perturbation mechanism should therefore be treated as a diagnostic and
training signal, not as a substitute for real intervention data.

## Attention and Interpretability

Causpera exposes attention maps and causal-mask values for inspection. These
outputs can be useful for debugging, diagnostics, and hypothesis generation.

However, inspectability is not the same as validated interpretability.

The learned mask may show which temporal relationships the model relies on, but
additional validation is needed before claiming that the mask explains the true
reason behind a prediction.

Recommended validation steps include:

- comparing learned masks with known synthetic causal edges;
- running ablation studies by removing high-mask and low-mask features;
- testing mask stability across random seeds;
- comparing results against baseline transformers;
- validating explanations with domain experts where applicable.

## Continual Learning and Consolidation

Causpera includes parameter-importance tracking and consolidation penalties to
reduce destructive drift during learning.

This mechanism can support continual-learning experiments, but it does not
eliminate catastrophic forgetting in all settings.

The current consolidation approach should be evaluated against established
continual-learning baselines and under different sequential task conditions.

Important open questions include:

- whether the importance scores remain stable across tasks;
- whether consolidation protects useful knowledge without blocking adaptation;
- how the method behaves under task distribution shift;
- how it compares with replay-based, regularization-based, and adapter-based
  continual-learning methods.

## Uncertainty-Gated Computation

Causpera routes inputs through lighter or deeper computation paths based on
uncertainty. This is intended to explore whether computation can be allocated
dynamically rather than applied uniformly to every input.

The current uncertainty signal is suitable for routing experiments, but it should
not be treated as a fully calibrated uncertainty estimate without further
evaluation.

Before using uncertainty-gated routing in high-stakes contexts, the model should
be tested for:

- calibration error;
- selective prediction behavior;
- failure cases under distribution shift;
- confidence on ambiguous or noisy inputs;
- routing stability across datasets;
- relationship between uncertainty, accuracy, and compute use.

## Compute and Energy Estimates

Causpera reports estimated compute savings based on routing behavior and the
fraction of encoder computation skipped.

These estimates are useful for comparing model behavior during experiments, but
they are not hardware-level energy measurements.

The current metrics should not be reported as actual latency, FLOP reduction,
GPU utilization, memory savings, or energy consumption unless validated with
proper profiling tools.

Hardware-specific evaluation should include:

- wall-clock latency;
- FLOP estimation;
- memory usage;
- GPU or CPU utilization;
- batch-size sensitivity;
- throughput under realistic inference settings;
- profiler-based comparison against fixed-depth baselines.

## Toy Benchmark Limitations

The included toy benchmark is intentionally simple. It creates temporal
sequences where a small number of lagged features determine the label, while
other features act as correlated distractors.

This benchmark is useful for smoke testing because the relevant temporal
structure is known. However, it does not represent the complexity of real-world
causal systems.

The toy benchmark does not capture:

- complex feedback loops;
- multiple interacting causal pathways;
- delayed effects across long horizons;
- distribution shift;
- missing data;
- noisy labels at production scale;
- domain-specific measurement constraints.

Performance on the toy benchmark should therefore be treated as a basic
sanity check, not as evidence of general causal reasoning capability.

## Production Readiness

Causpera is not production-ready for high-stakes decision systems in its current
form.

Before any production use, additional work would be required in areas such as:

- robustness testing;
- calibration;
- monitoring;
- security review;
- data governance;
- model risk management;
- reproducibility;
- documentation of assumptions;
- auditability of routing and mask behavior;
- domain-specific validation.

For sensitive domains such as healthcare, finance, safety, or public-sector
decisioning, Causpera should be treated only as a research component until it has
undergone rigorous validation.

## Recommended Evaluation Path

Future evaluation should proceed through staged validation.

1. Compare Causpera against fixed-depth transformer baselines on the toy temporal
   benchmark.
2. Run ablation studies for causal-sensitive masking, consolidation, and
   uncertainty-gated routing.
3. Test whether learned masks recover known lagged causal features in synthetic
   settings.
4. Evaluate routing behavior against accuracy, uncertainty, and estimated compute
   savings.
5. Compare consolidation behavior against continual-learning baselines.
6. Test on more realistic temporal datasets with known or partially known causal
   structure.
7. Add profiler-level measurements for latency, FLOPs, memory use, and throughput.
8. Validate interpretability claims through feature ablation, mask stability, and
   domain review.

## Summary

Causpera is useful as a controlled research prototype for testing how
causal-sensitive sparse learning, memory consolidation, and adaptive computation
can work together in one architecture.

Its current outputs should be interpreted as experimental signals, not definitive
causal explanations or production guarantees. Broader claims about causality,
robustness, interpretability, continual learning, and compute efficiency should
depend on future benchmarks, ablation studies, and hardware-level evaluation.