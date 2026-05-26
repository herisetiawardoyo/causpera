# Causpera

Causpera is an experimental PyTorch framework for exploring how causal-sensitive
sparse learning, adaptive memory consolidation, and uncertainty-gated computation
can be combined in a single model architecture.

The project is designed as a practical research toolkit. It provides a clean
package structure, explicit assumptions, a temporal toy benchmark, automated
tests, and documentation suitable for a public GitHub repository.

Causpera builds on several research directions that are often studied
separately: sparse attention, causal reasoning, continual learning, and adaptive
computation. The purpose of this implementation is to make their integration
testable in code, rather than presenting the model as a completed solution to
causal discovery or continual learning.

The project combines three practical mechanisms:

| Mechanism | Module | Purpose |
| --- | --- | --- |
| Causal-sensitive sparse attention | `causpera.attention` | Modulates attention with a temporal causal-mask proxy. |
| Causal-importance consolidation | `causpera.plasticity` | Tracks important parameters and penalizes destructive drift. |
| Uncertainty-gated energy routing | `causpera.model` | Uses fast layers for confident inputs and deeper layers for uncertain inputs. |

## Status

This is an experimental research toolkit. The causal mask should be understood
as a differentiable approximation based on temporal order, directional
compatibility, and perturbation sensitivity. It is intended for experimentation,
inspection, and benchmarking, not as formal proof of causality from observational
data.

The current implementation is most useful for:

- testing architectural ideas around selective attention and adaptive inference;
- experimenting with parameter-importance tracking for continual learning;
- studying how uncertainty can influence compute allocation;
- building small benchmarks before moving to more realistic datasets.

## Install

```bash
git clone https://github.com/your-org/causpera.git
cd causpera
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
````

On macOS or Linux:

```bash
source .venv/bin/activate
```

## Quick Start

```python
import torch

from causpera import CausperaClassifier, CausperaConfig

config = CausperaConfig(
    input_dim=8,
    d_model=64,
    num_classes=2,
    num_layers_fast=1,
    num_layers_total=3,
    num_heads=4,
    max_lag=3,
)

model = CausperaClassifier(config)
x = torch.randn(16, 12, 8)
y = torch.randint(0, 2, (16,))

output = model(x)
loss, parts = model.compute_loss(output, y)
loss.backward()
model.update_importance(causal_strength=output.causal_mask.mean())

inference = model.infer(x)
print(inference.logits.shape)
print(inference.stats.fast_ratio)
```

## Train the Toy Benchmark

```bash
python examples/train_toy.py --epochs 5
```

The toy task generates temporal sequences where only a few lagged features
determine the label. Other features are correlated distractors. This makes it
useful for smoke-testing causal-mask behavior without overstating real-world
causal discovery.

## Tests and Quality Checks

```bash
ruff check .
pytest
```

## Repository Layout

```text
causpera/
|-- pyproject.toml
|-- README.md
|-- docs/
|   |-- architecture.md
|   |-- gap_analysis.md
|   `-- research_limitations.md
|-- examples/
|   `-- train_toy.py
|-- src/
|   `-- causpera/
|       |-- attention.py
|       |-- config.py
|       |-- data.py
|       |-- losses.py
|       |-- metrics.py
|       |-- model.py
|       `-- plasticity.py
`-- tests/
    |-- test_attention.py
    |-- test_data.py
    |-- test_loss.py
    |-- test_model.py
    `-- test_plasticity.py
```

## Design Rationale

Causpera is organized around a practical research question:

Can a model learn more selectively, preserve useful knowledge over time, and
adjust its computation depth according to uncertainty?

The implementation approaches this question through three design choices.

First, attention is made more selective through a temporal causal-mask proxy.
The mask is not treated as causal proof. Instead, it provides a trainable signal
that encourages the model to focus on temporally plausible and perturbation-
sensitive relationships.

Second, the model tracks parameter importance during learning. This allows
important parameters to be protected against destructive drift, while less
critical parts of the model remain flexible enough to adapt to new data.

Third, inference is routed through different computation depths. Inputs with
higher confidence can use a lighter path, while more uncertain inputs can use
deeper layers. This makes compute allocation part of the model behavior rather
than a fixed architectural assumption.

## Research Scope

Causpera is intended for experimentation and ablation studies. It should be
evaluated through controlled benchmarks before being applied to real-world
decision systems.

The current version focuses on:

* temporal sequence classification;
* causal-mask diagnostics;
* adaptive fast/deep inference;
* parameter-importance tracking;
* toy data with known lagged causal structure;
* unit tests for core model behavior.

Future work may include stronger causal discovery methods, intervention-based
datasets, richer continual-learning benchmarks, calibration metrics, and
comparisons against transformer baselines, EWC-style methods, and conditional
computation models.

## Limitations

The current implementation has important limitations:

* The causal mask is an approximation, not a formal causal identification method.
* Observational data alone is not sufficient to establish causality in general.
* Perturbation sensitivity is a proxy and should be validated carefully.
* The toy benchmark is useful for smoke testing but does not represent real-world
  causal complexity.
* Adaptive routing should be benchmarked against fixed-depth baselines before
  making efficiency claims.
* Consolidation reduces destructive drift but does not eliminate catastrophic
  forgetting in all settings.

These limitations are intentional design boundaries. They are documented so the
project can be extended and evaluated without overstating what the current
prototype demonstrates.

## Why This Version

This version emphasizes a more defensible public research position:

* Causpera is presented as an experimental integration framework, not a fully
  proven new learning paradigm.
* Causal language is used carefully and tied to approximation, diagnostics, and
  empirical testing.
* The implementation focuses on testable mechanisms rather than broad claims.
* Adaptive inference saves transformer depth, not only classifier-head depth.
* Tests cover attention, data, loss, routing, and plasticity basics.
* Documentation separates architecture, gap analysis, and research limitations.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.