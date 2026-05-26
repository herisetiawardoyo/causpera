# Contributing

Thank you for your interest in improving Causpera.

Causpera is an experimental research framework for causal-sensitive sparse
learning, adaptive memory consolidation, and uncertainty-gated computation. The
repository is intentionally small, inspectable, and research-oriented. New
contributions should preserve that character.

Contributions are welcome when they improve correctness, clarity,
reproducibility, test coverage, documentation, or experimental value.

## Development Setup

Create and activate a local virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
````

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Checks

Run the same checks used by CI before opening a pull request:

```bash
ruff check .
pytest
```

Pull requests should pass linting and tests before review.

## Contribution Scope

Useful contributions include:

* bug fixes;
* clearer documentation;
* additional unit tests;
* improved benchmark scripts;
* better diagnostics for masks, routing, uncertainty, or consolidation;
* stronger baseline comparisons;
* cleaner model APIs;
* reproducibility improvements;
* profiling utilities for latency, FLOPs, memory, or throughput.

Contributions should avoid unnecessary complexity. Causpera should remain easy
to read, inspect, and modify.

## Code Style

Please keep the codebase simple and explicit.

Prefer:

* typed function signatures where practical;
* small modules with clear responsibilities;
* readable tensor shapes and comments where useful;
* deterministic test cases;
* minimal hidden behavior;
* clear names for experimental assumptions.

Avoid:

* large untested abstractions;
* unnecessary framework dependencies;
* undocumented changes to model behavior;
* silent changes to public APIs;
* broad claims embedded in code comments or documentation.

## Tests

New behavior should include tests where practical.

Tests should cover:

* expected tensor shapes;
* loss behavior;
* routing behavior;
* attention-mask behavior;
* plasticity or consolidation behavior;
* failure cases for invalid input shapes or configuration values.

For experimental components, tests do not need to prove research effectiveness.
They should verify that the implementation behaves as intended.

## Documentation

Documentation updates are expected when a contribution changes:

* public APIs;
* model configuration;
* training behavior;
* inference behavior;
* routing statistics;
* loss terms;
* benchmark assumptions;
* research limitations.

Documentation should be precise and conservative. It should explain what the
component does, what assumptions it makes, and what it does not prove.

## Research Claims

Causpera uses practical causal proxies. Contributions should avoid claims that
the model proves causality from observational data.

A learned mask, attention pattern, or perturbation response should not be
described as causal proof. It may be described as:

* a causal-sensitive proxy;
* an inductive bias;
* a diagnostic signal;
* an experimental approximation;
* a hypothesis-generating mechanism.

New causal discovery components should document:

* their assumptions;
* their expected use case;
* their failure modes;
* their validation setup;
* their relationship to existing causal inference or causal discovery methods.

## Benchmarks and Results

Benchmark contributions should clearly state:

* dataset source;
* task definition;
* train, validation, and test split;
* baseline models;
* metrics;
* random seeds;
* hardware environment where relevant;
* limitations of the result.

Efficiency-related claims should not rely only on routing ratios or skipped
layers. Claims about latency, FLOPs, memory use, throughput, or energy should be
supported by profiler-level measurements.

## Pull Request Guidelines

Before opening a pull request:

1. run `ruff check .`;
2. run `pytest`;
3. update documentation if behavior changes;
4. keep the pull request focused;
5. describe the motivation and scope of the change;
6. mention any known limitations.

A good pull request should make the project easier to test, easier to inspect,
or more useful for controlled experimentation.

## License

By contributing to Causpera, you agree that your contribution will be released
under the repository license.