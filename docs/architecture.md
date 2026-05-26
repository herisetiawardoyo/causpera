# Architecture

Causpera is built as a sequence classifier with three cooperating mechanisms.

```text
input [B, T, input_dim]
  |
  v
linear projection + learned position embedding
  |
  v
fast causal encoder layers
  |
  +--> fast pooled logits + uncertainty estimate
  |
  +--> uncertain samples continue through extra slow layers
  |
  v
classifier logits
```

## Causal Sparse Attention

Each attention layer estimates a soft edge mask with shape `[B, H, T, T]`,
where rows are target tokens and columns are source tokens. The estimator uses:

- directional compatibility between source and target token states;
- perturbation sensitivity from a lightweight hidden-state intervention proxy;
- a temporal lag prior that blocks future-to-past leakage by default.

The mask is applied in log-space to attention logits:

```text
attention_logits = qk / sqrt(d_head) + log(causal_mask)
```

This keeps the layer differentiable while discouraging attention on unlikely
temporal causes.

## Adaptive Synaptic Consolidation

The consolidation tracker keeps:

- `reference`: parameter snapshots after consolidation;
- `importance`: an exponential moving average of squared gradients or an
  empirical Fisher estimate.

The penalty is:

```text
strength * sum_i importance_i * (theta_i - reference_i)^2
```

This is useful for continual-learning experiments where a new task should not
freely overwrite parameters that were important for earlier tasks.

## Epistemic Energy Routing

Inference first runs only the fast layers. The model estimates normalized
predictive entropy from MC dropout samples on the fast hidden state.

Low-uncertainty samples return fast logits. High-uncertainty samples continue
through the remaining layers before classification. This creates actual depth
savings because the extra layers are skipped for fast-routed samples.

