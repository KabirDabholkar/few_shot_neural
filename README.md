# Few Shot Neural

A Python package for computing few-shot learning metrics for evaluating neural latent variable models.

## Installation

```bash
pip install -e .
```

## Usage

```python
from few_shot_neural import get_k_shot_co_bps, get_k_shot_co_bps_ensemble

# Compute k-shot co-bps
score = get_k_shot_co_bps(
    train_latents=train_latents,
    train_spikes=train_spikes,
    test_latents=test_latents,
    test_spikes=test_spikes,
    k_trials=5
)

# Compute ensemble k-shot co-bps
ensemble_score = get_k_shot_co_bps_ensemble(
    train_latents=train_latents,
    train_spikes=train_spikes,
    test_latents=test_latents,
    test_spikes=test_spikes,
    k_trials=5,
    n_ensemble=10
)
```