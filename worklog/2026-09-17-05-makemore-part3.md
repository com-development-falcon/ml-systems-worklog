# 05 — makemore part 3: activations, gradients, BatchNorm

**Date:** 2026-09-17
**Phase:** 1, item 2 (makemore, part 3 of 5)

Took the part-2 MLP and learned to read its health instead of just its loss: histograms of activations and gradients per layer, the tanh saturation mask (white = |h| > 0.99, gradient 1-t^2 ~ 0 there; a solid white column = a dead neuron), per-layer mean/std/saturation printout, and the update:data ratio lr*grad.std()/data.std() (~1e-3 is healthy; std, not mean, because both distributions are zero-mean and std = RMS = typical magnitude).

The core math, now fully understood: h = sum of n products, variances add, so sigma_h = sqrt(n) * sigma_w * sigma_x. The sum itself grows activations by sqrt(fan_in) — that's what saturates tanh, no one chooses it. Kaiming init (sigma_w = 1/sqrt(n)) cancels the growth once at init; BatchNorm measures and divides out sigma every forward pass, with learnable gain/bias to negotiate back from unit-Gaussian. Verified sqrt(n) growth empirically with a randn matmul.

2026 caveat, recorded deliberately: BatchNorm itself is legacy in transformers — its batch coupling (train/eval trap) is why LayerNorm/RMSNorm won, and RMSNorm is literally just the divide-by-sigma step, per-example. The layer aged; the variance arithmetic and the read-the-vitals diagnostic habit are the current frontier (loss spikes, logit growth, muP are the scaled-up versions of tonight's white pixels).

Next: part 4, backprop ninja — tensor-level backprop by hand.

## Hours
~3h build+watch
