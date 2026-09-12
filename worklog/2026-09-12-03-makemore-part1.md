# 03 — makemore part 1: bigram language model

**Date:** 2026-09-12
**Phase:** 1, item 2 (makemore, part 1 of 5)

Built the bigram character-level LM on names.txt (32,032 names) two ways, typed from the video: (1) the counts model — 27x27 count matrix N, row-normalized with +1 smoothing into P; (2) the same model as a one-layer neural net — one-hot input, 27x27 weight matrix W, softmax, trained with NLL + 0.01*(W^2).mean() regularization, lr 50, 300 steps. Counts model loss ~2.45; trained net ~2.483; uniform baseline log(27) ~3.30.

What clicked:
- NLL is accumulated surprise. Rare-but-real events (names starting with "o", p=0.012) cost loss even for a perfect model; the model can't avoid paying for rare events, only avoid paying extra by keeping probabilities honest. The counts model is already the optimum of this loss.
- Connection to CS229: squared error is just NLL under a Gaussian output assumption; cross-entropy is NLL under a categorical one. Same principle, different output distribution.
- W.exp() never equals N, and shouldn't: softmax is invariant to adding a constant per row of logits, so exp(W) is only determined up to a per-row scale. "The net learns the counts" means equal after row-normalization — verified by comparing softmax(W) against N row-normalized. Logits are only defined up to a per-row constant; this is also why stable cross-entropy subtracts the max logit.

Next: makemore part 2 (MLP — splits, minibatches, learning rate).

## Hours
~3h build+watch
