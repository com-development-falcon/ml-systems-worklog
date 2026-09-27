# 07 — makemore part 5: WaveNet — makemore complete

**Date:** 2026-09-27
**Phase:** 1, item 2 (makemore, part 5 of 5 — item complete)

Built the hierarchical (WaveNet-style) character LM: block_size 8, embeddings fused pairwise through three FlattenConsecutive -> Linear -> BatchNorm1d -> Tanh levels (8 -> 4 -> 2 -> 1), PyTorch-ified into my own Embedding/Linear/BatchNorm1d/Tanh/Sequential classes. 22,397 params.

Numbers: train 1.9444, **val 2.0275** (part-2 flat MLP: 2.1353; Karpathy lands ~2.03). Samples are name-shaped: daleena, hamiline, maddyx.

Bugs and lessons this part:
- lr decay silently never happened: `else: 0.01` without `lr =` — a legal Python statement that does nothing. Third species of silent bug (1: stale variables, 2: non-idempotent cells, 3: no-op statements).
- Non-idempotent dataset cell: in-place random.shuffle re-shuffles an already-shuffled list on rerun; seed makes the permutation deterministic, not the outcome. Fix: reload inputs inside the cell.
- Batch size is a property of the data fed, never of the model: same layers served B=32 in training and B=22,767 in full-dev eval; layer.out shape printouts are a snapshot of the *last* forward pass, not the training loop.
- BatchNorm asymmetry: with Linear -> BN -> Tanh, init gain (5/3) and pre-BN bias are irrelevant (BN erases scale, mean-subtraction erases bias -> bias=False); the output layer has no norm after it, so its 0.1 down-scaling still matters.
- Reading tools: plot log10(loss) (or log-scale axis) because loss improvements are multiplicative; 0.3 on log10 = 2x; perplexity = e^loss (val 2.0275 -> ppl ~7.6, uniform = 27).

Also nailed down before closing: BatchNorm1d on 3-D input — a channel/feature is a slot of the output vector (one weight column's recipe); statistics pool over every axis that indexes examples of that slot (batch AND position), so stats shape (1,1,68) over BxL samples, not (1,4,68) over B. Makemore complete: bigram counts -> softmax regression -> MLP -> diagnosed internals -> hand-derived backprop -> hierarchical model. Next: nanoGPT.

## Hours
~4h across two evenings
