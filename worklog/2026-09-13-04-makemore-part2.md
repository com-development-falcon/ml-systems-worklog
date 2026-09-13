# 04 — makemore part 2: MLP

**Date:** 2026-09-13
**Phase:** 1, item 2 (makemore, part 2 of 5)

Built the Bengio-style MLP character LM from the video: embedding table C, hidden tanh layer, train/dev/test splits (80/10/10), minibatch training, learning-rate sweep via torch.linspace over log-space. Then tuned past the video's result.

Numbers: dev loss **2.1353** (video lands ~2.17; bigram baseline 2.45; uniform 3.30). Train/dev gap small, so not memorizing.

Two bugs, one lesson: both sampling bugs were stale notebook variables — first the context update read a dead variable, then after renaming, the model input still did. Code ran fine both times; output was garbage both times. ML bugs are silent and semantic: sampling from the model is a debugging tool, not a demo. New habits: distinct names for eval results (loss_tr/loss_dev), `del` old names after renames, Restart & Run All when numbers look impossible.

Also learned: minibatch loss is a noisy estimate — real evaluation is a deliberate full pass over train and dev; the jitter in the loss curve is sampling noise, not learning.

Next: makemore part 3 (activations, gradients, BatchNorm).

## Hours
~4h build+watch+debug
