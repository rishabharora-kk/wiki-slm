# Pre-registration (DRAFT - freeze by commit SHA before any trunk checkpoint exists)

Threshold: an effect counts only if larger than 2x the seed standard deviation.
TODO before freezing: read Catalan-Tatjer et al. (arXiv 2510.06213) in full to check for sub-100M GGUF overlap.

- H1: for C0, dNLL at Q4_K_M and Q3_K_M rises monotonically with tokens per parameter.
- H2: R (raw constant-LR) checkpoints show a flat or smaller trend; most of the dNLL jump comes from the cooldown.
- H3: C2 cuts Q4_0 dNLL by >=50% vs C0 at <=0.01 nats F16 NLL cost, and beats imatrix PTQ.
- H4: C0-EMA and LAWA lower dNLL vs C0 at matched F16 NLL.
- H5 (exploratory): C2's gain partly transfers to Q4_K_M/Q3_K_M, not Q2_K.

Null: dNLL flat across 10-80 tokens/param and/or cooldown has no effect; reported as such.
