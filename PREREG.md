# Pre-registration v2 (DRAFT until frozen by commit SHA, before any trunk checkpoint exists)

Revised after a second research pass (docs/research-pass2.md). Prior work we do NOT claim:
cooldown/LR-decay drives PTQ error and LAWA helps (Catalan-Tatjer et al., 2510.06213, ICLR 2026);
QAT fused with cooldown beats post-hoc QAT (Dremov et al., 2509.22935); Q4_0 QAT for GGUF ships in Gemma 3;
schedule effects vanish below ~50M params (Thomassen, 2605.25966).

**Scope of the claim.** In llama.cpp block formats, at 50M params and 10-80 tokens/param:
(a) a two-sided test of Kumar (error rises with data) vs Geiping (flat under WSD);
(b) Q4_0-exact QAT fused into the cooldown vs post-hoc QAT vs imatrix PTQ;
(c) whether Q4_0-QAT robustness transfers to K-quants / I-quants.

## Setup
- Model: d=512, L=8, FFN 1536 (all multiples of 256), 8 heads / 4 KV, tied embeddings, SmolLM2 tokenizer.
  Report total AND non-embedding tokens/param.
- Cooldown: 20% of milestone tokens, 1-sqrt shape, final LR 0 (ablation: final LR = 10% of peak).
- Arms per milestone: R raw; C0 cooldown (EMA tracked inside); C2 fused Q4_0 QAT cooldown; C3 = C0 then post-hoc QAT (same tokens as C2's cooldown);
  LAWA from trunk checkpoints. 3 seeds of C0 and C2 at 20 tok/param.
- GGUF: pinned llama.cpp commit, `--pure` quants (secondary: default mixed), per-file tensor-type table via gguf-dump.
  imatrix calibrated on a split disjoint from evaluation. Evaluate on held-out Wikipedia AND an out-of-domain sample (FineWeb-Edu).
- External anchor: SmolLM2-135M/360M through the identical pipeline.
- Primary metric: mean KL(F32 GGUF || quantized) on >=1M held-out tokens; secondary dNLL, top-1 flip rate.
  Compare quantization error at matched FP validation loss as well as by tokens/param. CIs: document-level bootstrap.

## Hypotheses
- H1 (two-sided): slope of KL vs log(tokens/param) for C0 at pure Q4_0 is either positive (Kumar) or equivalent to zero within TOST bound
  +-0.1 x mean KL (Geiping). Either outcome is reported.
- H2: R checkpoints show smaller KL than C0 at matched tokens (cooldown increases error) - expected to replicate Geiping; replication stated as such.
- H3: C2 has lower Q4_0 KL than C3 and than C0+imatrix PTQ, with 95% CIs excluding 0 across 3 seeds; C2 FP NLL within 0.01 nats of C0.
- H4: EMA(C0) lowers KL vs C0 at matched FP loss. (LAWA reported as replication.)
- H5 (exploratory): transfer = fraction of C2's Q4_0 KL gain retained at Q4_K, Q3_K, IQ4_XS, Q2_K.

Effects count only if larger than 2x the seed SD. Null results are reported as nulls.
Confounds acknowledged: calibration/eval domain overlap, embedding share (~half of params), data repetition near 80 tok/param.
