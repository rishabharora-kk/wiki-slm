# Plan: "Does it survive Q4?"

Training dynamics and format-exact QAT for sub-100M LMs deployed with llama.cpp.

**Model S**: Llama arch, d=512, L=8, 8 heads / 4 KV heads, FFN 1536, SmolLM2 tokenizer (49,152, tied embeddings), ~50M params.
All matrix input dims are multiples of 256 so llama.cpp k-quants never fall back.
**Compute**: Kaggle T4x2 only (P100 unsupported by current image), fp16 + GradScaler, SDPA, DDP.

## Design
One WSD trunk at constant LR to 4B tokens; milestones at 0.5/1/2/4B tokens (10/20/40/80 tokens/param).
Each milestone branches into: R (raw), C0 (20% linear cooldown), C0-EMA, LAWA, C2 (cooldown with Q4_0-exact fake-quant STE), plus 2 extra C0 seeds at 1B.
Baselines: GGUF Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q4_0/Q3_K_M/Q2_K, with and without imatrix; SmolLM2-135M, Pythia-70M/160M, GPT-2.
Primary metric: dNLL = NLL(quant) - NLL(F16) on 1M held-out test tokens via llama.cpp, plus KL divergence.
Secondary: BPB, HellaSwag/Winogrande, laptop tokens/s, RAM, J/token, NLL-vs-bytes Pareto.

## Phases and acceptance checks
0. Harness + throughput: resume test (kill@100, resume, loss at 200 within 1e-3); >=40k tok/s on T4x2; LR sweep; fakequant bit-exact vs llama-quantize --pure Q4_0; GGUF F16 ppl within 1% of HF; PREREG.md SHA predates trunk.
1. Data: wikimedia/wikipedia 20231101.en, hash split by sha1(id)%1000 (0-4 val, 5-9 test), uint16 shards; no ID in two splits.
2. Trunk (~18-27 GPU-h): 4 milestones, no unrecovered spike >0.3 nats.
3. Branches (~15 GPU-h).
4. Eval (llama.cpp ppl/KL, lm-eval): one JSONL row per (model, quant); paired-bootstrap CIs.
5. Edge bench (RTX 3050 laptop): llama-bench, CV<5%, energy labelled exploratory.
6. Write-up: every number regenerates from results/ with one command.

## Honest framing
~4.8B tokens is about one pass over Wikipedia. The model is a research instrument, not a chatbot.
A null result (flat dNLL across 10-80 tok/param) is publishable and is pre-registered as such.
Prior art not to claim: first small wiki LM; first GGUF QAT; "overtraining hurts quantization" (Kumar 2411.04330, Ouyang 2411.17691); "LR decay causes it" (Catalan-Tatjer 2510.06213, ICLR 2026).
