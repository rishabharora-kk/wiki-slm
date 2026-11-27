# wiki-slm operator state (repo is the only memory)

Goal: execute docs/PLAN.md ("Does it survive Q4?"). Owner: Claude as CEO/operator, user authorized 24/7 autonomy
(2026-11-26). Hard limits: no spending, never read/print Kaggle credentials, user's HF_TOKEN is theirs to add.

## Done
- Repo public; plan + draft PREREG.md; data/prepare.py; train/train.py + fakequant.py (CPU smoke: trunk, QAT branch, resume all pass).

## In flight
- Kaggle CPU kernel `kingkiller2164/wiki-slm-prep` (private) tokenizing wikimedia/wikipedia 20231101.en -> /kaggle/working/out (uint16 shards, meta.json). Pushed 2026-11-26; started RUNNING.

## Decisions (2026-11-26)
- Second research pass (docs/research-pass2.md) narrowed the claim; PREREG.md rewritten to v2 (still DRAFT). Trainer cooldown is now 1-sqrt with --final_lr_frac.
- HF_TOKEN is NOT needed until final model upload; user adds it as a Kaggle Secret on the training notebook then.
- Training kernel can use prep kernel output via kernel_sources (no local copy needed).

## TODO from pass 2
- C3 (post-hoc QAT) arm: needs a branch mode from a finished C0 final.pt (no optimizer state) - add --init_from.
- Reserve cooldown/eval shards (check meta.json token total vs 4B trunk + 0.8B cooldown); separate imatrix calibration split; OOD eval set (FineWeb-Edu sample).
- Pin llama.cpp commit; --pure quants; gguf-dump type table; fakequant bit-match test (max abs diff 0).
- SmolLM2-135M/360M anchor through same eval; 3 seeds C0/C2 @20 tok/param.

## Next (in order)
1. When prep COMPLETE: `kaggle kernels output kingkiller2164/wiki-slm-prep -p <dir>` check meta.json (~4-5B tokens, no id overlap), upload as Kaggle dataset `wiki-smollm2-u16` (private first).
2. Phase 0 GPU kernel (T4x2, ~2h): measure tok/s (>=40k), torch.compile on sm_75?, LR sweep {1e-3,2e-3,4e-3} @80M tokens, resume test. Adjust --micro/--lr.
3. Read arXiv 2510.06213 fully, then freeze PREREG.md (commit SHA predates trunk).
4. write fakequant bit-exact check vs llama-quantize --pure Q4_0 (needs llama.cpp build) ; GGUF F16 ppl parity.
5. Trunk in commit-mode chained sessions (resume auto); then branches; eval; edge bench; write-up.

## Blocked
(none)

## Log
- 2026-11-27: Phase 0 v1 FAILED fast (~2.5 min GPU): CUDA OOM at --micro 16 (3GB fp32 logits). Fixed to micro 8, relaunched as v2 (commit 'Fix OOM'). GPU used so far: <0.1h.
- 2026-11-26: prep COMPLETE (4.88B train tok). Phase 0 GPU kernel kingkiller2164/wiki-slm-phase0 pushed (T4x2, kernel_sources=prep; est ~2 GPU-h). Read its log/outputs next: tok/s, LR sweep val losses (lr1e-3/2e-3/4e-3), resume check.
