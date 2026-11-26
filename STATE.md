# wiki-slm operator state (repo is the only memory)

Goal: execute docs/PLAN.md ("Does it survive Q4?"). Owner: Claude as CEO/operator, user authorized 24/7 autonomy
(2026-11-26). Hard limits: no spending, never read/print Kaggle credentials, user's HF_TOKEN is theirs to add.

## Done
- Repo public; plan + draft PREREG.md; data/prepare.py; train/train.py + fakequant.py (CPU smoke: trunk, QAT branch, resume all pass).

## In flight
- Kaggle CPU kernel `kingkiller2164/wiki-slm-prep` (private) tokenizing wikimedia/wikipedia 20231101.en -> /kaggle/working/out (uint16 shards, meta.json). Pushed 2026-11-26; started RUNNING.

## Next (in order)
1. When prep COMPLETE: `kaggle kernels output kingkiller2164/wiki-slm-prep -p <dir>` check meta.json (~4-5B tokens, no id overlap), upload as Kaggle dataset `wiki-smollm2-u16` (private first).
2. Phase 0 GPU kernel (T4x2, ~2h): measure tok/s (>=40k), torch.compile on sm_75?, LR sweep {1e-3,2e-3,4e-3} @80M tokens, resume test. Adjust --micro/--lr.
3. Read arXiv 2510.06213 fully, then freeze PREREG.md (commit SHA predates trunk).
4. write fakequant bit-exact check vs llama-quantize --pure Q4_0 (needs llama.cpp build) ; GGUF F16 ppl parity.
5. Trunk in commit-mode chained sessions (resume auto); then branches; eval; edge bench; write-up.

## Blocked
(none)

## Log
