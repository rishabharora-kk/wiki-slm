# Second-pass research summary (2026-11-26)
Key verified findings (full report in the session; sources below):
- 2510.06213 (ICLR 2026): OLMo/SmolLM3 trajectories + Pythia-160M WSD cooldowns at 60-625 tok/param; GPTQ only; no GGUF/QAT/EMA. Under WSD, error flat (160M) or slowly growing (70M); LAWA beats cooldown at 3-bit.
- 2509.22935 (Apple): 86M-759M models, int 1-6 bit LSQ/ParetoQ; QAT+cooldown fusion helps at 4/6 bit; no block formats, no GGUF.
- 2605.25966: 1,345 QAT runs 3M-350M; at 50M+ 33% warmdown optimal, <50M within seed noise.
- 2604.15167: Pythia-160m checkpoints, INT4 gap grows after FP convergence.
- Community: KL not perplexity (llama.cpp #4110, 2610.04403); imatrix calibrated on Wikipedia overfits Wikipedia benchmarks (Unsloth); QAT recipes opaque, Google's 54% claim unverified by third parties; embedding sensitivity disputed (llama.cpp #12741 vs ik_llama #288); Arm KleidiAI accelerates only Q4_0; Pi 5 energy/token is a solved tutorial (Arm learning path).
- Patents: none found (databases not queried directly); fake-quant QAT is standard industry practice -> claim only empirical findings.
Caveats: Reddit/HN threads were not directly retrievable; patent landscape unchecked.
