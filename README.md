# RTLReviewBench

**RTLReviewBench** is a simulation-gated benchmark for evaluating whether Large Language Models (LLMs) can reliably audit a frozen Register-Transfer Level (RTL) Verilog/SystemVerilog implementation against a natural-language hardware specification *without writing code, editing implementations, or invoking an RTL simulator*.

Across **170 matched design pairs (340 balanced buggy/conforming cases)** spanning **126 unique hardware designs** and **17,623 evaluated API queries** over **12 commercial and open-weight LLMs**, RTLReviewBench separates false acceptance (**FAR**, missed RTL bugs) from false rejection (**FRR**, valid RTL rejected as buggy), contrasts real LLM failures against design-matched synthetic mutants, measures **Reason-First** prompting vs. **Verdict-First** schemas (including a `2x2` factorial ablation across `gpt-4.1`, `gpt-4.1-mini`, and `gemma-4-31b-it`: `+7.9 pp` key-order effect, `-1.3 pp` post-verdict tracing, `+6.6 pp` interaction), evaluates **5-sample stochastic self-consistency** (`T=0.7`) and **bidirectional pairwise comparison**, and audits all **1,309 Clean-114** (1,322 Full-120) True Positive explanations via blinded multi-judge + human verification.

---

## Corpus Architecture

| Track / Slice | Pairs | Cases | Design Composition | Role in Evaluation |
|---|---|---|---|---|
| **Core Clean-114** *(Primary)* | **114** | **228** | 17 ChipBench sequential (`arithmetic`/`assignment`) + 97 RTLRepairBench RealBugs (`90` base + `7` tuned); `80` sequential / `34` combinational | **Primary unbiased conformance split** (excludes 6 upstream oracle artifacts) |
| **Oracle-Artifact-6** *(Audit)* | 6 | 12 | `P018, P022, P028` (`#5`/`#10` inertial delays); `P020` (`always @(*)` time-0 event); `P047, P084` (`initial` power-on state) | Separated benchmark-validity audit (`50%` of Full-120 FNs stem from these 6 pairs) |
| **Core Full-120** | 120 | 240 | Unfiltered Core union (`Core Clean-114` + `Oracle-Artifact-6`) | Full sensitivity reporting alongside Clean-114 |
| **SynMut-50** | 50 | 100 | Single-operator semantic mutants (`op_swap`, `swap_logical`, `swap_ternary`, `flip_equality`, `off_by_one_literal`, `flip_reset_polarity`) | Synthetic contrast track (`36` sequential / `14` combinational) |
| **Matched-42** | 42 | 84 | Identical upstream VerilogEval v2 designs present in both **Core Clean-114** and **SynMut-50** | Within-design synthetic-vs-real contrast |

Every pair is compiled and differentially verified using **Icarus Verilog 11.0** (`iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb` / `vvp`) and sanitized against filename, `RefModule`, task-ID, and comment hint leakage.

---

## Core Clean-114 Leaderboard (12 Models, Greedy `T = 0.0`)

| Model | Paradigm | Clean-114 Acc. (95% CI) | FAR (Miss) | FRR (Over-Rej.) | Both-Right Pairs | Brier Score | Reason-First Acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| `claude-haiku-5.5-thinking` | Test-Time Reasoning | **97.4%** [94.4, 99.0] | 2.6% (3/114) | **1.8%** (2/114) | **94.7%** | **0.0280** | 97.4% |
| `claude-haiku-5.5` | Budget Commercial | **97.4%** [94.4, 99.0] | 3.5% (4/114) | **1.8%** (2/114) | **94.7%** | 0.0310 | 97.4% |
| `gemini-3.1-pro` | Flagship Commercial | 96.9% [93.8, 98.8] | 0.9% (1/114) | 5.3% (6/114) | 93.9% | 0.0299 | **98.2%** |
| `gemini-3.8-flash` | Budget Commercial | 93.4% [89.4, 96.3] | **0.0%** (0/114) | 13.2% (15/114) | 86.8% | 0.0590 | 93.9% |
| `claude-4.6-sonnet` | Flagship Commercial | 93.0% [88.9, 95.9] | 2.6% (3/114) | 11.4% (13/114) | 87.7% | 0.0600 | 93.4% |
| `gpt-oss-120b` | Open-Weight On-Prem | 91.7% [87.3, 94.9] | 1.8% (2/114) | 14.9% (17/114) | 84.2% | 0.0727 | 90.4% |
| `o4-mini` | Test-Time Reasoning | 90.8% [86.3, 94.2] | 2.6% (3/114) | 15.8% (18/114) | 82.5% | 0.1048 | 93.9% |
| `gpt-5.4` | Flagship Commercial | 83.8% [78.3, 88.3] | 3.5% (4/114) | 28.9% (33/114) | 69.3% | 0.1558 | 88.6% |
| `gemma-4-31b-it` | Open-Weight On-Prem | 78.9% [73.1, 84.1] | 3.5% (4/114) | 38.6% (44/114) | 57.9% | 0.2100 | 90.4% |
| `gpt-4.1-mini` | Budget Commercial | 71.9% [65.6, 77.7] | 2.6% (3/114) | 53.5% (61/114) | 45.6% | 0.2348 | 83.8% |
| `gpt-4.1` | Flagship Commercial | 69.7% [63.3, 75.6] | 1.8% (2/114) | 58.8% (67/114) | 40.4% | 0.2716 | 86.0% |
| `llama-3.3-70b-instruct` | Open-Weight On-Prem | 66.2% [59.7, 72.3] | 26.3% (30/114) | 41.2% (47/114) | 36.0% | 0.3278 | 67.1% |
| *Top-3 Diverse Maj. (in-sample best triple)* | Cross-Family Ensemble | **98.2%** [95.6, 99.5] | 0.9% (1/114) | 2.6% (3/114) | **96.5%** | --- | --- |
| *Top-3 Maj. (unconstrained 5-fold CV)* | Cross-Family Ensemble | 97.4% | --- | --- | --- | --- | --- |

---

## RTLReviewBench-Hard (Headroom Split)

`manifests/hard_public.jsonl` / `manifests/hard_gold.json`: the **22 Clean pairs (44 cases)** on which at least 6 of the 12 reviewers misclassify a sibling under Verdict-First.

| Model | Hard Acc. | Hard FAR | Hard FRR | Hard Pair-Both | Hard Reason-First Acc. |
|---|---:|---:|---:|---:|---:|
| `claude-haiku-5.5` | **88.6%** | 18.2% | **4.5%** | **77.3%** | **90.9%** |
| `claude-haiku-5.5-thinking` | **88.6%** | 13.6% | 9.1% | **77.3%** | **90.9%** |
| `gemini-3.1-pro` | 84.1% | 4.5% | 27.3% | 68.2% | **90.9%** |
| `gpt-oss-120b` | 75.0% | 9.1% | 40.9% | 54.5% | 72.7% |
| `gemini-3.8-flash` | 72.7% | **0.0%** | 54.5% | 45.5% | 77.3% |
| `claude-4.6-sonnet` | 70.5% | 13.6% | 45.5% | 50.0% | 77.3% |
| `o4-mini` | 70.5% | 13.6% | 45.5% | 45.5% | 81.8% |
| `gemma-4-31b-it` | 52.3% | 13.6% | 81.8% | 4.5% | 63.6% |
| `gpt-4.1` | 50.0% | 9.1% | 90.9% | 4.5% | 65.9% |
| `gpt-5.4` | 50.0% | 13.6% | 86.4% | 9.1% | 56.8% |
| `llama-3.3-70b-instruct` | 50.0% | 22.7% | 77.3% | 9.1% | 52.3% |
| `gpt-4.1-mini` | 47.7% | 9.1% | 95.5% | 4.5% | 63.6% |

Scores for these 12 models are in-sample (they defined the split). Hard is a headroom target for future reviewers; please report Pair-Both alongside accuracy.

---

## Licenses

This repository is released under the MIT License ([`LICENSE`](LICENSE)). Derived benchmark items retain their upstream licenses:

- **ChipBench:** MIT License
- **RTLRepairBench (RealBugs & Sem85, derived from VerilogEval v2):** Apache-2.0 / MIT License

---

## Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@misc{mahmud2026rtlreviewbench,
  title  = {{RTLReviewBench}: {LLM} Hardware Reviewers Rarely Miss Bugs but Routinely Reject Correct {RTL}},
  author = {Mahmud, Yasir},
  year   = {2026},
  url    = {https://github.com/yasirmahmud/RTLReviewBench},
  note   = {arXiv preprint (identifier to be added upon release)}
}
```

---

## Evaluating a New Model

Models receive only the blinded `*_public.jsonl` manifests; the `*_gold.json` files are used for scoring. The runner talks to any OpenAI-compatible endpoint configured through environment variables (nothing is stored in the repository):

```bash
pip install -e .
export OPENAI_API_KEY=<your key>
export OPENAI_BASE_URL=<OpenAI-compatible base URL; defaults to https://api.openai.com/v1>

# Score one split: core_clean (default), core_full, synmut50, or hard
python -m rtlreviewbench.run --model <model_id> --split core_clean
```

Optional paper-protocol variants (Reason-First, SynMut-50, Pairwise, SC@5):

```bash
python scripts/run_extensions.py --task reason_first --models <model_id>
python scripts/run_extensions.py --task pairwise --models <model_id>
```

## Reproducing the Paper's Numbers (Zero API Calls)

All raw model and judge responses are included, so every reported statistic (Clean-114/Full-120, Matched-42, SynMut-50, Reason-First, 2x2 ablation, Pairwise, SC@5, cross-validated ensembles, rationale audit) can be recomputed offline:

```bash
python -m pytest -q
python scripts/evaluate_extensions.py   # writes results_ext/all_metrics.json
```

---

## Repository Structure

```text
RTLReviewBench/
├── manifests/       # Benchmark splits: blinded *_public.jsonl inputs and *_gold.json labels
├── rtlreviewbench/  # Prompt construction, response parsing, scoring, and the run CLI
├── scripts/         # Paper-protocol runner and the offline evaluator
├── results_v2/      # Raw Verdict-First responses from the 12 evaluated models
├── results_ext/     # Raw responses for the extension studies + all_metrics.json
├── configs/         # Source provenance (upstream repos and commits) and model list
└── tests/           # Scoring and label-leakage tests
```
