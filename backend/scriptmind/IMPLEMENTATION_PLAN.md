# ScriptMind implementation plan and decision record

## Objective

Add ScriptMind as a fifth, isolated research strategy. It performs binary scam
detection, current-intent classification, next-intent/utterance prediction, and an
evidence-linked preventive rationale. Existing Classical, FCNN, Transformer and
Unsupervised behavior and artifacts are out of scope and immutable inputs.

## Isolation contract

- Existing datasets, embeddings, split manifests and experiment results are read-only.
- Derived data is written only to `backend/datasets/dataset_scriptmind/`.
- Models and metrics are written only to `backend/experiment_results/scriptmind/`.
- Existing SLURM files, including `array_grid.sh`, are not reused or modified.
- API credentials come from environment variables. Provider billing is never enabled
  by code.

## Data and annotation protocol

1. Inventory 9,776 training conversations and reconstruct external-validation groups
   from `n_turns` resets. Create deterministic IDs only in derived files.
2. Group exact/near duplicates and split conversations 70/10/20 before prefixes.
   External validation remains untouched by development.
3. Apply Horus Crime Script Taxonomy v1 with one primary intent per Suspect turn.
4. Annotate 32 deterministic shards with Gemini and local `gpt-oss-20b`; accept exact
   label/stage agreement and send only disagreements to Gemini adjudication.
5. Require pilot Cohen's kappa >= 0.75 and arbitration below 15%. Audit 200 adjudicated
   examples and require agreement >= 0.80 before treating labels as final.
6. Build Scam examples from conversation prefixes and the next real Suspect turn.
   Balance Ham 1:1 only in training; keep natural test/external prevalence.
7. Export analytical Parquet, chat JSONL, transition counts and standardized residuals.

The first full read-only inventory (2026-09-15) found 9,776 training conversations,
1,083 reconstructed external-validation conversations, 141,990 parsed turns, 372
quarantined conversations, 8,591 duplicate groups, and 48,289 Suspect-turn annotation
items. Source hashes were unchanged after the run. These numbers are observations, not
hard-coded acceptance values; later source revisions must create a new manifest.

## Model protocol

- Models: Llama 3.2 1B Instruct and Llama 3.1 8B Instruct.
- QLoRA: NF4 4-bit, BF16, rank 16, alpha 32, dropout 0.05, all linear attention/FFN
  projections, Paged AdamW, LR 1e-4, 5 epochs, effective batch 64, clipping 1.0.
- Context: 4,096 tokens; retain the latest prompt tokens and the complete target.
- Seeds: 42, 52 and 62 over one fixed split.
- Ablations: detection-only, no-next-utterance, and external-only versus all training
  origins.

## Evaluation and acceptance

Measure binary metrics, intent exact match/macro-F1, ROUGE-L/semantic next-utterance
similarity, JSON validity, evidence validity, token usage and latency. Aggregate three
seeds with bootstrap 95% intervals; use McNemar and Holm correction in final analysis.
An LLM judge is primary only after correlation >= 0.70 with the human audit.

The implementation is accepted when it can produce an isolated CSID, six primary
checkpoints, internal/external evaluations, ablations, consolidated report and notebook,
while source dataset/embedding/cache hashes remain unchanged. The cognitive participant
study is a later ethics-gated project and is not simulated with LLM personas.

## SLURM DAG

```text
audit -> {gemini[32%1], local[32%2]} -> adjudicate[32%1] -> build
      -> train[2 models x 3 seeds] -> evaluate -> report
```

The submission script supports `--dry-run`, `--resume`, `--force`, `--from STAGE`, and
`--only STAGE`. Resume is the default and every stage uses manifests/done markers.

The metrics notebook is `backend/notebooks/ScriptMind_metrics.ipynb`. It reads inventory,
agreement, transition, model/seed/split, ablation, validity, efficiency, judge, and
read-only baseline artifacts without starting jobs.
