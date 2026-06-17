# ADR 003 — Choice of Evaluation Metrics

## Status
Accepted

## Context
Need to measure whether fine-tuning improved medical reasoning quality.

## Decision
Use three-layer evaluation:
1. Exact match accuracy on final answer line
2. Validation loss curve during training
3. Qualitative spot-checks (read 10-20 generations per run)

## Rationale
- Exact match: simple, deterministic, easy to compute at scale
- Val loss: available for free during training, catches overfitting early
- Qualitative: charts can mislead, reading output cannot
- LLM-as-judge (planned for M5): more nuanced than exact match,
  catches correct reasoning with different wording

## Consequences
- Exact match will be 0% for most runs (too strict for generative output)
- Val loss is the primary signal during M3 sweep comparison
- Qualitative checks are time-consuming but irreplaceable