# ADR 001 — Choice of Base Model Variant

## Status
Accepted

## Context
We needed to choose between gemma-4-E2B (base pretrained) and 
gemma-4-E2B-it (instruction tuned) as our starting point.

## Decision
Use google/gemma-4-E2B-it (instruction-tuned variant).

## Rationale
- Starting from instruction-tuned model is closer to real deployment scenario
- SFT step is genuinely additive: teaching domain reasoning on top of 
  existing instruction-following ability
- Base pretrained model would require more data and steps to learn 
  basic instruction following before learning medical reasoning

## Consequences
- Fine-tuning converges faster (model already knows how to follow instructions)
- Risk of catastrophic forgetting of general instruction following is lower
- Cannot measure improvement in raw instruction following ability