# ADR 002 — Choice of LoRA Configuration

## Status
Accepted

## Context
Three fine-tuning options available: Full SFT, LoRA, QLoRA.
Hardware constraint: free T4 GPU with 15.6GB VRAM.

## Decision
Use QLoRA (LoRA on 4-bit quantized base) with r=16, alpha=16.

## Rationale
- Full SFT requires ~24GB VRAM — exceeds T4 capacity
- LoRA without quantization uses 8-10GB — fits but leaves little headroom
- QLoRA uses 6-8GB — comfortable on T4, allows larger eval batches
- r=16 is the recommended default for domain adaptation tasks
- alpha=16 (equal to r) is standard starting point per LoRA paper

## Consequences
- Slight quality cost vs full precision LoRA (typically <1% on narrow tasks)
- Cannot run full SFT for comparison without A100 access
- Adapter size ~100MB vs 5GB full model — easy to save and share