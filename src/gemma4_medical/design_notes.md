# Design Notes

## M1 Baseline Observations (10 manual reads)

### Example 1 — Triangle of Doom question
- Base model confused Triangle of Doom with Hesselbach's triangle (different anatomical regions)
- Got boundaries wrong: said "inferior epigastric artery + inguinal canal" vs correct "vas deferens + testicular vessels"
- Generated confident but fabricated anatomy
- No explicit reasoning chain — jumped straight to answer
- Expected answer was cleaner, more accurate, and included clinical context

### General failure patterns to watch for:
- [ ] Wrong anatomical structure names
- [ ] Fabricated drug names/dosages
- [ ] Missing step-by-step reasoning chain
- [ ] Confident wrong answers

### Failure Patterns Observed
1. Confuses anatomical structures (Triangle of Doom vs Hesselbach's)
2. Gives mechanism/pathway instead of specific finding asked
3. Wrong clinical numbers (HIV transmission: said 1-5%, correct is 15-45%)
4. Misses predisposing/underlying factor — answers surface question instead
5. No explicit chain-of-thought reasoning steps
6. Truncation at 256 tokens mid-answer on calculation questions

### Per-example verdict
- Ex1 Triangle of Doom: WRONG (wrong boundaries)
- Ex2 Brain biopsy: WRONG (vascular dementia vs Lewy bodies)  
- Ex3 Boerhaave syndrome: CORRECT
- Ex4 Burns nutrition: PARTIAL (correct approach, truncated)
- Ex5 Erythropoietin: PARTIAL (pathway correct, missed apoptosis angle)
- Ex6 Neurofibrillary tangles: WRONG (missed amyloid plaques)
- Ex7 HIV transmission: WRONG (1-5% vs correct 15-45%)
- Ex8 GI bleed cirrhosis: WRONG (missed TIPS)
- Ex9 Leg infection: WRONG (missed tinea pedis)
- Ex10 TB hepatotoxicity: WRONG (missed acetylation)

### Baseline exact-match accuracy: 0.0% (0/50)
### Qualitative score: ~1/10 fully correct, 2/10 partial

## What fine-tuning should fix
- Explicit CoT reasoning before final answer
- Domain-specific clinical knowledge gaps
- Correct clinical numbers and statistics
- Identifying root cause / predisposing factors
_______________________________________________________________________________________

## M2 Training Run — lora-r16-lr2e4-baseline

### Config
- Model: gemma-4-E2B-it (4-bit QLoRA)
- LoRA rank: 16, alpha: 16, dropout: 0.0
- Learning rate: 2e-4, cosine scheduler
- Batch size: 1, gradient accumulation: 8 (effective batch: 8)
- Max seq length: 512 (reduced from 2048 due to T4 VRAM constraints)
- Train size: 5000 examples, Eval size: 100 examples
- Epochs: 1 (625 total steps)
- Hardware: Colab T4 (15GB VRAM)

### Environment Issues Encountered
- Kaggle had Pillow version conflict (11.3.0 vs 12.2.0) with Unsloth
- Switched to Google Colab which has cleaner base environment
- OOM errors resolved by: load_in_4bit=True, max_seq_length=512, 
  eval set reduced to 100 examples, per_device_eval_batch_size=1

### Training Dynamics
| Step | Train Loss | Val Loss |
|------|------------|----------|
| 50   | 0.511      | 2.925    |
| 100  | 0.347      | 2.808    |
| 150  | 0.307      | 2.625    |
| 200  | 0.289      | 2.592    |
| 250  | 0.278      | 2.583    |
| 300  | 0.273      | 2.576    |

### Observations
- Both train and val loss dropped consistently — healthy learning ✅
- Diminishing returns after step 200 — model converges fast on 5k examples
- No overfitting observed — val loss followed train loss down throughout
- Justifies using max_steps=200 for M3 sweep runs

### M2 Acceptance Criteria (from assignment)
- [x] Run completes without OOM
- [x] Training loss decreases monotonically after warmup
- [x] Adapter saves cleanly
- [x] Inference check vs M1 baseline


## M2 vs M1 Inference Comparison

### Triangle of Doom question
- M1: Confused with Hesselbach's, no reasoning shown
- M2: Shows CoT reasoning structure ✅, still wrong anatomy ⚠️
- Improvement: Reasoning FORMAT learned, factual accuracy needs more data/longer sequences
- Root cause of remaining errors: max_seq_length=512 truncates long medical answers during training
- Fix for M3: Consider whether longer sequences improve factual accuracy


## M3 Hyperparameter Sweep Results

### Run 1 — LR=5e-5 (m3-lr5e5-r16-drop0)
- Final train loss: 0.445 | Final val loss: 2.918
- Verdict: UNDERFITTING ❌
- Loss curve nearly flat after step 150
- LR too low — insufficient weight updates in 200 steps
- Quote for report: "LR=5e-5 shows clear underfitting — loss curve 
  nearly flat after step 150, suggesting the learning rate is 
  insufficient for meaningful weight updates within 200 steps"

### Run 2 — LR=5e-4 (m3-lr5e4-r16-drop0)
- Final train loss: 0.283 | Final val loss: 2.742
- Verdict: BEST VAL LOSS SO FAR ✅
- Contrary to hypothesis — no instability observed
- grad_norm stayed low (0.10) throughout — stable training
- Learns faster than M2 baseline in early steps
- Best LR candidate so far

### Run 3 — r=8 (m3-lr2e4-r8-drop0)
- Final train loss: 0.478 | Final val loss: 2.886
- Verdict: WORSE THAN r=16 ❌
- Val loss unstable at step 150 (went up then down)
- Smaller adapter lacks capacity for medical reasoning

### Run 4 — r=32 (m3-lr2e4-r32-drop0)
- Hypothesis: more capacity, might overfit on 5k examples
- Watch for: train-val gap growing after step 150

### Run 5 - 
-
-

## M3 Hyperparameter Sweep — Final Results

### All Runs Summary
| Run  |  LR  |Rank|Dropout|Val Loss@200|   Verdict    |
|------|------|----|-------|------------|--------------|
| M2   | 2e-4 | 16 |  0.0  |   2.485*   | Best overall |
| Run1 | 5e-5 | 16 |  0.0  |   2.918    | Underfit     |
| Run2 | 5e-4 | 16 |  0.0  |   2.742    | Best LR      |
| Run3 | 2e-4 | 8  |  0.0  |   2.886    | Low capacity |
| Run4 | 2e-4 | 32 |  0.0  |   2.765    | Best rank    |
| Run5 | 2e-4 | 16 |  0.05 |   2.818    | Dropout hurts|
*M2 ran full 625 steps, others 200 steps

### Key Findings
1. LR=5e-5 — clear underfitting, loss barely moves
2. LR=5e-4 — best among 200-step LR runs, stable training
3. r=8 — worse than r=16, insufficient capacity
4. r=32 — marginally better than r=16, more capacity helps
5. dropout=0.05 — slightly worse val loss, not beneficial here
6. Best config for M4: LR=2e-4, r=16 (M2 config, proven stable)

### Environment Notes
- All runs: max_seq_length=512, load_in_4bit=True, 5000 train examples
- Colab T4 15GB, ~25 min per 200-step run
- Adapters pushed to HuggingFace Hub after each run


## M4 — QLoRA vs LoRA Comparison

|    Config     | Peak VRAM |Val Loss@200|   Time   |
|---------------|-----------|------------|----------|
| QLoRA (4-bit) |  9.5 GB   |    2.765   | 25 min   |
| LoRA (full)   |  11.25 GB |    2.787   | 22.2 min |

### Findings
- QLoRA used 1.75GB less VRAM than full precision LoRA
- QLoRA achieved BETTER val loss (2.765 vs 2.787)
- Full precision was slightly faster (22 vs 25 min)
- Conclusion: QLoRA is the better choice for this task on T4
- 4-bit quantization cost = zero quality loss on narrow medical domain
- Report quote: "QLoDA matched and slightly exceeded full-precision 
  LoRA quality while using 15% less VRAM, confirming that 4-bit 
  quantization is appropriate for domain adaptation on narrow tasks"

