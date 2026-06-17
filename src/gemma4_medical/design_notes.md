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
- [ ] Inference check vs M1 baseline (pending Cell 5)