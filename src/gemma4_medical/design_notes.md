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

