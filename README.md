# Gemma 4 E2B Medical Reasoning Fine-tuning

Fine-tuned Gemma 4 E2B (2.3B) on medical SFT dataset using QLoRA.

## Results
- **Best config:** r=32, LR=2e-4, QLoRA
- **Validation loss:** 2.765 (200 steps)
- **VRAM:** 9.5GB (T4 GPU)

## Adapters
- [Main repo](https://huggingface.co/your-username/gemma4-medical-lora)
- [M4 full precision](https://huggingface.co/your-username/gemma4-medical-m4-lora-full)
- [All M3 runs](https://huggingface.co/your-username?search=gemma4-medical)

## Report
See `report.md` for full technical analysis.

## Quick Start
```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "your-username/gemma4-medical-m3-lr2e4-r32-drop0"
)
```

✅ M0 Setup
✅ M1 Baseline
✅ M2 LoRA training
✅ M3 Hyperparameter sweep (5 runs)
✅ M4 QLoRA comparison
✅ M5 Early stopping
✅ Design notes
✅ Technical report
✅ All adapters on HF Hub
⏳ Demo video 