from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

def load_medical_dataset(tokenizer: PreTrainedTokenizer):
    """Load and split the medical SFT dataset."""
    ds = load_dataset(
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "en",
        split="train"
    )

    # Split: last 500 for testing
    test_ds = ds.select(range(len(ds) - 500, len(ds)))
    train_ds = ds.select(range(len(ds) - 500))

    return train_ds, test_ds

def format_example(example: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """Format a single example into chat template format."""
    question = example["Question"]
    cot = example["Complex_CoT"]
    response = example["Response"]

    # Combine CoT and response as the assistant turn
    full_answer = f"{cot}\n\n**Final Answer:** {response}"

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": full_answer}]
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}