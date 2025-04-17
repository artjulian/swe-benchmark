from datasets import load_dataset
import json

# Load the full test split from SWE-bench_Verified
dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

# Save as JSONL
with open("swe_bench_verified.jsonl", "w") as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write("\n")

print("Saved dataset as swe_bench_verified.jsonl")