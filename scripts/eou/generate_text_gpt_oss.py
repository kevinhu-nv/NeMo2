import os

# Set custom cache directories
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

from transformers import pipeline
import torch

# Hugging Face access token
access_token = None

# Model configuration
model_id = "openai/gpt-oss-20b"

# Create pipeline with proper configuration
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
    token=access_token,
)

# Example 1: Basic conversation with harmony format
messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

print("=== Example 1: Basic Conversation ===")
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
print("\n" + "="*50 + "\n")

# Example 2: Speaker turn segmentation (your original task)
messages = [
    {"role": "user", "content": "Segment the following paragraph into speaker turns. Mark each turn with // at the end.\n\n"
     "He said how happy you will be. I will do my best, said the innkeeper. Well then, we will go into Paris."},
]

print("=== Example 2: Speaker Turn Segmentation ===")
outputs = pipe(
    messages,
    max_new_tokens=200,
    do_sample=False,
)
print(outputs[0]["generated_text"][-1])
print("\n" + "="*50 + "\n")

# Example 3: Reasoning with different levels
reasoning_prompts = [
    {"role": "user", "content": "Reasoning: low\n\nWhat is 15 * 23?"},
    {"role": "user", "content": "Reasoning: medium\n\nWhat is 15 * 23?"},
    {"role": "user", "content": "Reasoning: high\n\nWhat is 15 * 23?"},
]

reasoning_levels = ["Low", "Medium", "High"]

for i, (prompt, level) in enumerate(zip(reasoning_prompts, reasoning_levels)):
    print(f"=== Example 3.{i+1}: {level} Reasoning ===")
    outputs = pipe(
        [prompt],
        max_new_tokens=150,
        do_sample=False,
    )
    print(outputs[0]["generated_text"][-1])
    print("\n" + "="*50 + "\n")

# Example 4: Function calling demonstration
function_prompt = {
    "role": "user", 
    "content": "What's the weather like in San Francisco? Please provide the information in a structured format."
}

print("=== Example 4: Function Calling ===")
outputs = pipe(
    [function_prompt],
    max_new_tokens=300,
)
print(outputs[0]["generated_text"][-1]) 