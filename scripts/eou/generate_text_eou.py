import os

# Set custom cache directories
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Hugging Face access token
access_token = None

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "meta-llama/Llama-3.1-70B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, token=access_token)

# text = "swear to me walton that he shall not escape that you will seek him and satisfy my vengeance in his death and do i dare to ask of you to undertake my pilgrimage to endure the hardships that i have undergone no"
text = "for once he was grateful to the forest because it had forbidden him to glance perpetually back at that dismal and pensive picture then he became aware of twigs hastily lopped off of bushes bent and torn of the uncovering through these careless means of an old path"

prompt = f"""
#### AI Assistant Information
- Response Requirements: Segment a given text into turns.
- Prohibited Content: The AI assistant must not provide illegal or harmful information and should refuse to answer politically sensitive questions.

#### Turn Segmentation Task
- Number of turns: Generate a number of turns you think is appropriate
- Exact match: The output should be exactly the same as the input text, except the turns are added.

#### Output Format
"turn_1": <first_turn>
"turn_2": <second_turn>
"turn_3": <third_turn>
"turn_4": <forth_turn> [EOU]

#### Example
This is an example of how the output should look like given the input text:

## Input:
he said how happy you will be i will do my best said the inn keeper of the pont du gard shutting up his knife well then we will go into paris.

## Output:
turn_1: he said how happy you will be
turn_2: i will do my best said the inn keeper of the pont du gard shutting up his knife
turn_3: well then we will go into paris [EOU]

#### Now parse the following text into turns:
{text}.

#### Output
"""

out = pipe(prompt, max_new_tokens=200, do_sample=False)
generated_text = out[0]["generated_text"]

print("Generated text:")
print(generated_text)
print("\n" + "="*50)
print("Parsed turns:")