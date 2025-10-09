import os
import json
import time
from typing import List, Dict
from pathlib import Path

# Set custom cache directories
os.environ["HF_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["TORCH_HOME"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"
os.environ["NEMO_CACHE_DIR"] = "/lustre/fsw/portfolios/convai/users/kevinhu/results/HFCACHE"

# Import Mistral inference libraries
try:
    from mistral_inference.transformer import Transformer
    from mistral_inference.generate import generate
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    MISTRAL_INFERENCE_AVAILABLE = True
except ImportError:
    print("Mistral inference not available, falling back to transformers")
    MISTRAL_INFERENCE_AVAILABLE = False
    from transformers import pipeline

def setup_mistral_inference():
    """Setup Mistral inference model and tokenizer."""
    if not MISTRAL_INFERENCE_AVAILABLE:
        return None, None
    
    # Path to downloaded model (you'll need to download it first)
    model_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please download the model first using:")
        print("huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --include 'params.json,consolidated.safetensors,tokenizer.model.v3' --local-dir ~/mistral_models/7B-Instruct-v0.3")
        return None, None
    
    print("Loading Mistral tokenizer...")
    tokenizer = MistralTokenizer.from_file(str(model_path / "tokenizer.model.v3"))
    
    print("Loading Mistral model...")
    model = Transformer.from_folder(str(model_path))
    
    return model, tokenizer

def setup_transformers_fallback():
    """Setup transformers as fallback."""
    access_token = None
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    
    print("Setting up transformers pipeline...")
    pipe = pipeline("text-generation", model=model_id, tokenizer=model_id, token=access_token)
    return pipe

def create_turn_detection_prompt(text: str) -> str:
    """Create prompt for turn detection task."""
    return f"""<s>[INST] Segment the following text into speaker turns. Add [EOU] at the end of each turn.

Text: {text}

Output: [/INST]"""

def create_timestamp_turn_detection_prompt(text: str) -> str:
    """Create prompt for timestamp-based turn detection task."""

    return f"""
#### AI Assistant Requirements:
- General Requirements: Segment a given text into turns
- Input text: The <|integer|> means the timestamps of a word. Each word has a start and an end timestamp. The integer is the frame index with a frame rate of 80 ms.
- Segmentation tips: From one turn to the next, usually there is a duration gap.
- Number of words in a turn: Include at least 5 words in every turn.
- Include an explanation of your reasoning.

#### Turn Segmentation Task
- Number of turns: Generate a number of turns you think is appropriate
- Exact match: The output should be exactly the same as the Input_without_timestamps, except the turns are added.

#### Output Format
"turn_1": <first_turn>
"turn_2": <second_turn>
"turn_3": <third_turn>
"turn_4": <forth_turn>

#### Example
This is an example of how the output should look like given the input text:

## Input:
<|3|> he <|4|> <|6|> said <|7|> <|15|> how <|16|> <|19|> happy <|20|> <|24|> you <|25|> <|27|> will <|28|> <|30|> be <|31|> <|54|> i <|59|> <|59|> will <|60|> <|63|> do <|64|> <|66|> my <|67|> <|70|> best <|71|> <|83|> said <|84|> <|86|> the <|87|> <|88|> inn <|91|> <|92|> keeper <|96|> <|97|> of <|98|> <|99|> the <|100|> <|100|> pont <|104|> <|105|> du <|106|> <|108|> gard <|112|> <|117|> shutting <|123|> <|123|> up <|124|> <|126|> his <|127|> <|128|> knife <|132|> <|155|> well <|156|> <|160|> then <|163|> <|166|> we <|167|> <|168|> will <|169|> <|171|> go <|172|> <|173|> into <|174|> <|175|> paris <|180|>

## Input_without_timestamps:
he said how happy you will be i will do my best said the inn keeper of the pont du gard shutting up his knife well then we will go into paris

## Output:
turn_1: he said how happy you will be
turn_2: i will do my best said the inn keeper of the pont du gard shutting up his knife
turn_3: well then we will go into paris
## End of Output

Now parse the following text into turns:
{text}

#### Output
"""

def process_single_text_mistral(model, tokenizer, text: str, text_id: str = None, use_timestamps: bool = False) -> Dict:
    """Process a single text using Mistral inference."""
    try:
        # Create prompt based on whether we're using timestamps
        if use_timestamps:
            prompt = create_timestamp_turn_detection_prompt(text)
        else:
            prompt = create_turn_detection_prompt(text)
        
        # Create completion request
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        
        # Encode to tokens
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        
        # Generate
        out_tokens, _ = generate(
            [tokens],
            model,
            max_tokens=200,
            temperature=0.0,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        # Decode result
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        
        # Extract only the generated part (after the prompt)
        prompt_length = len(tokenizer.instruct_tokenizer.tokenizer.decode(tokens))
        response = result[prompt_length:].strip()
        
        return {
            "text_id": text_id,
            "input_text": text,
            "generated_text": response,
            "success": True,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "text_id": text_id,
            "input_text": text,
            "error": str(e),
            "success": False,
            "timestamp": time.time()
        }

def process_single_text_transformers(pipe, text: str, text_id: str = None, use_timestamps: bool = False) -> Dict:
    """Process a single text using transformers fallback."""
    
    # Create prompt based on whether we're using timestamps
    if use_timestamps:
        prompt = create_timestamp_turn_detection_prompt(text)
    else:
        prompt = create_turn_detection_prompt(text)
    
    out = pipe(prompt, max_new_tokens=200, do_sample=False)
    generated_text = out[0]["generated_text"]

    print(f"Generated text: {generated_text}")

    import pdb; pdb.set_trace()

    # TODO(kevinhu): Add fixes for generated turns:
    # 1: Ensure each turn has at least 5 words
    # 2: Ensure there is at least 1.5 second gaps between turns
    
    # Extract only the generated part
    prompt_length = len(prompt)
    response = generated_text[prompt_length:].strip()
    
    return {
        "text_id": text_id,
        "input_text": text,
        "generated_text": response,
        "success": True,
        "timestamp": time.time()
    }

def batch_process_mistral(texts: List[str], model, tokenizer, batch_size: int = 4, use_timestamps: bool = False) -> List[Dict]:
    """Process texts in batches using Mistral inference."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = [f"text_{j+1}" for j in range(i, min(i+batch_size, len(texts)))]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Prepare batch of tokenized inputs
        token_batches = []
        for text in batch_texts:
            if use_timestamps:
                prompt = create_timestamp_turn_detection_prompt(text)
            else:
                prompt = create_turn_detection_prompt(text)
            
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = tokenizer.encode_chat_completion(completion_request).tokens
            token_batches.append(tokens)
        
        try:
            # Run batch generation
            out_tokens_batch, _ = generate(
                token_batches,
                model,
                max_tokens=200,
                temperature=0.0,
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            
            # Process results
            for j, (text, text_id, out_tokens) in enumerate(zip(batch_texts, batch_ids, out_tokens_batch)):
                result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens)
                
                # Extract only the generated part
                prompt_length = len(tokenizer.instruct_tokenizer.tokenizer.decode(token_batches[j]))
                response = result[prompt_length:].strip()
                
                results.append({
                    "text_id": text_id,
                    "input_text": text,
                    "generated_text": response,
                    "success": True,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            print(f"Batch processing failed: {e}")
            # Fallback to individual processing
            for text, text_id in zip(batch_texts, batch_ids):
                result = process_single_text_mistral(model, tokenizer, text, text_id, use_timestamps)
                results.append(result)
    
    return results

def batch_process_transformers(texts: List[str], pipe, batch_size: int = 4, use_timestamps: bool = False) -> List[Dict]:
    """Process texts in batches using transformers."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = [f"text_{j+1}" for j in range(i, min(i+batch_size, len(texts)))]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Process each text individually (transformers pipeline doesn't support true batching)
        for text, text_id in zip(batch_texts, batch_ids):
            result = process_single_text_transformers(pipe, text, text_id, use_timestamps)
            results.append(result)
    
    return results

def save_results(results: List[Dict], output_file: str = "mistral_batch_results.json"):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def parse_turns_from_output(generated_text: str) -> List[str]:
    """Parse turns from the generated output."""
    turns = []
    
    # Look for [EOU] markers
    if "[EOU]" in generated_text:
        parts = generated_text.split("[EOU]")
        for part in parts:
            part = part.strip()
            if part:
                turns.append(part)
    # Look for turn_1, turn_2 format
    elif "turn_1:" in generated_text:
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('turn_'):
                # Extract the turn content
                turn_content = line.split(':', 1)[1].strip()
                if turn_content:
                    turns.append(turn_content)
    else:
        # Fallback: split by lines
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Text:') and not line.startswith('Output:'):
                turns.append(line)
    
    return turns

def load_texts_from_json(json_file_path: str, max_texts: int = None) -> List[str]:
    """Load texts from JSON file from the 'answer' field."""
    texts = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                texts.append(data['answer'])
        
        print(f"Loaded {len(texts)} texts from {json_file_path}")
        return texts
        
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def main():
    # Load texts from JSON file
    json_file_path = "/lustre/fsw/portfolios/convai/users/kevinhu/TestData/datasets/LibriSpeech/LibriSpeech/time/train_tarred.all_new_manifest.json/sharded_manifests/manifest_0.json"
    
    # Load texts (limit to first 10 for testing, remove max_texts parameter to process all)
    texts = load_texts_from_json(json_file_path, max_texts=10)
    
    if not texts:
        print("No texts loaded from JSON file. Using sample texts instead.")
        # Fallback to sample texts
        texts = [
            "he said how happy you will be i will do my best said the inn keeper of the pont du gard shutting up his knife well then we will go into paris",
            "swear to me walton that he shall not escape that you will seek him and satisfy my vengeance in his death and do i dare to ask of you to undertake my pilgrimage to endure the hardships that i have undergone no"
        ]
    
    use_timestamps = True  # Set to True since the texts contain timestamp markers like <|4|>
    
    print(f"Processing {len(texts)} texts with batch inference...")
    
    # Try to use Mistral inference first
    model, tokenizer = setup_mistral_inference()
    
    if model is not None and tokenizer is not None:
        print("\n=== Using Mistral Inference ===")
        start_time = time.time()
        results = batch_process_mistral(texts, model, tokenizer, batch_size=2, use_timestamps=use_timestamps)
        end_time = time.time()
        print(f"Mistral inference processing took {end_time - start_time:.2f} seconds")
    else:
        print("\n=== Using Transformers Fallback ===")
        pipe = setup_transformers_fallback()
        start_time = time.time()
        results = batch_process_transformers(texts, pipe, batch_size=2, use_timestamps=use_timestamps)
        end_time = time.time()
        print(f"Transformers processing took {end_time - start_time:.2f} seconds")
    
    # Save results
    save_results(results)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nSummary: {successful}/{len(texts)} texts processed successfully")
    
    # Print results one by one
    print("\n=== Results One by One ===")
    for i, result in enumerate(results):
        print(f"\n{'='*80}")
        print(f"TEXT {i+1}/{len(results)}")
        print(f"{'='*80}")
        
        print(f"INPUT TEXT:")
        print(f"{result['input_text']}")
        print()
        
        if result["success"]:
            print(f"GENERATED OUTPUT:")
            print(f"{result['generated_text']}")
            print()
            
            # Parse turns
            turns = parse_turns_from_output(result['generated_text'])
            print(f"PARSED TURNS ({len(turns)} total):")
            for j, turn in enumerate(turns, 1):
                print(f"Turn {j}: {turn}")
        else:
            print(f"ERROR: {result['error']}")
        
        print(f"{'='*80}")
        print()

if __name__ == "__main__":
    main() 