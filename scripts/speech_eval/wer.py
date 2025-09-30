import json
import argparse
from jiwer import wer, process_words
from whisper_normalizer.english import EnglishTextNormalizer

def parse_json_file(file_path):
    """
    Parses a JSON file that can be either:
    1. JSONL format (each line is a JSON object) - new format
    2. Concatenated JSON objects separated by '}{' - old format
    
    Returns a list of dicts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Check if it's JSONL format (each line is a valid JSON object)
        lines = content.strip().split('\n')
        if len(lines) > 1:
            # Try to parse as JSONL first
            try:
                jsonl_data = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        jsonl_data.append(json.loads(line.strip()))
                return jsonl_data
            except json.JSONDecodeError:
                # Not JSONL, fall back to old format parsing
                pass
        
        # Fall back to old format parsing (concatenated JSON objects)
        # The file is not a valid JSON array, but a concatenation of JSON objects
        # separated by '}{', so we split and fix it
        objects = content.strip().split('}{')
        for i, obj in enumerate(objects):
            if not obj.strip():
                continue
            if not obj.startswith('{'):
                obj = '{' + obj
            if not obj.endswith('}'):
                obj = obj + '}'
            data.append(json.loads(obj))
    
    return data

def get_error_details(ref, hyp):
    """
    Calculate detailed error statistics between reference and hypothesis.
    Returns a tuple of (substitutions, deletions, insertions, total_errors).
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    # Use jiwer's process_words to get detailed statistics
    measures = process_words([ref], [hyp])
    
    # Extract error counts from the measures
    substitutions = measures.substitutions
    deletions = measures.deletions
    insertions = measures.insertions
    total_errors = substitutions + deletions + insertions
    
    return substitutions, deletions, insertions, total_errors

def main():
    parser = argparse.ArgumentParser(description="Compute WER using jiwer for a JSON file with text normalization.")
    parser.add_argument("--json_file", type=str, help="Path to the JSON file.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable text normalization")
    parser.add_argument("--verbose", action="store_true", help="Print individual utterance details")
    args = parser.parse_args()

    data = parse_json_file(args.json_file)

    # Initialize normalizer
    if not args.no_normalize:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = lambda x: x  # Identity function

    refs = []
    hyps = []
    normalized_refs = []
    normalized_hyps = []
    
    for entry in data:
        ref = entry.get("src_text", "")
        hyp = entry.get("pred_src_text", "")

        # Remove any '^' characters from hyp before normalization
        hyp = hyp.replace('^', '')

        if ref.strip() == "" and hyp.strip() == "":
            continue  # skip empty pairs
        
        # Store original texts
        refs.append(ref)
        hyps.append(hyp)
        
        # Store normalized texts
        normalized_ref = normalizer(ref)
        normalized_hyp = normalizer(hyp)
        normalized_refs.append(normalized_ref)
        normalized_hyps.append(normalized_hyp)
       

    if not refs or not hyps:
        print("No valid reference/hypothesis pairs found.")
        return

    # Compute WER on normalized text (weighted average)
    total_wer = 0.0
    total_ref_words = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    
    for idx, (ref, hyp) in enumerate(zip(normalized_refs, normalized_hyps)):
        ref_words = ref.split()
        total_ref_words += len(ref_words)
        pair_wer = wer([ref], [hyp])
        
        # Get detailed error statistics
        subs, dels, ins, total_errors = get_error_details(ref, hyp)
        total_substitutions += subs
        total_deletions += dels
        total_insertions += ins
        
        if args.verbose:
            print(f"[REF]\t{ref}")
            print(f"[HYP]\t{hyp}")
            print(f"WER: {pair_wer:.4f} ({pair_wer*100:.2f}%)")
            print(f"Errors: {subs} substitutions, {dels} deletions, {ins} insertions")
            print(f"Utterance {idx+1}: WER={pair_wer:.4f} ({pair_wer*100:.2f}%) - S:{subs} D:{dels} I:{ins}")
            print("-" * 50)
        
        total_wer += pair_wer * len(ref_words)
    
    # Weighted average WER
    corpus_wer = total_wer / total_ref_words if total_ref_words > 0 else 0.0
    
    # Also compute simple average WER for comparison
    simple_wer = wer(normalized_refs, normalized_hyps)
    
    print(f"\n=== CORPUS SUMMARY ===")
    print(f"Normalized WER (weighted): {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
    print(f"Normalized WER (simple): {simple_wer:.4f} ({simple_wer*100:.2f}%)")
    print(f"Total utterances: {len(refs)}")
    print(f"Total reference words: {total_ref_words}")
    print(f"Total errors: {total_substitutions + total_deletions + total_insertions}")
    print(f"  - Substitutions: {total_substitutions}")
    print(f"  - Deletions: {total_deletions}")
    print(f"  - Insertions: {total_insertions}")
    
    if not args.no_normalize:
        print("Note: Text was normalized using Whisper's EnglishTextNormalizer")

if __name__ == "__main__":
    main()