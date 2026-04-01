"""
Bridge script: convert duplex STT inference output to VoiceAgentBench format and run VAB evaluation.

Takes duplex STT inference metadata JSONs (from validation_logs/metadatas/) and the
original VAB dataset JSON, merges predictions into VAB format, then calls VAB evaluators.

Usage:
    # Evaluate single_tool subset
    python scripts/speech_eval/eval_vab.py \
        --inference_json "path/to/validation_logs/metadatas/single_tool_rank*.json" \
        --vab_json path/to/single_tool_english.json \
        --evaluator single \
        --output_dir path/to/output \
        --use_gt_user_text

    # Evaluate all subsets (uses default VAB data paths)
    python scripts/speech_eval/eval_vab.py \
        --inference_dir path/to/validation_logs/metadatas \
        --evaluator all \
        --output_dir path/to/output \
        --use_gt_user_text
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# VAB subset config
# ---------------------------------------------------------------------------

VAB_SUBSETS = {
    "single_tool": {"evaluator": "single", "query_field": "query"},
    "single_tool_retrieval": {"evaluator": "single", "query_field": "query"},
    "parallel_tool": {"evaluator": "multiple", "query_field": "query"},
    "seqdep_tool": {"evaluator": "dependent", "query_field": "query"},
    "multi_turn": {"evaluator": "multiturn", "query_field": "chat_history"},
    "safety": {"evaluator": "safety", "query_field": "user_request"},
}

# Mapping from inference cut_id prefix to VAB subset name
# VAB cut IDs look like: "single_tool_0", "parallel_tool_15", "safety_3", etc.
SUBSET_ID_PREFIXES = {
    "single_tool_retrieval": "single_tool_retrieval",
    "single_tool": "single_tool",
    "parallel_tool": "parallel_tool",
    "seqdep_tool": "seqdep_tool",
    "multi_turn": "multi_turn",
    "safety": "safety",
}


# ---------------------------------------------------------------------------
# Parse duplex STT inference output
# ---------------------------------------------------------------------------

def load_inference_entries(pattern: str) -> list[dict]:
    """Load inference metadata from JSONL files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")

    entries = []
    seen_ids = set()
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                eid = entry.get("id", "")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    entries.append(entry)

    print(f"Loaded {len(entries)} unique entries from {len(files)} files")
    return entries


def parse_toolcall_from_pred_text(pred_text: str) -> str:
    """Extract the raw text after FC BOS/EOS markers from duplex pred_text.

    The duplex model outputs something like:
      " <$0.72$>  <|FC:12.24|> Let me see what I can find"
    or with actual TOOLCALL tags from the LLM-based eval pipeline.

    For VAB, we need the generated_text that the VAB parser will parse.
    The duplex model doesn't directly output function call syntax —
    we need to go through the LLM-based FC eval to get the actual tool call.
    Here we just return the raw pred_text cleaned of timing markers.
    """
    # Remove timing markers like <|1.23|>, <$1.23$>, <|FC:1.23|>, <$FC:1.23$>
    cleaned = re.sub(r"<[\|$](?:FC:)?[\d.]+[\|$]>", "", pred_text)
    return cleaned.strip()


def extract_user_text_from_pred_src(pred_src_text: str, fc_eos_time: float | None = None) -> str:
    """Extract user text from predicted source text with timestamps."""
    if not pred_src_text:
        return ""
    segments = []
    pattern = r'<\|((?!FC:)[\d.]+)\|>(.*?)<\$((?!FC:)[\d.]+)\$>'
    for m in re.finditer(pattern, pred_src_text, re.DOTALL):
        start_time = float(m.group(1))
        text = m.group(2).strip()
        end_time = float(m.group(3))
        segments.append((start_time, text, end_time))

    if fc_eos_time is not None:
        relevant = [text for _start, text, end in segments if end < fc_eos_time]
    else:
        relevant = [text for _start, text, _end in segments]

    return " ".join(relevant) if relevant else ""


def parse_fc_timestamps(pred_text: str) -> tuple:
    """Parse FC BOS and EOS timestamps from pred_text."""
    bos_match = re.search(r'<\|FC:([\d.]+)\|>', pred_text)
    eos_match = re.search(r'<\$FC:([\d.]+)\$>', pred_text)
    bos = float(bos_match.group(1)) if bos_match else None
    eos = float(eos_match.group(1)) if eos_match else None
    return bos, eos


def parse_tools_from_system_prompt(system_prompt: str) -> list[dict]:
    """Extract tool definitions from system prompt."""
    match = re.search(r"<AVAILABLE_TOOLS>(.*?)</AVAILABLE_TOOLS>", system_prompt, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
    return []


def parse_toolcall_tag(text: str) -> list[dict]:
    """Parse <TOOLCALL>[...]</TOOLCALL> into list of {name: {param: [value]}} dicts.

    Converts from Glaive/NeMo format:
        [{"name": "func", "arguments": {"p": v}}]
    to VAB format:
        [{"func": {"p": [v]}}]
    """
    match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", text, re.DOTALL)
    if not match:
        return []
    try:
        calls = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    result = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        # Wrap scalar values in lists (VAB format)
        vab_args = {}
        for k, v in args.items():
            if isinstance(v, list):
                vab_args[k] = v
            else:
                vab_args[k] = [v]
        result.append({name: vab_args})
    return result


# ---------------------------------------------------------------------------
# LLM-based tool call generation (reuse eval_fc_toolcall logic)
# ---------------------------------------------------------------------------

def build_fc_prompt_for_vab(tools: list[dict], user_text: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> list[dict]:
    """Build a chat prompt for LLM-based FC generation in VAB format."""
    tools_str = json.dumps(tools)
    system_msg = (
        f"You are a helpful assistant with access to the following tools:\n"
        f"{tools_str}\n\n"
        f"If the user's request requires calling a tool, respond ONLY with the function call "
        f"in this exact format: function_name(param1='value1', param2='value2')\n"
        f"If multiple tools are needed, separate them with commas.\n"
        f"If no tool is needed, respond normally.\n"
        f"Do NOT include any explanation, just the function call."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_text},
    ]


def generate_toolcall_with_llm(
    tools: list[dict],
    user_text: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tokenizer=None,
    llm_model=None,
) -> str:
    """Use a local LLM to generate a tool call from user text + tools."""
    if llm_model is None or tokenizer is None:
        return ""

    messages = build_fc_prompt_for_vab(tools, user_text, model_name)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return generated.strip()


# ---------------------------------------------------------------------------
# LLM-based Inverse Text Normalization (ITN)
# ---------------------------------------------------------------------------

_ITN_PROMPT = (
    "Convert the following spoken-form text to proper written form. "
    "Convert spoken numbers to digits, expand abbreviations (e.g. mister → Mr.), "
    "use proper symbols (e.g. percent → %, dollars → $, slash → /), "
    "and fix formatting. Output ONLY the converted text, nothing else.\n\n"
    "Spoken: {text}\n\nWritten:"
)


def apply_itn_fn(text: str, tokenizer, llm_model) -> str:
    """Apply LLM-based ITN to a single text."""
    if not text.strip():
        return text
    import torch
    messages = [
        {"role": "system", "content": "You are a text normalization assistant."},
        {"role": "user", "content": _ITN_PROMPT.format(text=text)},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    inputs = {"input_ids": input_ids.to(llm_model.device)}
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, max_new_tokens=256, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    response = response.split("\n")[0].strip()
    return response if response else text


# ---------------------------------------------------------------------------
# Bridge: merge duplex inference with VAB dataset
# ---------------------------------------------------------------------------

def merge_inference_with_vab(
    inference_entries: list[dict],
    vab_data: list[dict],
    subset_name: str,
    use_gt_user_text: bool = False,
    apply_itn: bool = False,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tokenizer=None,
    llm_model=None,
) -> list[dict]:
    """Merge duplex inference predictions into VAB evaluation format.

    For each VAB entry, finds the matching inference entry and:
    1. Extracts user text (GT or predicted)
    2. Generates tool call via LLM using user text + tools
    3. Parses the LLM output into VAB response format

    Returns list of dicts ready for VAB evaluators.
    """
    # Build lookup: inference entries by ID
    inf_by_id = {e["id"]: e for e in inference_entries}

    merged = []
    missing = 0
    no_fc = 0

    for vab_entry in vab_data:
        vab_id = vab_entry["id"]
        inf_entry = inf_by_id.get(vab_id)

        if inf_entry is None:
            missing += 1
            continue

        # Get user text
        gt_src_text = inf_entry.get("src_text", "")
        pred_src_text_raw = inf_entry.get("pred_src_text", "")
        if use_gt_user_text:
            user_text = gt_src_text
        else:
            fc_bos, fc_eos = parse_fc_timestamps(inf_entry.get("pred_text", ""))
            user_text = extract_user_text_from_pred_src(pred_src_text_raw, fc_eos)

        # Check if FC was detected
        pred_text = inf_entry.get("pred_text", "")
        fc_bos, fc_eos = parse_fc_timestamps(pred_text)
        fc_detected = fc_bos is not None

        # Apply ITN to predicted user text
        user_text_itn = ""
        if apply_itn and user_text and not use_gt_user_text:
            user_text_itn = apply_itn_fn(user_text, tokenizer, llm_model)

        # (debug logging deferred until after comparing pred vs expected)

        # Extract tools from system prompt
        system_prompt = inf_entry.get("system_prompt", "")
        tools = parse_tools_from_system_prompt(system_prompt)

        # Generate tool call via LLM (use ITN text if available)
        llm_input_text = user_text_itn if user_text_itn else user_text
        generated_text = ""
        if fc_detected and llm_input_text:
            generated_text = generate_toolcall_with_llm(
                tools, llm_input_text, model_name, tokenizer, llm_model
            )
        else:
            no_fc += 1
            if not fc_detected:
                print(f"  WARNING: No FC detected in pred_text")
                print(f"  system_prompt: {system_prompt[:500]}")
            if not user_text:
                print(f"  WARNING: Empty user text")

        # Parse LLM output into VAB response format
        # VAB parser expects function_name(param1='value1', ...) format
        # which is exactly what we ask the LLM to produce
        from voice_agent_bench.utils.parser import parse_response
        evaluator_type = VAB_SUBSETS[subset_name]["evaluator"]
        parsed_response = parse_response(generated_text, evaluator_type)

        if evaluator_type in ("single", "multiturn"):
            response = parsed_response  # list of [{name: {params}}] or None
        elif evaluator_type == "multiple":
            response = parsed_response if parsed_response else []
            response = [r for r in response if r is not None]
        else:
            response = None

        expected = vab_entry.get("expected_tool_call", [])

        # Only print debug info when pred doesn't match expected
        if response != expected:
            print(f"\n\033[91m[MISMATCH]\033[0m [{vab_id}] fc_detected={fc_detected}")
            print(f"  GT src_text:   {gt_src_text[:200]}")
            print(f"  pred_src_text: {pred_src_text_raw[:200]}")
            print(f"  user_text:     {user_text[:200]}")
            if user_text_itn:
                print(f"  user_text_itn: {user_text_itn[:200]}")
            if not fc_detected:
                print(f"  WARNING: No FC detected in pred_text")
            if not user_text:
                print(f"  WARNING: Empty user text")
            print(f"  LLM generated: {generated_text[:200]}")
            print(f"  parsed resp:   {response}")
            print(f"  expected:      {expected}")

        # Build merged entry — copy original VAB fields + add predictions
        out = dict(vab_entry)
        out["generated_text"] = generated_text
        out["response"] = response
        out["fc_detected"] = fc_detected
        out["duplex_pred_text"] = pred_text
        out["user_text_used"] = user_text
        out["user_text_itn"] = user_text_itn
        out["gt_src_text"] = gt_src_text

        merged.append(out)

    print(f"Merged {len(merged)} entries ({missing} missing from inference, {no_fc} no FC detected)")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bridge duplex STT inference to VoiceAgentBench evaluation")
    parser.add_argument("--inference_json", type=str, default=None,
                        help="Glob pattern for inference metadata JSONs (e.g., 'path/*_rank*.json')")
    parser.add_argument("--inference_dir", type=str, default=None,
                        help="Directory containing inference metadata JSONs (auto-discovers subsets)")
    parser.add_argument("--vab_json", type=str, default=None,
                        help="Path to VAB dataset JSON for a single subset")
    parser.add_argument("--vab_data_dir", type=str, default=None,
                        help="Base directory for VAB dataset JSONs (auto-discovers subsets)")
    parser.add_argument("--evaluator", type=str, default="single",
                        help="VAB evaluator type: single, multiple, dependent, multiturn, safety, or 'all'")
    parser.add_argument("--subset", type=str, default=None,
                        help="VAB subset name (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write merged JSONs and results")
    parser.add_argument("--use_gt_user_text", action="store_true",
                        help="Use ground truth user text instead of ASR-predicted text")
    parser.add_argument("--apply_itn", action="store_true", default=True,
                        help="Apply LLM-based ITN to predicted user text before tool call generation (default: True)")
    parser.add_argument("--no_itn", action="store_false", dest="apply_itn",
                        help="Disable LLM-based ITN")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM model for generating tool calls from user text")
    parser.add_argument("--vab_repo_dir", type=str,
                        default="/lustre/fsw/portfolios/llmservice/users/kevinhu/projects/VoiceAgentBench",
                        help="Path to cloned VoiceAgentBench repo")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory for downloading VAB data")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only produce merged JSON, skip running VAB evaluators (no OpenAI key needed)")
    args = parser.parse_args()

    # Add VAB repo to path
    vab_bench_dir = os.path.join(args.vab_repo_dir, "voice_agent_bench")
    if vab_bench_dir not in sys.path:
        sys.path.insert(0, vab_bench_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load LLM for tool call generation
    print(f"Loading LLM: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=args.cache_dir
    )

    # Determine which subsets to evaluate
    if args.evaluator == "all":
        subsets_to_eval = list(VAB_SUBSETS.keys())
    elif args.subset:
        subsets_to_eval = [args.subset]
    else:
        # Infer subset from evaluator type
        subsets_to_eval = [k for k, v in VAB_SUBSETS.items() if v["evaluator"] == args.evaluator]

    for subset_name in subsets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {subset_name}")
        print(f"{'='*60}")

        evaluator_type = VAB_SUBSETS[subset_name]["evaluator"]

        # Load inference entries
        if args.inference_json:
            inf_pattern = args.inference_json
        elif args.inference_dir:
            # Try with vab_ prefix first (e.g., vab_single_tool_rank0.json), then without
            inf_pattern = os.path.join(args.inference_dir, f"vab_{subset_name}_rank*.json")
            if not glob.glob(inf_pattern):
                inf_pattern = os.path.join(args.inference_dir, f"{subset_name}_rank*.json")
        else:
            print(f"  No inference data for {subset_name}, skipping")
            continue

        try:
            inference_entries = load_inference_entries(inf_pattern)
        except FileNotFoundError as e:
            print(f"  {e}, skipping")
            continue

        # Load VAB dataset JSON
        if args.vab_json:
            vab_json_path = args.vab_json
        elif args.vab_data_dir:
            vab_json_path = os.path.join(args.vab_data_dir, f"{subset_name}_english.json")
        else:
            # Download from HuggingFace
            from huggingface_hub import hf_hub_download
            json_paths = {
                "single_tool": "data/single_tool_data/english/single_tool_english.json",
                "single_tool_retrieval": "data/single_tool_retrieval_data/english/single_tool_retrieval_english.json",
                "parallel_tool": "data/parallel_tool_data/english/parallel_tool_english.json",
                "seqdep_tool": "data/seqdep_tool_data/english/seqdep_tool_english.json",
                "multi_turn": "data/multi_turn_data/english/multi_turn_english.json",
                "safety": "data/safety_data/english/safety_english.json",
            }
            vab_json_path = hf_hub_download(
                "krutrim-ai-labs/VoiceAgentBench",
                json_paths[subset_name],
                repo_type="dataset",
                cache_dir=args.cache_dir,
            )

        print(f"Loading VAB data from: {vab_json_path}")
        with open(vab_json_path) as f:
            vab_data = json.load(f)
        print(f"Loaded {len(vab_data)} VAB entries")

        # Merge inference with VAB data
        merged = merge_inference_with_vab(
            inference_entries, vab_data, subset_name,
            use_gt_user_text=args.use_gt_user_text,
            apply_itn=args.apply_itn,
            model_name=args.model,
            tokenizer=tokenizer,
            llm_model=llm_model,
        )

        # Save merged results
        suffix = "_gt" if args.use_gt_user_text else "_pred"
        merged_jsonl_path = output_dir / f"{subset_name}{suffix}_responses.jsonl"
        with open(merged_jsonl_path, "w") as f:
            for entry in merged:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved merged JSONL: {merged_jsonl_path}")
        # Also save as .json for VAB evaluate.py compatibility
        merged_path = output_dir / f"{subset_name}{suffix}_responses.json"
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        # Run VAB evaluation
        if not args.skip_eval:
            print(f"Running VAB {evaluator_type} evaluator...")
            try:
                from evaluators import evaluator_mapping
                evaluator = evaluator_mapping[evaluator_type]()
                results = evaluator.evaluate(merged)

                results_path = output_dir / f"{subset_name}{suffix}_results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved results: {results_path}")

                # Print metrics summary (Table 2 style from VAB paper)
                n = len(results)
                summary_path = output_dir / "vab_summary.txt"
                summary_line = ""
                if n > 0 and evaluator_type == "safety":
                    # Safety subset uses single score = Refusal Rate (RR)
                    rr = sum(r.get("score", 0) for r in results) / n * 100
                    summary_line = f"[{subset_name}] (RR) Refusal Rate: {rr:.2f}%  (n={n})"
                elif n > 0 and evaluator_type == "multiple":
                    # Multiple evaluator: scores are per-function inside 'scores' list
                    all_ts, all_tcs, all_pf = [], [], []
                    for r in results:
                        for fn_score in r.get("scores", []):
                            s = fn_score.get("score", [0, 0, 0])
                            all_ts.append(s[0]); all_tcs.append(s[1]); all_pf.append(s[2])
                    total = len(all_ts) if all_ts else 1
                    ts = sum(all_ts) / total * 100
                    tcs = sum(all_tcs) / total * 100
                    pf = sum(all_pf) / total * 100
                    summary_line = (f"[{subset_name}] (TS) Tool Selection: {ts:.2f}%  "
                          f"(TCS) Tool Call Structure: {tcs:.2f}%  "
                          f"(PF) Parameter Filling: {pf:.2f}%  (n={total})")
                elif n > 0:
                    # single / dependent / multiturn: score is [TS, TCS, PF]
                    ts = sum(r.get("score", [0,0,0])[0] for r in results) / n * 100
                    tcs = sum(r.get("score", [0,0,0])[1] for r in results) / n * 100
                    pf = sum(r.get("score", [0,0,0])[2] for r in results) / n * 100
                    summary_line = (f"[{subset_name}] (TS) Tool Selection: {ts:.2f}%  "
                          f"(TCS) Tool Call Structure: {tcs:.2f}%  "
                          f"(PF) Parameter Filling: {pf:.2f}%  (n={n})")
                if summary_line:
                    print(f"\n  {summary_line}")
                    with open(summary_path, "a") as sf:
                        sf.write(summary_line + "\n")
            except Exception as e:
                print(f"  Evaluation failed: {e}")
                print(f"  You can run manually: cd {args.vab_repo_dir}/voice_agent_bench && "
                      f"python evaluate.py --src_file {merged_path} --evaluator {evaluator_type}")
        else:
            print(f"Skipping evaluation (--skip_eval). Run manually:")
            print(f"  cd {args.vab_repo_dir}/voice_agent_bench && "
                  f"python evaluate.py --src_file {merged_path} --evaluator {evaluator_type}")


if __name__ == "__main__":
    main()
