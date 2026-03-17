"""
Evaluate tool call generation using an LLM on FC datasets.

Supports three input formats:
  1. Lhotse shar (cuts.*.jsonl.gz) — Glaive-style FC data
  2. BFCL JSONL — Berkeley Function Calling Leaderboard format
  3. Inference JSON — Duplex STT model validation output (per-rank JSONL)

Usage:
    # BFCL simple set
    python scripts/speech_data_generation/eval_fc_toolcall.py \
        --bfcl_data /path/to/BFCL_v4_simple_python.json \
        --bfcl_answer /path/to/possible_answer/BFCL_v4_simple_python.json

    # Glaive FC test set (Lhotse shar)
    python scripts/speech_data_generation/eval_fc_toolcall.py \
        --shar_path /path/to/shar

    # Duplex STT inference output (glob pattern for per-rank files)
    python scripts/speech_data_generation/eval_fc_toolcall.py \
        --inference_json '/path/to/validation_logs/metadatas/bfcl_v3_rank*.json' \
        --model nvidia/Nemotron-Mini-4B-Instruct

    # Save detailed results
    python scripts/speech_data_generation/eval_fc_toolcall.py \
        --bfcl_data /path/to/data.json --bfcl_answer /path/to/answer.json \
        --output_jsonl /path/to/results.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bfcl_ast_checker import ast_checker as bfcl_ast_checker


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

def infer_bfcl_category(cut_id: str) -> str:
    """Infer BFCL category from cut ID prefix."""
    for prefix in ("parallel_multiple_", "parallel_", "multiple_", "simple_", "irrelevance_"):
        if cut_id.startswith(prefix):
            return prefix.rstrip("_")
    return "simple"  # fallback


# ---------------------------------------------------------------------------
# Shar parsing
# ---------------------------------------------------------------------------

def parse_tools_from_system_prompt(system_text: str) -> list[dict]:
    """Extract tool definitions from <AVAILABLE_TOOLS>[...]</AVAILABLE_TOOLS> text."""
    m = re.search(r"<AVAILABLE_TOOLS>\s*(\[.*?\])\s*</AVAILABLE_TOOLS>", system_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return []
    return []


def parse_toolcall(text: str) -> list[dict]:
    """Extract tool call JSON from <TOOLCALL>[...]</TOOLCALL> text."""
    m = re.search(r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return []
    return []


def extract_fc_examples_from_cut(cut: dict) -> list[dict]:
    """Extract (system_prompt, tools, conversation_history, user_text, gt_toolcall) tuples.

    For multi-turn conversations, each tool call gets the full conversation
    history up to that point as context.
    """
    sups = cut["supervisions"]
    if not sups or sups[0]["speaker"] != "system":
        return []

    system_text = sups[0]["text"]
    tools = parse_tools_from_system_prompt(system_text)
    if not tools:
        return []

    examples = []
    history = []  # conversation history as list of {"role": ..., "content": ...}

    i = 1  # skip system supervision
    while i < len(sups):
        sup = sups[i]
        speaker = sup["speaker"]
        text = sup["text"]
        custom = sup.get("custom", {}) or {}
        function = custom.get("function", "")

        if speaker in ("User", "user") and not function:
            # Regular user turn — this might precede a tool call
            user_text = text

            # Check if next supervision is an Assistant tool call
            if i + 1 < len(sups):
                next_sup = sups[i + 1]
                next_custom = next_sup.get("custom", {}) or {}
                next_function = next_custom.get("function", "")
                if next_sup["speaker"] in ("Assistant", "assistant") and next_function.startswith("<TOOLCALL>"):
                    gt_toolcall_raw = next_function
                    gt_toolcall = parse_toolcall(gt_toolcall_raw)

                    examples.append({
                        "cut_id": cut["id"],
                        "category": infer_bfcl_category(cut["id"]),
                        "system_prompt": system_text,
                        "tools": tools,
                        "history": list(history),  # copy
                        "user_text": user_text,
                        "gt_toolcall_raw": gt_toolcall_raw,
                        "gt_toolcall": gt_toolcall,
                    })

                    # Add to history: user turn, assistant toolcall
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": f"[Tool call: {gt_toolcall_raw}]"})
                    i += 2  # skip user + assistant toolcall

                    # Skip tool response and assistant response, adding them to history
                    while i < len(sups):
                        s = sups[i]
                        sc = s.get("custom", {}) or {}
                        sf = sc.get("function", "")
                        if sf.startswith("<TOOL_RESPONSE>"):
                            history.append({"role": "tool", "content": sf})
                            i += 1
                        elif s["speaker"] in ("Assistant", "assistant") and not sf:
                            history.append({"role": "assistant", "content": s["text"]})
                            i += 1
                            break
                        else:
                            break
                    continue
            # Not followed by a tool call
            category = infer_bfcl_category(cut["id"])
            if category == "irrelevance":
                # Irrelevance: no GT tool call — model should abstain
                examples.append({
                    "cut_id": cut["id"],
                    "category": category,
                    "system_prompt": system_text,
                    "tools": tools,
                    "history": list(history),
                    "user_text": user_text,
                    "gt_toolcall_raw": "",
                    "gt_toolcall": [],
                })
            history.append({"role": "user", "content": user_text})
            i += 1
        else:
            # Skip other supervisions (tool responses, etc. without preceding user turn)
            i += 1

    return examples


def load_shar_cuts(shar_path: str) -> list[dict]:
    """Load all cuts from a Lhotse shar directory."""
    cut_files = sorted(glob.glob(f"{shar_path}/cuts.*.jsonl.gz"))
    if not cut_files:
        raise FileNotFoundError(f"No cuts.*.jsonl.gz files found in {shar_path}")

    cuts = []
    for f in cut_files:
        with gzip.open(f, "rt") as fin:
            for line in fin:
                if line.strip():
                    cuts.append(json.loads(line))
    return cuts


# ---------------------------------------------------------------------------
# BFCL JSONL parsing
# ---------------------------------------------------------------------------

def load_bfcl_jsonl(path: str) -> list[dict]:
    """Load a BFCL JSONL file (one JSON object per line)."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def normalize_bfcl_tool_schema(func: dict) -> dict:
    """Normalise a BFCL tool schema so the AST checker can use it.

    BFCL uses ``"type": "dict"`` in parameter schemas whereas the checker
    (and OpenAI) uses ``"type": "object"``.  We also ensure ``required``
    exists.
    """
    func = dict(func)  # shallow copy
    if "parameters" in func:
        params = dict(func["parameters"])
        if params.get("type") == "dict":
            params["type"] = "object"
        params.setdefault("required", [])
        # Recurse into properties to fix nested "dict" → "object"
        if "properties" in params:
            for prop_name, prop_val in params["properties"].items():
                if isinstance(prop_val, dict) and prop_val.get("type") == "dict":
                    params["properties"][prop_name] = {**prop_val, "type": "object"}
        func["parameters"] = params
    return func


def extract_fc_examples_from_bfcl(
    data_entries: list[dict], answer_entries: list[dict]
) -> list[dict]:
    """Convert BFCL data + possible-answer entries into our unified example format.

    Each example has the same keys as ``extract_fc_examples_from_cut`` output:
        cut_id, system_prompt, tools, history, user_text, gt_toolcall,
        gt_toolcall_raw, bfcl_possible_answer, bfcl_func_descriptions
    """
    answer_by_id = {e["id"]: e for e in answer_entries}

    examples = []
    for entry in data_entries:
        eid = entry["id"]
        answer = answer_by_id.get(eid)
        if answer is None:
            continue

        # question is [[{role, content}, ...]] — take the first conversation
        messages = entry["question"][0]
        user_text = messages[-1]["content"] if messages else ""

        # Build history from earlier messages (if multi-turn)
        history = []
        for msg in messages[:-1]:
            history.append({"role": msg["role"], "content": msg["content"]})

        # Tool schemas — normalise type: dict → object
        raw_tools = entry.get("function", [])
        tools = [normalize_bfcl_tool_schema(f) for f in raw_tools]

        # ground_truth is already in BFCL format: [{func_name: {param: [values]}}]
        bfcl_possible_answer = answer["ground_truth"]

        # Convert BFCL GT to Glaive-style for the simple comparison metric
        # {func: {p: [v1, v2]}} → {"name": func, "arguments": {p: v1}}
        gt_toolcall = []
        for gt_item in bfcl_possible_answer:
            for func_name, params in gt_item.items():
                args = {}
                for k, v_list in params.items():
                    # Pick the first non-empty acceptable value
                    for v in v_list:
                        if v != "":
                            args[k] = v
                            break
                gt_toolcall.append({"name": func_name, "arguments": args})

        examples.append({
            "cut_id": eid,
            "category": infer_bfcl_category(eid),
            "system_prompt": "You are a helpful assistant with access to the following tools.",
            "tools": tools,
            "history": history,
            "user_text": user_text,
            "gt_toolcall_raw": json.dumps(bfcl_possible_answer),
            "gt_toolcall": gt_toolcall,
            # Keep native BFCL fields so we can pass them directly to the checker
            "bfcl_possible_answer": bfcl_possible_answer,
            "bfcl_func_descriptions": tools,
        })

    return examples


# ---------------------------------------------------------------------------
# Inference JSON parsing (duplex STT model output)
# ---------------------------------------------------------------------------

def parse_fc_timestamps(pred_text: str) -> tuple[float | None, float | None]:
    """Extract FC BOS and FC EOS timestamps from pred_text.

    FC BOS: <|FC:X.XX|>
    FC EOS: <$FC:X.XX$>
    Returns (fc_bos_time, fc_eos_time). Either may be None.
    If FC BOS found but FC EOS missing, estimate FC EOS = FC BOS + 1.0.
    """
    fc_bos = None
    fc_eos = None

    m = re.search(r'<\|FC:([\d.]+)\|>', pred_text)
    if m:
        fc_bos = float(m.group(1))

    m = re.search(r'<\$FC:([\d.]+)\$>', pred_text)
    if m:
        fc_eos = float(m.group(1))

    if fc_bos is not None and fc_eos is None:
        fc_eos = fc_bos + 1.0

    return fc_bos, fc_eos


def extract_user_text_from_pred_src(pred_src_text: str, fc_eos_time: float) -> str:
    """Extract the predicted user text from pred_src_text.

    Parse segments delimited by <|X.XX|> (start) and <$X.XX$> (end).
    Return concatenated text from segments whose end timestamp < fc_eos_time.
    Only matches non-FC markers (i.e. excludes <|FC:...|> and <$FC:...$>).
    """
    segments = []
    pattern = r'<\|((?!FC:)[\d.]+)\|>(.*?)<\$((?!FC:)[\d.]+)\$>'
    for m in re.finditer(pattern, pred_src_text, re.DOTALL):
        start_time = float(m.group(1))
        text = m.group(2).strip()
        end_time = float(m.group(3))
        segments.append((start_time, text, end_time))

    relevant = [text for _start, text, end in segments if end < fc_eos_time]
    return " ".join(relevant) if relevant else ""


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


def apply_itn_batch(
    texts: list[str],
    model,
    tokenizer,
    is_nemotron: bool,
    max_new_tokens: int = 256,
) -> list[str]:
    """Apply LLM-based ITN to a batch of texts (one at a time)."""
    n = len(texts)
    results = []
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            print(f"  ITN [{i + 1}/{n}]")
        if not text.strip():
            results.append(text)
            continue
        prompt = _ITN_PROMPT.format(text=text)
        if is_nemotron:
            prompt = f"<extra_id_0>System\nYou are a text normalization assistant.\n<extra_id_1>User\n{prompt}<extra_id_1>Assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            messages = [
                {"role": "system", "content": "You are a text normalization assistant."},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = {"input_ids": input_ids.to(model.device)}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        # Take only the first line to avoid extra commentary
        response = response.split("\n")[0].strip()
        results.append(response if response else text)
    return results


def load_inference_json(path: str) -> list[dict]:
    """Load inference JSONL files.

    path can be a glob pattern (e.g. dir/bfcl_v3_rank*.json)
    or a directory (auto-appends *rank*.json).
    Merges all matching files into a single list, deduplicates by id.
    """
    if os.path.isdir(path):
        pattern = os.path.join(path, "*rank*.json")
    else:
        pattern = path
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")

    seen_ids = set()
    entries = []
    for f in files:
        with open(f) as fin:
            for line in fin:
                if line.strip():
                    entry = json.loads(line)
                    eid = entry.get("id", "")
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        entries.append(entry)

    print(f"Loaded {len(entries)} unique entries from {len(files)} files")
    return entries


def extract_fc_examples_from_inference(entries: list[dict]) -> list[dict]:
    """Convert inference JSON entries into eval examples.

    For each entry:
    1. Parse system_prompt to extract tools
    2. Parse pred_text for FC BOS/EOS timestamps
    3. If FC BOS found: extract user text from pred_src_text
       If not: user_text = "" (marks as no FC prediction)
    4. GT toolcall from tool_call field

    Returns list of dicts with keys:
        cut_id, system_prompt, tools, history, user_text,
        gt_toolcall, gt_toolcall_raw,
        fc_detected (bool), fc_bos_time, fc_eos_time
    """
    examples = []
    for entry in entries:
        system_prompt = entry.get("system_prompt", "")
        tools = parse_tools_from_system_prompt(system_prompt)
        tools = [normalize_bfcl_tool_schema(f) for f in tools]

        pred_text = entry.get("pred_text", "")
        pred_src_text = entry.get("pred_src_text", "")

        fc_bos, fc_eos = parse_fc_timestamps(pred_text)
        fc_detected = fc_bos is not None

        if fc_detected:
            user_text = extract_user_text_from_pred_src(pred_src_text, fc_eos)
        else:
            user_text = ""

        # GT toolcall from tool_call field (list of "<TOOLCALL>...</TOOLCALL>" strings)
        gt_toolcall_raw_list = entry.get("tool_call", [])
        gt_toolcall = []
        gt_toolcall_raw = ""
        for raw in gt_toolcall_raw_list:
            gt_toolcall_raw = raw
            gt_toolcall.extend(parse_toolcall(raw))

        cut_id = entry.get("id", "")
        examples.append({
            "cut_id": cut_id,
            "category": infer_bfcl_category(cut_id),
            "system_prompt": system_prompt,
            "tools": tools,
            "history": [],
            "user_text": user_text,
            "user_text_itn": "",  # populated later by LLM-based ITN
            "src_text": entry.get("src_text", ""),
            "gt_toolcall_raw": gt_toolcall_raw,
            "gt_toolcall": gt_toolcall,
            "fc_detected": fc_detected,
            "fc_bos_time": fc_bos,
            "fc_eos_time": fc_eos,
        })

    return examples


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt_nemotron(
    system_prompt: str, tools: list[dict], history: list[dict], user_text: str
) -> str:
    """Build prompt using Nemotron-Mini-4B-Instruct template."""
    # Strip the raw <AVAILABLE_TOOLS> block from the system prompt — we pass tools via <tool> tags instead
    clean_system = re.sub(r"<AVAILABLE_TOOLS>.*?</AVAILABLE_TOOLS>", "", system_prompt, flags=re.DOTALL).strip()
    if not clean_system:
        clean_system = "You are a helpful assistant. Use the provided tools when the user's request requires them."
    tools_str = "\n".join(f"<tool> {json.dumps(t)} </tool>" for t in tools)
    prompt = (
        f"<extra_id_0>System\n"
        f"{clean_system}\n\n"
        f"{tools_str}\n\n"
    )
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            prompt += f"<extra_id_1>User\n{content}\n"
        elif role == "assistant":
            prompt += f"<extra_id_1>Assistant\n{content}\n"
        elif role == "tool":
            prompt += f"<extra_id_1>Tool\n{content}\n"
    prompt += f"<extra_id_1>User\n{user_text}\n<extra_id_1>Assistant\n"
    return prompt


def build_messages_generic(
    system_prompt: str, tools: list[dict], history: list[dict], user_text: str
) -> list[dict]:
    """Build messages for standard chat template (Llama, Qwen, etc.)."""
    # Strip raw <AVAILABLE_TOOLS> block — we format tools ourselves
    clean_system = re.sub(r"<AVAILABLE_TOOLS>.*?</AVAILABLE_TOOLS>", "", system_prompt, flags=re.DOTALL).strip()
    if not clean_system:
        clean_system = "You are a helpful assistant."
    tools_str = "\n".join(json.dumps(t) for t in tools)
    messages = [
        {
            "role": "system",
            "content": (
                f"{clean_system}\n\n"
                f"You have access to the following tools:\n{tools_str}\n\n"
                f"When the user's request requires a tool, respond with a JSON tool call "
                f"wrapped in <TOOLCALL> tags, e.g.: "
                f'<TOOLCALL>[{{"name": "tool_name", "arguments": {{...}}}}]</TOOLCALL>'
            ),
        },
    ]
    for turn in history:
        role = turn["role"]
        if role == "tool":
            role = "user"  # most models don't have a tool role in chat template
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str = "auto"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    system_prompt: str,
    tools: list[dict],
    history: list[dict],
    user_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    is_nemotron: bool = True,
) -> str:
    if is_nemotron:
        prompt = build_prompt_nemotron(system_prompt, tools, history, user_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    else:
        messages = build_messages_generic(system_prompt, tools, history, user_text)
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = {"input_ids": input_ids.to(model.device)}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)
    return response.strip()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def normalize_value(v):
    """Normalize a value for comparison (lowercase strings, sort lists)."""
    if isinstance(v, str):
        return v.lower().strip()
    if isinstance(v, list):
        normalized = [normalize_value(x) for x in v]
        try:
            return sorted(normalized)
        except TypeError:
            # Can't sort dicts or mixed types — use json serialization for stable order
            return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(v, dict):
        return {k: normalize_value(val) for k, val in sorted(v.items())}
    return v


# ---------------------------------------------------------------------------
# Post-processing normalization for predicted tool calls
# ---------------------------------------------------------------------------

_US_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia",
}

_COUNTRY_NAMES = {
    "usa", "us", "united states", "united states of america",
    "uk", "united kingdom", "england", "great britain",
    "canada", "australia", "france", "germany", "italy", "spain",
    "japan", "china", "india", "brazil", "mexico", "russia",
}

# Common abbreviation ↔ expansion pairs (normalized to the longer form)
_ABBREVIATION_MAP = {
    # Units — both singular and plural map to the same canonical form
    "mi": "miles", "mile": "miles",
    "km": "kilometers", "kilometer": "kilometers",
    "m": "meters", "meter": "meters",
    "ft": "feet", "foot": "feet",
    "lb": "pounds", "lbs": "pounds", "pound": "pounds",
    "kg": "kilograms", "kilogram": "kilograms",
    "oz": "ounces", "ounce": "ounces",
    "g/mol": "grams/mole", "grams/mol": "grams/mole",
    "g": "grams", "gram": "grams",
    # Currency
    "usd": "dollar", "dollars": "dollar",
    "eur": "euro", "euros": "euro",
    "gbp": "pound sterling",
    # Language codes
    "en": "english", "es": "spanish", "fr": "french", "de": "german",
    "it": "italian", "pt": "portuguese", "zh": "chinese", "ja": "japanese",
    "ko": "korean", "ru": "russian", "ar": "arabic", "hi": "hindi",
    # Time
    "sec": "seconds", "second": "seconds",
    "min": "minutes", "minute": "minutes",
    "hr": "hours", "hrs": "hours", "hour": "hours",
}
# Build normalized lookup: all forms map to the same canonical value
_ABBREV_NORMALIZE = {}
for form, canonical in _ABBREVIATION_MAP.items():
    _ABBREV_NORMALIZE[form.lower()] = canonical.lower()
    _ABBREV_NORMALIZE[canonical.lower()] = canonical.lower()


def _strip_location_suffix(s: str) -> str:
    """Strip US state abbreviation/name or country suffix from a location string.

    Examples:
        "Seattle, WA" → "Seattle"
        "Los Angeles, California" → "Los Angeles"
        "Rome, Italy" → "Rome"
        "New York, NY" → "New York"

    Does NOT strip if the result would be empty or if there's no comma.
    """
    if "," not in s:
        return s

    # Split on last comma
    parts = s.rsplit(",", 1)
    if len(parts) != 2:
        return s

    base = parts[0].strip()
    suffix = parts[1].strip()

    if not base:
        return s

    # Check if suffix is a US state abbreviation
    if suffix.upper() in _US_STATE_ABBREVS:
        return base

    # Check if suffix is a US state name
    if suffix.lower() in _US_STATE_NAMES:
        return base

    # Check if suffix is a country name
    if suffix.lower() in _COUNTRY_NAMES:
        return base

    # Check for "City, State Abbreviation." (with period)
    if suffix.rstrip(".").upper() in _US_STATE_ABBREVS:
        return base

    return s


def _strip_leading_article(s: str) -> str:
    """Strip leading 'The' from a string.

    "The British Museum" → "British Museum"
    """
    if s.lower().startswith("the ") and len(s) > 4:
        return s[4:]
    return s


def _try_type_coerce(v):
    """Try to coerce a string value to a number or bool.

    "42" → 42, "3.14" → 3.14, "true" → True, "false" → False
    """
    if not isinstance(v, str):
        return v

    s = v.strip()

    # Bool
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # Integer
    try:
        iv = int(s)
        if str(iv) == s:  # exact roundtrip (no leading zeros, etc.)
            return iv
    except (ValueError, OverflowError):
        pass

    # Float
    try:
        fv = float(s)
        return fv
    except (ValueError, OverflowError):
        pass

    return v


def _normalize_abbreviation(s: str) -> str:
    """Normalize common abbreviations and singular/plural to a canonical form."""
    key = s.lower().strip()
    if key in _ABBREV_NORMALIZE:
        return _ABBREV_NORMALIZE[key]
    return s


def _normalize_arg_value(v):
    """Recursively normalize a single argument value."""
    if isinstance(v, str):
        v = _strip_location_suffix(v)
        v = _strip_leading_article(v)
        v = _try_type_coerce(v)
        if isinstance(v, str):
            v = _normalize_abbreviation(v)
    elif isinstance(v, list):
        normalized = [_normalize_arg_value(x) for x in v]
        # Sort for order-insensitive comparison
        try:
            normalized.sort(key=lambda x: json.dumps(x, sort_keys=True, default=str))
        except TypeError:
            pass
        v = normalized
    elif isinstance(v, dict):
        v = {k2: _normalize_arg_value(v2) for k2, v2 in v.items()}
    return v


def normalize_toolcall_for_eval(
    calls: list[dict],
    tool_schemas: list[dict] | None = None,
) -> list[dict]:
    """Post-process tool calls to fix trivial formatting differences.

    Normalizations applied:
      1. Strip US state/country suffixes from location-like strings
      2. Strip leading articles ("The British Museum" → "British Museum")
      3. Type coercion: string "42" → 42, "true" → True
      4. Sort list arguments (order-insensitive comparison)
      5. Normalize abbreviations: "en" → "english", "mi" → "miles"
      6. Simple singular/plural normalization
      7. Remove optional params with default-like values (0, 1, false, "all", "")

    Args:
        calls: List of tool call dicts [{"name": ..., "arguments": {...}}].
        tool_schemas: Optional tool schemas for identifying optional params.

    Returns:
        New list of normalized tool call dicts (originals are not modified).
    """
    if not calls:
        return calls

    # Build lookup: func_name → set of required params
    required_by_func = {}
    if tool_schemas:
        for t in tool_schemas:
            t_inner = t.get("function", t) if "function" in t else t
            name = t_inner.get("name", "")
            req = set(t_inner.get("parameters", {}).get("required", []))
            required_by_func[name] = req

    result = []
    for call in calls:
        new_call = {"name": call.get("name", "")}
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            new_call["arguments"] = args
            result.append(new_call)
            continue

        new_args = {}
        required = required_by_func.get(call.get("name", ""), set())
        for k, v in args.items():
            nv = _normalize_arg_value(v)

            # Remove optional params with default-like values
            if k not in required and tool_schemas:
                if nv in (0, 1, False, "", "all", "none", None):
                    continue
                if isinstance(nv, str) and nv.lower() in ("all", "none", "default", "any"):
                    continue

            new_args[k] = nv

        new_call["arguments"] = new_args
        result.append(new_call)
    return result


def _compare_single_call(p: dict, g: dict) -> tuple[bool, bool]:
    """Compare a single predicted call against a single ground-truth call.

    Returns (name_match, args_match).
    """
    p_name = p.get("name", "").lower().strip()
    g_name = g.get("name", "").lower().strip()
    name_match = p_name == g_name

    p_args = normalize_value(p.get("arguments", {}))
    g_args = normalize_value(g.get("arguments", {}))
    args_match = p_args == g_args

    return name_match, args_match


def compare_toolcall(pred: list[dict], gt: list[dict], category: str = "simple") -> dict:
    """Compare predicted vs ground truth tool calls.

    For simple/multiple: compare pred[0] vs gt[0].
    For parallel/parallel_multiple: unordered matching by function name.
    For irrelevance: both should be empty (model abstains).

    Returns dict with:
        - name_match: bool (all function names correct)
        - args_match: bool (all arguments match)
        - exact_match: bool (both name and args match for all calls)
        - pred_valid_json: bool (predicted output was valid JSON)
        - num_pred: int
        - num_gt: int
        - details: str (human-readable comparison)
    """
    base = {"num_pred": len(pred), "num_gt": len(gt)}

    # Both empty → match (irrelevance correctly abstained, or both have no calls)
    if not pred and not gt:
        return {**base, "name_match": True, "args_match": True, "exact_match": True,
                "pred_valid_json": True, "details": "both empty"}

    if not pred:
        return {**base, "name_match": False, "args_match": False, "exact_match": False,
                "pred_valid_json": False, "details": "pred empty, gt has calls"}

    if not gt:
        return {**base, "name_match": False, "args_match": False, "exact_match": False,
                "pred_valid_json": True, "details": "gt empty, pred has calls"}

    # Count mismatch
    if len(pred) != len(gt):
        return {**base, "name_match": False, "args_match": False, "exact_match": False,
                "pred_valid_json": True,
                "details": f"count mismatch: pred={len(pred)} gt={len(gt)}"}

    if "parallel" in category:
        # Unordered matching: match by function name, then compare args
        gt_remaining = list(range(len(gt)))
        all_name_match = True
        all_args_match = True
        details = []
        for p in pred:
            p_name = p.get("name", "").lower().strip()
            matched = False
            for j in gt_remaining:
                g_name = gt[j].get("name", "").lower().strip()
                if p_name == g_name:
                    nm, am = _compare_single_call(p, gt[j])
                    if not am:
                        all_args_match = False
                        details.append(f"args mismatch for {p_name}")
                    gt_remaining.remove(j)
                    matched = True
                    break
            if not matched:
                all_name_match = False
                details.append(f"no GT match for pred name '{p_name}'")

        if gt_remaining:
            all_name_match = False
            unmatched = [gt[j].get("name", "?") for j in gt_remaining]
            details.append(f"unmatched GT names: {unmatched}")

        exact_match = all_name_match and all_args_match
        return {**base, "name_match": all_name_match, "args_match": all_args_match,
                "exact_match": exact_match, "pred_valid_json": True,
                "details": "; ".join(details) if details else "exact match"}
    else:
        # simple / multiple: compare pred[0] vs gt[0]
        nm, am = _compare_single_call(pred[0], gt[0])
        exact_match = nm and am
        details = []
        if not nm:
            details.append(f"name: pred='{pred[0].get('name','')}' gt='{gt[0].get('name','')}'")
        if not am:
            p_args = normalize_value(pred[0].get("arguments", {}))
            g_args = normalize_value(gt[0].get("arguments", {}))
            details.append(f"args: pred={json.dumps(p_args)} gt={json.dumps(g_args)}")
        return {**base, "name_match": nm, "args_match": am, "exact_match": exact_match,
                "pred_valid_json": True,
                "details": "; ".join(details) if details else "exact match"}


# ---------------------------------------------------------------------------
# BFCL AST-based scoring (adapted from gorilla/BFCL v3)
# ---------------------------------------------------------------------------

def unwrap_tool_descriptions(tools: list[dict]) -> list[dict]:
    """Unwrap OpenAI-format tool dicts to bare function schemas.

    Glaive tools may be ``{"type": "function", "function": {...}}`` or bare
    ``{"name": ..., "parameters": ...}``.  The BFCL checker expects the bare form.
    """
    result = []
    for t in tools:
        if "function" in t and "type" in t:
            result.append(t["function"])
        else:
            result.append(t)
    return result


def glaive_to_bfcl_output(calls: list[dict]) -> list[dict]:
    """Convert Glaive call format to BFCL model_output format.

    Glaive:  ``[{"name": "func", "arguments": {"p": v}}]``
    BFCL:    ``[{"func": {"p": v}}]``
    """
    return [{c["name"]: c.get("arguments", {})} for c in calls]


def glaive_to_bfcl_possible_answer(
    calls: list[dict], tools: list[dict] | None = None
) -> list[dict]:
    """Convert Glaive GT to BFCL possible_answer format.

    Glaive:  ``[{"name": "func", "arguments": {"p": v}}]``
    BFCL:    ``[{"func": {"p": [v]}}]``

    Each param value is wrapped in a list because BFCL supports multiple
    acceptable values per parameter.  For optional parameters (not in the
    tool schema's ``required`` list), ``""`` is appended so the BFCL checker
    knows the parameter may be omitted.
    """
    # Build lookup: func_name → set of required params
    required_by_func = {}
    if tools:
        for t in tools:
            t_inner = t.get("function", t) if "function" in t else t
            name = t_inner.get("name", "")
            req = set(t_inner.get("parameters", {}).get("required", []))
            required_by_func[name] = req

    result = []
    for c in calls:
        func_name = c["name"]
        args = c.get("arguments", {})
        required = required_by_func.get(func_name, set())
        wrapped = {}
        for k, v in args.items():
            if k in required:
                wrapped[k] = [v]
            else:
                # Optional param: add "" sentinel so checker allows omission
                wrapped[k] = [v, ""]
        result.append({func_name: wrapped})
    return result


def bfcl_compare_toolcall(
    pred: list[dict],
    gt: list[dict],
    tools: list[dict],
    category: str = "simple",
    bfcl_possible_answer: list[dict] | None = None,
    bfcl_func_descriptions: list[dict] | None = None,
) -> dict:
    """Compare tool calls using BFCL AST-based checker.

    Args:
        pred: Predicted tool calls in Glaive format.
        gt: Ground truth tool calls in Glaive format.
        tools: Tool schemas.
        category: BFCL category (simple, multiple, parallel, parallel_multiple, irrelevance).
        bfcl_possible_answer: If provided, use this directly as the BFCL
            possible_answer (native format from BFCL dataset).
        bfcl_func_descriptions: If provided, use this directly as the BFCL
            func_descriptions.

    Returns dict with:
        - bfcl_valid: bool (passes BFCL AST check)
        - bfcl_error: list[str]
        - bfcl_error_type: str
    """
    # Irrelevance: model should abstain (produce no tool calls)
    if category == "irrelevance":
        func_descriptions = bfcl_func_descriptions if bfcl_func_descriptions is not None else unwrap_tool_descriptions(tools)
        model_output = glaive_to_bfcl_output(pred) if pred else []
        try:
            result = bfcl_ast_checker(func_descriptions, model_output, [], category)
        except Exception as e:
            return {"bfcl_valid": False, "bfcl_error": [f"Checker exception: {e}"],
                    "bfcl_error_type": "checker_exception"}
        return {"bfcl_valid": result.get("valid", False),
                "bfcl_error": result.get("error", []),
                "bfcl_error_type": result.get("error_type", "")}

    if not pred and not gt:
        return {"bfcl_valid": True, "bfcl_error": [], "bfcl_error_type": ""}
    if not pred:
        return {"bfcl_valid": False, "bfcl_error": ["No prediction"], "bfcl_error_type": "no_prediction"}
    if not gt:
        return {"bfcl_valid": False, "bfcl_error": ["No ground truth"], "bfcl_error_type": "no_ground_truth"}

    func_descriptions = bfcl_func_descriptions if bfcl_func_descriptions is not None else unwrap_tool_descriptions(tools)
    model_output = glaive_to_bfcl_output(pred)
    possible_answer = bfcl_possible_answer if bfcl_possible_answer is not None else glaive_to_bfcl_possible_answer(gt, tools)

    # Map category for the checker: "parallel_multiple" → "parallel" (checker uses "parallel" for both)
    checker_category = "parallel" if "parallel" in category else category

    try:
        result = bfcl_ast_checker(func_descriptions, model_output, possible_answer, checker_category)
    except Exception as e:
        return {
            "bfcl_valid": False,
            "bfcl_error": [f"Checker exception: {e}"],
            "bfcl_error_type": "checker_exception",
        }

    return {
        "bfcl_valid": result.get("valid", False),
        "bfcl_error": result.get("error", []),
        "bfcl_error_type": result.get("error_type", ""),
    }


def extract_toolcall_from_response(response: str) -> list[dict]:
    """Try to extract tool call JSON from the LLM response.

    Handles multiple formats:
        - <TOOLCALL>[...]</TOOLCALL>
        - <toolcall>[...]</toolcall>
        - Raw JSON [{"name": ..., "arguments": ...}]
        - Raw JSON {"name": ..., "arguments": ...}
    """
    # Try <TOOLCALL> tags with JSON array inside (case-insensitive)
    m = re.search(r"<(?:TOOLCALL|toolcall)>\s*(\[.*?\])\s*</(?:TOOLCALL|toolcall)>", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try multiple <toolcall>{...}</toolcall> tags (one object each)
    # Model may output separate tags for each call in parallel scenarios
    multi_matches = re.findall(
        r"<(?:TOOLCALL|toolcall)>\s*(\{.*?\})\s*</(?:TOOLCALL|toolcall)>", response, re.DOTALL
    )
    if multi_matches:
        calls = []
        for raw in multi_matches:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
        if calls:
            return calls

    # Try raw JSON array — find outermost [...] containing "name"
    # Use re.DOTALL so . matches newlines; avoid [^]] which breaks on multi-object arrays
    m = re.search(r'\[(\s*\{.*\})+\s*\]', response, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            # Validate it looks like tool calls
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "name" in parsed[0]:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try single JSON object
    m = re.search(r'\{[^{}]*"name"\s*:[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return [obj]
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval_inference_json(model, tokenizer, is_nemotron: bool, args):
    """Run evaluation on duplex STT inference JSON output.

    Computes FC detection precision/recall/F1 and tool call accuracy
    (on the true-positive subset where FC was both predicted and expected).
    """
    print(f"Loading inference JSON from: {args.inference_json}")
    if args.use_gt_user_text:
        print("Mode: GROUND-TRUTH user text (upper bound)")
    elif args.apply_itn:
        print("Mode: LLM-based ITN on predicted user text")
    else:
        print("Mode: raw predicted user text (no ITN)")
    if getattr(args, 'normalize', False):
        print("Post-processing normalization: ENABLED")
    entries = load_inference_json(args.inference_json)
    all_examples = extract_fc_examples_from_inference(entries)
    print(f"Extracted {len(all_examples)} examples")

    if args.max_examples and args.max_examples < len(all_examples):
        all_examples = all_examples[:args.max_examples]
        print(f"Limiting to {args.max_examples} examples")

    # Auto-suffix output filename based on mode (only if suffix not already present)
    if args.output_jsonl:
        p = Path(args.output_jsonl)
        if args.use_gt_user_text:
            suffix = "_gt"
        elif args.apply_itn:
            suffix = "_itn"
        else:
            suffix = "_raw"
        # Only add suffix if it's not already in the filename
        if not p.stem.endswith(suffix):
            args.output_jsonl = str(p.with_stem(p.stem + suffix))
        print(f"Output file: {args.output_jsonl}")

    # Run LLM-based ITN on predicted user texts
    if args.apply_itn and not args.use_gt_user_text:
        texts_to_itn = [ex["user_text"] for ex in all_examples if ex["fc_detected"]]
        if texts_to_itn:
            print(f"Running LLM-based ITN on {len(texts_to_itn)} predicted user texts...")
            itn_results = apply_itn_batch(texts_to_itn, model, tokenizer, is_nemotron)
            itn_iter = iter(itn_results)
            for ex in all_examples:
                if ex["fc_detected"]:
                    ex["user_text_itn"] = next(itn_iter)
            print("ITN complete")

    # FC detection counters
    fc_gt_positive = 0
    fc_pred_positive = 0
    fc_true_positive = 0

    # Tool call metrics (computed on TP set only)
    metrics = {
        "total": 0,
        "name_correct": 0,
        "args_correct": 0,
        "exact_match": 0,
        "valid_json": 0,
        "generation_failed": 0,
        "bfcl_valid": 0,
    }

    output_file = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w")

    for idx, ex in enumerate(all_examples):
        gt_has_fc = len(ex["gt_toolcall"]) > 0
        pred_has_fc = ex["fc_detected"]

        if gt_has_fc:
            fc_gt_positive += 1
        if pred_has_fc:
            fc_pred_positive += 1

        is_tp = gt_has_fc and pred_has_fc
        if is_tp:
            fc_true_positive += 1

        result = {
            "idx": idx,
            "cut_id": ex["cut_id"],
            "user_text": ex["user_text"],
            "user_text_itn": ex["user_text_itn"],
            "src_text": ex["src_text"],
            "fc_detected": ex["fc_detected"],
            "fc_bos_time": ex["fc_bos_time"],
            "fc_eos_time": ex["fc_eos_time"],
            "gt_has_fc": gt_has_fc,
        }

        if pred_has_fc:
            # Run LLM generation for entries where model predicted FC
            if args.use_gt_user_text:
                llm_user_text = ex["src_text"]
            elif args.apply_itn:
                llm_user_text = ex["user_text_itn"]
            else:
                llm_user_text = ex["user_text"]
            response = generate(
                model, tokenizer,
                system_prompt=ex["system_prompt"],
                tools=ex["tools"],
                history=ex["history"],
                user_text=llm_user_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                is_nemotron=is_nemotron,
            )

            pred_toolcall = extract_toolcall_from_response(response)

            # Apply post-processing normalization to pred only
            if getattr(args, 'normalize', False):
                pred_toolcall = normalize_toolcall_for_eval(pred_toolcall, ex["tools"])

            result["raw_response"] = response
            result["pred_toolcall"] = pred_toolcall

            if is_tp:
                # Compute tool call accuracy on true-positive entries
                metrics["total"] += 1
                category = ex.get("category", "simple")
                comparison = compare_toolcall(pred_toolcall, ex["gt_toolcall"], category=category)
                bfcl_result = bfcl_compare_toolcall(
                    pred_toolcall, ex["gt_toolcall"], ex["tools"],
                    category=category,
                )

                result.update(comparison)
                result.update(bfcl_result)
                result["gt_toolcall"] = ex["gt_toolcall"]

                if comparison["pred_valid_json"]:
                    metrics["valid_json"] += 1
                if comparison["name_match"]:
                    metrics["name_correct"] += 1
                if comparison["args_match"]:
                    metrics["args_correct"] += 1
                if comparison["exact_match"]:
                    metrics["exact_match"] += 1
                if not pred_toolcall:
                    metrics["generation_failed"] += 1
                if bfcl_result["bfcl_valid"]:
                    metrics["bfcl_valid"] += 1

                # Print details for mismatches
                if not bfcl_result["bfcl_valid"]:
                    detail = comparison['details'] if not comparison["exact_match"] else ""
                    bfcl_err = bfcl_result['bfcl_error_type']
                    print(f"    MISMATCH [{idx}] id={ex['cut_id']}: {detail}  (bfcl: {bfcl_err})")
                    print(f"      user (pred): {ex['user_text']}")
                    print(f"      user (itn):  {ex['user_text_itn']}")
                    print(f"      user (gt):   {ex['src_text']}")
                    tool_names = [t.get("name", "?") for t in ex.get("tools", [])]
                    print(f"      tools: {tool_names}")
                    print(f"      \033[91mpred: {json.dumps(pred_toolcall, ensure_ascii=False)}\033[0m")
                    print(f"      \033[92mgt:   {json.dumps(ex['gt_toolcall'], ensure_ascii=False)}\033[0m")

        if output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Progress logging
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(all_examples) - 1:
            prec = fc_true_positive / fc_pred_positive if fc_pred_positive else 0
            rec = fc_true_positive / fc_gt_positive if fc_gt_positive else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            n = metrics["total"]
            tc_str = ""
            if n > 0:
                tc_str = (
                    f"  name={metrics['name_correct']/n:.1%}"
                    f" args={metrics['args_correct']/n:.1%}"
                    f" exact={metrics['exact_match']/n:.1%}"
                    f" bfcl={metrics['bfcl_valid']/n:.1%}"
                )
            print(
                f"  [{idx + 1}/{len(all_examples)}] "
                f"FC: P={prec:.1%} R={rec:.1%} F1={f1:.1%}"
                f"{tc_str}"
            )

    if output_file:
        output_file.close()

    # Final report
    total = len(all_examples)
    prec = fc_true_positive / fc_pred_positive if fc_pred_positive else 0
    rec = fc_true_positive / fc_gt_positive if fc_gt_positive else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Inference JSON)")
    print("=" * 80)
    print(f"Total entries:             {total}")
    print(f"\n--- FC Detection ---")
    print(f"GT positive (has FC):      {fc_gt_positive}")
    print(f"Pred positive (FC BOS):    {fc_pred_positive}")
    print(f"True positive:             {fc_true_positive}")
    print(f"Precision:                 {prec:.1%}")
    print(f"Recall:                    {rec:.1%}")
    print(f"F1:                        {f1:.1%}")

    n = metrics["total"]
    if n > 0:
        print(f"\n--- Tool Call Accuracy (on {n} TP entries) ---")
        print(f"Function name accuracy:    {metrics['name_correct']}/{n} = {metrics['name_correct']/n:.1%}")
        print(f"Arguments accuracy:        {metrics['args_correct']}/{n} = {metrics['args_correct']/n:.1%}")
        print(f"Exact match (name+args):   {metrics['exact_match']}/{n} = {metrics['exact_match']/n:.1%}")
        print(f"BFCL AST accuracy:         {metrics['bfcl_valid']}/{n} = {metrics['bfcl_valid']/n:.1%}")
        print(f"Valid JSON output:         {metrics['valid_json']}/{n} = {metrics['valid_json']/n:.1%}")
        print(f"Failed to generate:        {metrics['generation_failed']}/{n} = {metrics['generation_failed']/n:.1%}")
    print("=" * 80)

    # Save metrics to JSON file
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        metrics_json_path = output_path.parent / f"{output_path.stem}_metrics.jsonl"
        
        metrics_dict = {
            "total_entries": total,
            "fc_detection": {
                "gt_positive": fc_gt_positive,
                "pred_positive": fc_pred_positive,
                "true_positive": fc_true_positive,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            },
            "tool_call_accuracy": {
                "total": n,
                "name_correct": metrics['name_correct'],
                "args_correct": metrics['args_correct'],
                "exact_match": metrics['exact_match'],
                "bfcl_valid": metrics['bfcl_valid'],
                "valid_json": metrics['valid_json'],
                "generation_failed": metrics['generation_failed'],
            },
        }
        
        # Add percentages for tool call accuracy
        if n > 0:
            metrics_dict["tool_call_accuracy"]["name_correct_pct"] = metrics['name_correct'] / n
            metrics_dict["tool_call_accuracy"]["args_correct_pct"] = metrics['args_correct'] / n
            metrics_dict["tool_call_accuracy"]["exact_match_pct"] = metrics['exact_match'] / n
            metrics_dict["tool_call_accuracy"]["bfcl_valid_pct"] = metrics['bfcl_valid'] / n
            metrics_dict["tool_call_accuracy"]["valid_json_pct"] = metrics['valid_json'] / n
            metrics_dict["tool_call_accuracy"]["generation_failed_pct"] = metrics['generation_failed'] / n
        
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output_jsonl}")
        print(f"Metrics saved to: {metrics_json_path}")


def _new_metrics_bucket():
    return {"total": 0, "name_correct": 0, "args_correct": 0,
            "exact_match": 0, "valid_json": 0, "generation_failed": 0, "bfcl_valid": 0}


def _update_metrics(bucket, comparison, bfcl_result, pred_toolcall):
    bucket["total"] += 1
    if comparison["pred_valid_json"]:
        bucket["valid_json"] += 1
    if comparison["name_match"]:
        bucket["name_correct"] += 1
    if comparison["args_match"]:
        bucket["args_correct"] += 1
    if comparison["exact_match"]:
        bucket["exact_match"] += 1
    if not pred_toolcall:
        bucket["generation_failed"] += 1
    if bfcl_result["bfcl_valid"]:
        bucket["bfcl_valid"] += 1


def _print_metrics(label, m):
    n = m["total"]
    if n == 0:
        print(f"  {label:25s} (no examples)")
        return
    print(f"  {label:25s} exact={m['exact_match']/n:5.1%}  bfcl={m['bfcl_valid']/n:5.1%}  "
          f"name={m['name_correct']/n:5.1%}  args={m['args_correct']/n:5.1%}  "
          f"json={m['valid_json']/n:5.1%}  (n={n})")


def run_eval(model, tokenizer, is_nemotron: bool, args):
    """Run evaluation on a Lhotse shar or BFCL dataset."""
    if getattr(args, 'inference_json', None):
        return run_eval_inference_json(model, tokenizer, is_nemotron, args)

    if args.bfcl_data:
        # BFCL JSONL format
        print(f"Loading BFCL data from: {args.bfcl_data}")
        print(f"Loading BFCL answers from: {args.bfcl_answer}")
        data_entries = load_bfcl_jsonl(args.bfcl_data)
        answer_entries = load_bfcl_jsonl(args.bfcl_answer)
        print(f"Loaded {len(data_entries)} data entries, {len(answer_entries)} answer entries")
        all_examples = extract_fc_examples_from_bfcl(data_entries, answer_entries)
    else:
        # Lhotse shar format
        print(f"Loading shar from: {args.shar_path}")
        cuts = load_shar_cuts(args.shar_path)
        print(f"Loaded {len(cuts)} cuts")
        all_examples = []
        for cut in cuts:
            all_examples.extend(extract_fc_examples_from_cut(cut))

    print(f"Extracted {len(all_examples)} tool call examples")

    # Print category distribution
    cat_counts = collections.Counter(ex.get("category", "simple") for ex in all_examples)
    for cat in sorted(cat_counts):
        print(f"  category '{cat}': {cat_counts[cat]} examples")

    if args.max_examples and args.max_examples < len(all_examples):
        all_examples = all_examples[:args.max_examples]
        print(f"Limiting to {args.max_examples} examples")

    # Run generation and evaluation
    do_normalize = getattr(args, 'normalize', False)
    if do_normalize:
        print("Post-processing normalization: ENABLED")

    results = []
    metrics = _new_metrics_bucket()
    per_category = collections.defaultdict(_new_metrics_bucket)

    output_file = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w")

    for idx, ex in enumerate(all_examples):
        category = ex.get("category", "simple")

        response = generate(
            model, tokenizer,
            system_prompt=ex["system_prompt"],
            tools=ex["tools"],
            history=ex["history"],
            user_text=ex["user_text"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            is_nemotron=is_nemotron,
        )

        pred_toolcall = extract_toolcall_from_response(response)

        # Apply post-processing normalization to pred only (GT is ground truth, never modified)
        if do_normalize:
            pred_toolcall = normalize_toolcall_for_eval(pred_toolcall, ex["tools"])

        comparison = compare_toolcall(pred_toolcall, ex["gt_toolcall"], category=category)

        # BFCL AST-based scoring
        bfcl_result = bfcl_compare_toolcall(
            pred_toolcall, ex["gt_toolcall"], ex["tools"],
            category=category,
            bfcl_possible_answer=ex.get("bfcl_possible_answer"),
            bfcl_func_descriptions=ex.get("bfcl_func_descriptions"),
        )

        _update_metrics(metrics, comparison, bfcl_result, pred_toolcall)
        _update_metrics(per_category[category], comparison, bfcl_result, pred_toolcall)

        result = {
            "idx": idx,
            "cut_id": ex["cut_id"],
            "category": category,
            "user_text": ex["user_text"],
            "history_len": len(ex["history"]),
            "gt_toolcall": ex["gt_toolcall"],
            "pred_toolcall": pred_toolcall,
            "raw_response": response,
            **comparison,
            **bfcl_result,
        }
        results.append(result)

        if output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Progress logging
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(all_examples) - 1:
            n = metrics["total"]
            print(
                f"  [{idx + 1}/{len(all_examples)}] "
                f"name_acc={metrics['name_correct']/n:.1%} "
                f"args_acc={metrics['args_correct']/n:.1%} "
                f"exact={metrics['exact_match']/n:.1%} "
                f"bfcl={metrics['bfcl_valid']/n:.1%} "
                f"json_valid={metrics['valid_json']/n:.1%}"
            )

        # Print details for mismatches
        if not bfcl_result["bfcl_valid"]:
            detail = comparison['details'] if not comparison["exact_match"] else ""
            bfcl_err = bfcl_result['bfcl_error_type']
            print(f"    MISMATCH [{idx}] id={ex['cut_id']} cat={category}: {detail}  (bfcl: {bfcl_err})")
            print(f"      user: {ex['user_text']}")
            # Print tools from system prompt in readable form
            tools = ex.get("tools", [])
            tool_names = [t.get("name", t.get("function", {}).get("name", "?")) for t in tools]
            print(f"      tools: {tool_names}")
            for t in tools:
                t_unwrapped = t.get("function", t) if "function" in t else t
                params = t_unwrapped.get("parameters", {}).get("properties", {})
                required = t_unwrapped.get("parameters", {}).get("required", [])
                param_strs = []
                for p, info in params.items():
                    req = "*" if p in required else ""
                    param_strs.append(f"{p}{req}:{info.get('type', '?')}")
                print(f"        {t_unwrapped.get('name', '?')}({', '.join(param_strs)})")
            print(f"      \033[91mpred toolcall request: {json.dumps(pred_toolcall, ensure_ascii=False)}\033[0m")
            print(f"      \033[92mgt toolcall request:   {json.dumps(ex['gt_toolcall'], ensure_ascii=False)}\033[0m")
            print(f"      raw model output: {response[:200]}...")

    if output_file:
        output_file.close()

    # Final metrics
    n = metrics["total"]
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total tool call examples:  {n}")
    if n > 0:
        print(f"Function name accuracy:    {metrics['name_correct']}/{n} = {metrics['name_correct']/n:.1%}")
        print(f"Arguments accuracy:        {metrics['args_correct']}/{n} = {metrics['args_correct']/n:.1%}")
        print(f"Exact match (name+args):   {metrics['exact_match']}/{n} = {metrics['exact_match']/n:.1%}")
        print(f"BFCL AST accuracy:         {metrics['bfcl_valid']}/{n} = {metrics['bfcl_valid']/n:.1%}")
        print(f"Valid JSON output:         {metrics['valid_json']}/{n} = {metrics['valid_json']/n:.1%}")
        print(f"Failed to generate:        {metrics['generation_failed']}/{n} = {metrics['generation_failed']/n:.1%}")

    # Per-category breakdown
    if len(per_category) > 1:
        print(f"\n--- Per-Category Results ---")
        for cat in sorted(per_category):
            _print_metrics(f"{cat}:", per_category[cat])

    print("=" * 80)
    print()
    print("Note: 'Exact match' uses normalized string comparison.")
    print("      'BFCL AST' uses BFCL v3 standardized comparison (case/punctuation tolerant).")
    print("      For irrelevance: exact/bfcl = abstention accuracy (model correctly produced no calls).")
    print("=" * 80)

    # Save metrics to JSON file
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        metrics_json_path = output_path.parent / f"{output_path.stem}_metrics.jsonl"
        
        metrics_dict = {
            "total": n,
            "name_correct": metrics['name_correct'],
            "args_correct": metrics['args_correct'],
            "exact_match": metrics['exact_match'],
            "bfcl_valid": metrics['bfcl_valid'],
            "valid_json": metrics['valid_json'],
            "generation_failed": metrics['generation_failed'],
        }
        
        # Add percentages
        if n > 0:
            metrics_dict["name_correct_pct"] = metrics['name_correct'] / n
            metrics_dict["args_correct_pct"] = metrics['args_correct'] / n
            metrics_dict["exact_match_pct"] = metrics['exact_match'] / n
            metrics_dict["bfcl_valid_pct"] = metrics['bfcl_valid'] / n
            metrics_dict["valid_json_pct"] = metrics['valid_json'] / n
            metrics_dict["generation_failed_pct"] = metrics['generation_failed'] / n
        
        # Add per-category metrics
        if len(per_category) > 1:
            metrics_dict["per_category"] = {}
            for cat in sorted(per_category):
                cat_metrics = per_category[cat]
                cat_n = cat_metrics["total"]
                metrics_dict["per_category"][cat] = {
                    "total": cat_n,
                    "name_correct": cat_metrics['name_correct'],
                    "args_correct": cat_metrics['args_correct'],
                    "exact_match": cat_metrics['exact_match'],
                    "bfcl_valid": cat_metrics['bfcl_valid'],
                    "valid_json": cat_metrics['valid_json'],
                    "generation_failed": cat_metrics['generation_failed'],
                }
                if cat_n > 0:
                    metrics_dict["per_category"][cat]["name_correct_pct"] = cat_metrics['name_correct'] / cat_n
                    metrics_dict["per_category"][cat]["args_correct_pct"] = cat_metrics['args_correct'] / cat_n
                    metrics_dict["per_category"][cat]["exact_match_pct"] = cat_metrics['exact_match'] / cat_n
                    metrics_dict["per_category"][cat]["bfcl_valid_pct"] = cat_metrics['bfcl_valid'] / cat_n
                    metrics_dict["per_category"][cat]["valid_json_pct"] = cat_metrics['valid_json'] / cat_n
                    metrics_dict["per_category"][cat]["generation_failed_pct"] = cat_metrics['generation_failed'] / cat_n
        
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output_jsonl}")
        print(f"Metrics saved to: {metrics_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FC tool call generation on Lhotse shar or BFCL datasets")

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--shar_path",
        help="Path to Lhotse shar directory containing cuts.*.jsonl.gz",
    )
    input_group.add_argument(
        "--bfcl_data",
        help="Path to BFCL JSONL data file (e.g. BFCL_v4_simple_python.json)",
    )
    input_group.add_argument(
        "--inference_json",
        help="Path to inference JSONL files (glob pattern, e.g. 'dir/bfcl_v3_rank*.json')",
    )

    parser.add_argument(
        "--bfcl_answer",
        help="Path to BFCL possible answer file (required when using --bfcl_data)",
    )
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--output_jsonl", default=None, help="Output JSONL file for detailed results")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1, help="Low temperature for more deterministic output")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--use_gt_user_text", action="store_true",
                        help="Use ground-truth user text (src_text) instead of ITN-normalized predicted text for LLM generation (upper bound)")
    parser.add_argument("--apply_itn", action="store_true",
                        help="Apply LLM-based inverse text normalization to ASR-predicted text before LLM generation")
    parser.add_argument("--device", default="auto", help="Device map for model loading")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply post-processing normalization to fix trivial formatting differences "
                             "(location suffixes, type coercion, list ordering, abbreviations, etc.)")
    args = parser.parse_args()

    if args.bfcl_data and not args.bfcl_answer:
        parser.error("--bfcl_answer is required when using --bfcl_data")

    model, tokenizer = load_model(args.model, device=args.device)
    is_nemotron = "nemotron" in args.model.lower()

    run_eval(model, tokenizer, is_nemotron, args)


if __name__ == "__main__":
    main()
