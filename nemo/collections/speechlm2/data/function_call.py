# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import random
import re
import torch
from dataclasses import dataclass
from typing import List

from lhotse import compute_num_frames
from lhotse.cut import Cut
from lhotse import SupervisionSegment

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id, collate_and_pad, collate_and_pad_1d, collate_and_pad_2d
from nemo.utils import logging


# One-time log for strip_text_before_toolcall
_logged_strip_toolcall_example = False
# One-time log for normalize_toolcall_arguments (when debug_fc is True)
_logged_normalize_toolcall_example = False

# Regex to find <TOOLCALL>...</TOOLCALL> blocks for argument normalization
_TOOLCALL_BLOCK_PATTERN = re.compile(r'<TOOLCALL>(.*?)</TOOLCALL>', re.DOTALL)

# Default template for augmenting function-calling system prompts that only contain <AVAILABLE_TOOLS>...</AVAILABLE_TOOLS>.
# Use {tools_content} as placeholder for the existing tools block.
# Literal braces in the template are escaped ({{ }}) so .format(tools_content=...) does not interpret them.
DEFAULT_FC_SYSTEM_PROMPT_TEMPLATE = """You can use the following tools to assist the user if required:
{tools_content}

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, {{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>

Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user."""

DEFAULT_FC_FILLER_RESPONSES = [
    "One moment, please.",
    "Give me a moment.",
    "Sure, one second.",
    "Let me pull that up.",
    "Just a moment.",
    "One second, please.",
    "Let me check.",
    "Checking on that.",
    "Hang on a sec.",
    "Looking it up.",
]

# Closing tags for TOOLRESPONSE (dataset uses <TOOL_RESPONSE>...</TOOL_RESPONSE> or <TOOLRESPONSE>...</TOOLRESPONSE>)
_TOOLRESPONSE_CLOSING_TAGS = ("</TOOLRESPONSE>", "</TOOL_RESPONSE>")


@dataclass
class FunctionCallData:
    """Stores function call metadata."""
    tokens: torch.Tensor
    length: torch.Tensor
    time: float
    step: int
    raw_text: str


@dataclass
class FunctionCallingBatch:
    """Stores batched function calling data."""
    calls: List[torch.Tensor]
    call_lengths: List[torch.Tensor]
    call_times: List[float]
    call_steps: List[int]
    call_raw_text: List[str]

    responses: List[torch.Tensor]
    response_lengths: List[torch.Tensor]
    response_times: List[float]
    response_steps: List[int]
    response_raw_text: List[str]


def _fc_system_prompt_can_be_augmented(prompt_text: str) -> bool:
    """Return True if the function-calling system prompt is tools-only and should be wrapped with full instructions.
    We only check for the instruction phrase; many FC datasets already have <TOOLCALL> in content but lack the preamble.
    """
    if not prompt_text or not prompt_text.strip():
        return False
    text = prompt_text.strip()
    # Already has the full format if it contains the instruction preamble
    if "If you decide to call any tool" in text:
        return False
    return True


def _augment_fc_system_prompt(prompt_text: str, template: str) -> str:
    """Wrap a tools-only system prompt with the full function-calling instruction template."""
    return template.format(tools_content=prompt_text.strip())


def _get_fc_prompt(
    prompt_text: str,
    fc_system_prompt: str | None,
) -> str:
    """
    Get function calling prompt text, optionally augmenting it if needed.
    
    If the FC prompt is tools-only (e.g. only <AVAILABLE_TOOLS>...</AVAILABLE_TOOLS>),
    it is wrapped with the full instruction template when fc_system_prompt is provided.
    
    Args:
        prompt_text: The original prompt text
        fc_system_prompt: The FC system prompt template to use for augmentation
        
    Returns:
        The (possibly augmented) prompt text
    """
    if fc_system_prompt and _fc_system_prompt_can_be_augmented(prompt_text):
        prompt_text = _augment_fc_system_prompt(prompt_text, fc_system_prompt)
    return prompt_text


def _validate_time(input_time: float, cut_duration: float) -> float:
    """Validate and clamp time to cut duration."""
    if input_time > cut_duration + 0.16:
        logging.info(f"{input_time} > {cut_duration} in cut")
    return min(input_time, cut_duration)


def _strip_text_before_toolcall(text: str) -> str:
    """Remove acknowledgement or any text before <TOOLCALL> in an assistant turn.

    When an assistant turn contains <think>...</think> and/or free text followed by <TOOLCALL>,
    we keep only from the first <TOOLCALL> to the end so the model sees only the
    tool-call block. If <TOOLCALL> is not present, the text is returned unchanged.
    """
    idx = text.find("<TOOLCALL>")
    if idx >= 0:
        return text[idx:]
    return text


def _normalize_toolcall_arguments(text: str) -> str:
    """Convert TOOLCALL content from raw-arguments format to no-escapes format.

    When data was created with only:
      "arguments": tool_call["function"]["arguments"]
    the 'arguments' value is a JSON string (e.g. \"{\\\"key\\\": \\\"value\\\"}\").
    This function parses that string so the stored format matches the try-block
    format where arguments are parsed with json.loads before json.dumps (so
    arguments are JSON objects, not escaped strings).
    """
    def replace_one(match):
        inner = match.group(1).strip()
        try:
            arr = json.loads(inner)
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(arr, list):
            return match.group(0)
        normalized = []
        for item in arr:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            normalized.append({**item, "arguments": args})
        return "<TOOLCALL>" + json.dumps(normalized) + "</TOOLCALL>"

    return _TOOLCALL_BLOCK_PATTERN.sub(replace_one, text)


def _process_function_supervisions(
    supervision: SupervisionSegment,
    tokenizer: TokenizerSpec,
    frame_length: float,
    target_sample_rate: int,
    cut_duration: float,
    debug_fc: bool = False,
    normalize_toolcall_arguments: bool = False,
    strip_text_before_toolcall: bool = False,
) -> FunctionCallData:
    """Process a single function call or response supervision segment.

    This function handles both function calls and responses equally - it simply
    tokenizes the text and computes timing information. The distinction between
    calls and responses is made at the batch collection level based on segment order.
    """
    func_text = supervision.custom['function']
    # Only strip text before <TOOLCALL> when segment contains <TOOLCALL> (agent turn); skip system and TOOL_RESPONSE.
    if strip_text_before_toolcall and "<TOOLCALL>" in func_text:
        global _logged_strip_toolcall_example
        if not _logged_strip_toolcall_example:
            before_strip = func_text
            func_text = _strip_text_before_toolcall(func_text)
            _logged_strip_toolcall_example = True
            if before_strip != func_text:
                logging.info("[FC] strip_text_before_toolcall example (one-time per process):")
                logging.info(f"[FC]   BEFORE: {before_strip}")
                logging.info(f"[FC]   AFTER:  {func_text}")
            else:
                logging.info("[FC] strip_text_before_toolcall: no change for this example (before == after)")
        else:
            func_text = _strip_text_before_toolcall(func_text)
    # Only normalize TOOLCALL arguments when segment contains <TOOLCALL>; skip TOOL_RESPONSE segments.
    if normalize_toolcall_arguments and "<TOOLCALL>" in func_text:
        global _logged_normalize_toolcall_example
        if not _logged_normalize_toolcall_example:
            before_norm = func_text
            func_text = _normalize_toolcall_arguments(func_text)
            _logged_normalize_toolcall_example = True
            if before_norm != func_text:
                logging.info("[FC] normalize_toolcall_arguments example (one-time per process):")
                logging.info(f"[FC]   BEFORE: {before_norm}")
                logging.info(f"[FC]   AFTER:  {func_text}")
            else:
                logging.info("[FC] normalize_toolcall_arguments: no change for this example (before == after)")
        else:
            func_text = _normalize_toolcall_arguments(func_text)

    func_tokens = torch.as_tensor(tokenizer.text_to_ids(func_text))

    func_start_time = _validate_time(supervision.start, cut_duration)
    func_start_step = compute_num_frames(
        duration=func_start_time,
        frame_shift=frame_length,
        sampling_rate=target_sample_rate
    )

    if debug_fc:
        logging.debug(f"[FC] Processing function segment: text='{func_text[:50]}...', "
                     f"tokens={len(func_tokens)}, start_time={func_start_time:.2f}s, step={func_start_step}")

    return FunctionCallData(
        tokens=func_tokens,
        length=torch.as_tensor(len(func_tokens)),
        time=func_start_time,
        step=func_start_step,
        raw_text=func_text
    )


def _extract_function_calling_data(
    function_supervisions: List[SupervisionSegment],
    tokenizer: TokenizerSpec,
    frame_length: float,
    target_sample_rate: int,
    cut_duration: float,
    debug_fc: bool = False,
    normalize_toolcall_arguments: bool = False,
    strip_text_before_toolcall: bool = False,
) -> FunctionCallingBatch:
    """
    Extract function calls and responses from segments.

    Assumes segments alternate: [call, response, call, response, ...]
    """
    if debug_fc:
        logging.info(f"[FC] Extracting function calling data from {len(segments)} segments")

    batch = FunctionCallingBatch(
        calls=[], call_lengths=[], call_times=[], call_steps=[], call_raw_text=[],
        responses=[], response_lengths=[], response_times=[], response_steps=[], response_raw_text=[]
    )

    for i in range(0, len(function_supervisions), 2):
        # Process function call (assistant turn): strip text before <TOOLCALL> when flag is True
        call = _process_function_supervisions(
            function_supervisions[i], tokenizer, frame_length, target_sample_rate, cut_duration,
            debug_fc, normalize_toolcall_arguments, strip_text_before_toolcall,
        )
        batch.calls.append(call.tokens)
        batch.call_lengths.append(call.length)
        batch.call_times.append(call.time)
        batch.call_steps.append(call.step)
        batch.call_raw_text.append(call.raw_text)

        # Process function response (user turn):
        if i + 1 < len(function_supervisions):
            response = _process_function_supervisions(
                function_supervisions[i + 1], tokenizer, frame_length, target_sample_rate, cut_duration,
                debug_fc, normalize_toolcall_arguments, False,
            )
            batch.responses.append(response.tokens)
            batch.response_lengths.append(response.length)
            batch.response_times.append(response.time)
            batch.response_steps.append(response.step)
            batch.response_raw_text.append(response.raw_text)
        else:
            # No response - add empty placeholders
            batch.responses.append(torch.tensor([], dtype=torch.long))
            batch.response_lengths.append(torch.as_tensor(0))
            batch.response_times.append(0.0)
            batch.response_steps.append(0)
            batch.response_raw_text.append("")

    if debug_fc:
        logging.info(f"[FC] Extracted {len(batch.calls)} function calls and {len(batch.responses)} responses")

    return batch


def _is_function_calling_cut(cut: Cut) -> bool:
    """Robustly detect if a cut contains function-calling supervision content."""
    if getattr(cut, "s2s_duplex_function_calling", False):
        return True
    if len(cut.supervisions) <= 1:
        return False
    # Typical FC format: first supervision is system and later turns contain custom["function"].
    if cut.supervisions[0].speaker != "system":
        return False
    for sup in cut.supervisions[1:]:
        custom = getattr(sup, "custom", None) or {}
        if (custom.get("function") or "").strip() != "":
            return True
    return False


def _is_agent_toolcall(sup) -> bool:
    """True if this supervision is an agent turn that is a TOOLCALL (not natural-language text)."""
    custom = getattr(sup, "custom", None) or {}
    raw = (custom.get("function") or getattr(sup, "text", None) or "").strip()
    return raw.startswith("<TOOLCALL>")


def _is_assistant_after_tool_response(supervision, all_supervisions, supervision_index, output_roles) -> bool:
    """Check if an assistant supervision comes immediately after a TOOL_RESPONSE.
    
    Returns True if this assistant turn should be skipped (comes after tool response).
    """
    # Only check assistant turns
    if supervision.speaker not in output_roles:
        return False
    
    # Look backwards to find if there was a TOOL_RESPONSE before this turn
    for i in range(supervision_index - 1, -1, -1):
        prev_sup = all_supervisions[i]
        custom = getattr(prev_sup, "custom", None) or {}
        # Only check the custom["function"] field for TOOL_RESPONSE tags.
        # Do NOT fall back to sup.text — system prompts may contain example
        # </TOOL_RESPONSE> tags that would cause false positives.
        func_text = (custom.get("function") or "").strip()
        if not func_text:
            continue

        # Check if previous turn contains TOOL_RESPONSE closing tag
        if any(tag in func_text for tag in _TOOLRESPONSE_CLOSING_TAGS):
            # Found a TOOL_RESPONSE, check if there are any other assistant turns between it and current
            for j in range(i + 1, supervision_index):
                intermediate_sup = all_supervisions[j]
                if intermediate_sup.speaker in output_roles:
                    # There's another assistant turn between TOOL_RESPONSE and current, so current is not immediately after
                    return False
            # Current assistant turn comes immediately after TOOL_RESPONSE
            return True
    
    return False


def _extract_target_text_after_tool_response(cut: Cut, output_roles: set) -> list:
    """
    Extract every assistant (output_roles) response that comes right after a
    <TOOLRESPONSE>...</TOOLRESPONSE> / <TOOL_RESPONSE>...</TOOL_RESPONSE> turn,
    and is actual natural-language text (not another TOOLCALL).

    - TOOL_RESPONSE lives in user (or system) turns, not agent turns.
    - Walk all supervisions in order. When a turn contains the closing tag, find
      the next turn with speaker in output_roles (agent). If that agent turn is
      a TOOLCALL, skip (don't add). If it is real text, append it to segments.
    """
    segments = []
    supervisions = list(cut.supervisions)
    for i, sup in enumerate(supervisions):
        custom = getattr(sup, "custom", None) or {}
        text = (custom.get("function") or getattr(sup, "text", None) or "").strip()
        if not any(tag in text for tag in _TOOLRESPONSE_CLOSING_TAGS):
            continue
        # This turn is a TOOL_RESPONSE. Find the next agent turn.
        for j in range(i + 1, len(supervisions)):
            next_sup = supervisions[j]
            if next_sup.speaker not in output_roles:
                continue
            # Next agent turn found.
            if _is_agent_toolcall(next_sup):
                break  # Agent sent another TOOLCALL, not a text response; don't add.
            next_text = (
                getattr(next_sup, "text", None)
                or (getattr(next_sup, "custom", None) or {}).get("orig_text")
                or ""
            ).strip()
            if next_text:
                segments.append(next_text)
            break

    return segments


def _extract_post_fc_response_aligned(
    cut: Cut, output_roles: set, debug: bool = False,
    frame_length: float = 0.08, target_sample_rate: int = 16000,
) -> list:
    """Extract post-FC response texts aligned with tool call index.

    Returns a list of length = number of tool calls (TOOLCALL segments).
    Each entry is the agent text after the corresponding tool response,
    or "" if no agent text follows (e.g., another tool call follows).
    The N-th entry corresponds to the N-th agent_fc_eos in target_tokens.
    """
    supervisions = list(cut.supervisions)
    # Find all TOOLCALL segments (agent turns with <TOOLCALL>)
    toolcall_indices = []
    for i, sup in enumerate(supervisions):
        if _is_agent_toolcall(sup):
            toolcall_indices.append(i)

    if debug and toolcall_indices:
        logging.info(f"[FC extract] cut={cut.id}: found {len(toolcall_indices)} TOOLCALL segments at supervision indices {toolcall_indices}")

    def _to_frame(time_s):
        if time_s is None:
            return "?"
        return compute_num_frames(time_s, frame_length, target_sample_rate)

    result = []
    for tc_num, tc_idx in enumerate(toolcall_indices):
        tc_sup = supervisions[tc_idx]
        tc_custom = getattr(tc_sup, "custom", None) or {}
        tc_text = (tc_custom.get("function") or getattr(tc_sup, "text", None) or "").strip()
        tc_time = getattr(tc_sup, "start", None)

        if debug:
            logging.info(
                f"[FC extract] cut={cut.id}, toolcall #{tc_num}: "
                f"frame={_to_frame(tc_time)}, text='{tc_text[:120]}{'...' if len(tc_text) > 120 else ''}'"
            )

        # Find the tool response after this tool call
        response_idx = None
        for j in range(tc_idx + 1, len(supervisions)):
            custom = getattr(supervisions[j], "custom", None) or {}
            text = (custom.get("function") or getattr(supervisions[j], "text", None) or "").strip()
            if any(tag in text for tag in _TOOLRESPONSE_CLOSING_TAGS):
                response_idx = j
                break

        if response_idx is None:
            if debug:
                logging.info(f"[FC extract] cut={cut.id}, toolcall #{tc_num}: no TOOL_RESPONSE found after this toolcall")
            result.append("")
            continue

        resp_sup = supervisions[response_idx]
        resp_custom = getattr(resp_sup, "custom", None) or {}
        resp_text = (resp_custom.get("function") or getattr(resp_sup, "text", None) or "").strip()
        resp_time = getattr(resp_sup, "start", None)

        if debug:
            logging.info(
                f"[FC extract] cut={cut.id}, toolcall #{tc_num}: "
                f"TOOL_RESPONSE at frame={_to_frame(resp_time)}, text='{resp_text[:120]}{'...' if len(resp_text) > 120 else ''}'"
            )

        # Find agent text after tool response
        agent_text = ""
        for j in range(response_idx + 1, len(supervisions)):
            next_sup = supervisions[j]
            if next_sup.speaker not in output_roles:
                continue
            if _is_agent_toolcall(next_sup):
                if debug:
                    logging.info(f"[FC extract] cut={cut.id}, toolcall #{tc_num}: next agent turn is another TOOLCALL, no post-response text")
                break  # Another tool call, no text response
            next_text = (
                getattr(next_sup, "text", None)
                or (getattr(next_sup, "custom", None) or {}).get("orig_text")
                or ""
            ).strip()
            agent_text = next_text
            break

        if debug:
            if agent_text:
                logging.info(
                    f"[FC extract] cut={cut.id}, toolcall #{tc_num}: "
                    f"post_response_text='{agent_text[:120]}{'...' if len(agent_text) > 120 else ''}'"
                )
            else:
                logging.info(f"[FC extract] cut={cut.id}, toolcall #{tc_num}: no post-response text (empty)")

        result.append(agent_text)

    return result


def extract_fc_batch_data(
    cuts, tokenizer, pad_id, frame_length, target_sample_rate,
    output_roles, debug_fc, normalize_toolcall_arguments,
    strip_text_before_toolcall,
) -> dict:
    """Extract function calling metadata from a batch of cuts.

    Iterates over cuts, extracts function segments from supervisions[1:],
    calls _extract_function_calling_data, collates results, and returns a dict
    with all FC batch keys.
    """
    if debug_fc:
        logging.info(f"[FC] Processing function calling batch with {len(cuts)} cuts")

    metadata = []
    num_turns = []
    # Separate storage for calls and responses
    function_calls, function_call_lengths = [], []
    function_call_times, function_call_steps = [], []
    function_call_raw_text = []
    function_responses, function_response_lengths = [], []
    function_response_times, function_response_steps = [], []
    function_response_raw_text = []

    # Iterate over all cuts in batch to extract function calling metadata
    for cut_id, cut in enumerate(cuts):
        # Note: First supervision (system) is now handled by collate_system_prompt as prompt_tokens
        # We only extract function calls and responses here (supervisions[1:])
        num_turns.append(len(cut.supervisions) - 1)  # 1st supervision is system prompt
        metadata.append({'audio_filepath': cut.id + '.wav'})

        if debug_fc:
            logging.debug(f"[FC] Processing cut {cut_id}: {cut.id}, supervisions={len(cut.supervisions)}")

        # Validate first supervision is system (now extracted as system prompt)
        if cut.supervisions[0].speaker != 'system':
            logging.error(f"Assertion failed: cut.id={cut.id}, first supervision speaker='{cut.supervisions[0].speaker}', expected='system'")
            logging.error(f"Cut object: {cut}")

        assert cut.supervisions[0].speaker == 'system'

        # Extract function segments (if they exist) from supervisions[1:]
        if len(cut.supervisions) > 1 and 'function' in cut.supervisions[1].custom:
            function_supervisions = [
                sup
                for sup in cut.supervisions[1:]
                if (sup.custom.get("function") or "").strip() != ""
            ]
            if debug_fc:
                logging.debug(f"[FC] Cut {cut_id} has {len(function_supervisions)} function segments")
        else:
            function_supervisions = []
            if debug_fc:
                logging.debug(f"[FC] Cut {cut_id} has no function segments")

        # Extract function calls and responses separately using helper
        if len(function_supervisions) > 0:
            fc_batch = _extract_function_calling_data(
                function_supervisions,
                tokenizer,
                frame_length,
                target_sample_rate,
                cut.duration,
                debug_fc,
                normalize_toolcall_arguments,
                strip_text_before_toolcall,
            )

            # Collate calls
            function_calls.append(collate_and_pad(fc_batch.calls, get_pad_id(tokenizer))[0])
            function_call_lengths.append(fc_batch.call_lengths)
            function_call_times.append(fc_batch.call_times)
            function_call_steps.append(fc_batch.call_steps)
            function_call_raw_text.append(fc_batch.call_raw_text)

            # Collate responses
            function_responses.append(collate_and_pad(fc_batch.responses, get_pad_id(tokenizer))[0])
            function_response_lengths.append(fc_batch.response_lengths)
            function_response_times.append(fc_batch.response_times)
            function_response_steps.append(fc_batch.response_steps)
            function_response_raw_text.append(fc_batch.response_raw_text)
        else:
            # No function segments in this cut (e.g., refusal-only samples in FC datasets):
            # append empty placeholders so batch size stays aligned with target_tokens.
            empty_calls = torch.empty((0, 0), dtype=torch.long)
            empty_lengths = torch.tensor([], dtype=torch.long)
            empty_times = torch.tensor([], dtype=torch.float)
            empty_steps = torch.tensor([], dtype=torch.long)

            function_calls.append(empty_calls)
            function_call_lengths.append(empty_lengths)
            function_call_times.append(empty_times)
            function_call_steps.append(empty_steps)
            function_call_raw_text.append([])

            function_responses.append(empty_calls)
            function_response_lengths.append(empty_lengths)
            function_response_times.append(empty_times)
            function_response_steps.append(empty_steps)
            function_response_raw_text.append([])

    # Collate function calling data if present
    if len(function_calls) > 0:
        # Collate function calls
        fc_req_tokens = collate_and_pad_2d(function_calls, get_pad_id(tokenizer))  # [b, t, l]
        fc_req_lens = collate_and_pad_1d(function_call_lengths)  # [b, t]
        # Convert times to float tensors before collating to preserve float dtype
        function_call_times_tensors = [
            torch.tensor(times, dtype=torch.float) if not isinstance(times, torch.Tensor)
            else (times if times.dtype == torch.float else times.float())
            for times in function_call_times
        ]
        # Use float pad_id and ensure output is float
        fc_req_times = collate_and_pad_1d(function_call_times_tensors, pad_id=-1.0)  # [b, t]
        fc_req_times = fc_req_times.float()  # Ensure float dtype
        fc_req_steps = collate_and_pad_1d(function_call_steps)  # [b, t]

        if debug_fc:
            logging.info(f"[FC] Collated function calls: shape={function_calls.shape}, "
                       f"lengths={function_call_lengths.shape}, steps={function_call_steps.shape}")

        # Collate function responses
        fc_res_tokens = collate_and_pad_2d(function_responses, get_pad_id(tokenizer))  # [b, t, l]
        fc_res_lens = collate_and_pad_1d(function_response_lengths)  # [b, t]
        # Convert times to float tensors before collating to preserve float dtype
        function_response_times_tensors = [
            torch.tensor(times, dtype=torch.float) if not isinstance(times, torch.Tensor)
            else (times if times.dtype == torch.float else times.float())
            for times in function_response_times
        ]
        # Use float pad_id and ensure output is float
        fc_res_times = collate_and_pad_1d(function_response_times_tensors, pad_id=-1.0)  # [b, t]
        fc_res_times = fc_res_times.float()  # Ensure float dtype
        fc_res_steps = collate_and_pad_1d(function_response_steps)  # [b, t]

        if debug_fc:
            logging.info(f"[FC] Collated function responses: shape={function_responses.shape}, "
                       f"lengths={function_response_lengths.shape}, steps={function_response_steps.shape}")
    else:
        if debug_fc:
            logging.info("[FC] No function calling data to collate")
        # No function calling data
        fc_req_tokens = None
        fc_req_lens = None
        fc_req_times = None
        fc_req_steps = None
        fc_res_tokens = None
        fc_res_lens = None
        fc_res_times = None
        fc_res_steps = None

    # Extract aligned post-FC response texts and tokenize them
    fc_post_res_tokens_per_cut = []
    fc_post_res_lens_per_cut = []
    for cut_id, cut in enumerate(cuts):
        texts = _extract_post_fc_response_aligned(
            cut, output_roles, debug=True,
            frame_length=frame_length, target_sample_rate=target_sample_rate,
        )
        logging.info(
            f"[FC extract] cut {cut_id} ({cut.id}): {len(texts)} aligned post-FC responses, "
            f"non-empty={sum(1 for t in texts if t)}"
        )
        cut_tokens = []
        cut_lens = []
        for turn_idx, text in enumerate(texts):
            if text:
                ids = torch.as_tensor(tokenizer.text_to_ids(text))
                logging.info(f"[FC extract] cut {cut_id}, turn {turn_idx}: tokenized post-FC response -> {len(ids)} tokens")
            else:
                ids = torch.empty(0, dtype=torch.long)
            cut_tokens.append(ids)
            cut_lens.append(torch.as_tensor(len(ids)))
        if cut_tokens:
            fc_post_res_tokens_per_cut.append(collate_and_pad(cut_tokens, get_pad_id(tokenizer))[0])
            fc_post_res_lens_per_cut.append(cut_lens)
        else:
            fc_post_res_tokens_per_cut.append(torch.empty((0, 0), dtype=torch.long))
            fc_post_res_lens_per_cut.append([])

    # Collate post-FC response tokens across cuts
    fc_post_res_tokens = collate_and_pad_2d(fc_post_res_tokens_per_cut, get_pad_id(tokenizer)) if fc_post_res_tokens_per_cut else None
    fc_post_res_lens = collate_and_pad_1d(fc_post_res_lens_per_cut) if fc_post_res_lens_per_cut else None

    result = {
        "metadata": metadata,
        "num_turns": torch.tensor(num_turns),
        # Function calls (separate)
        "fc_req_tokens": fc_req_tokens,
        "fc_req_lens": fc_req_lens,
        "fc_req_times": fc_req_times,
        "fc_req_steps": fc_req_steps,
        "fc_req_raw_text": function_call_raw_text,  # List[List[str]], raw TOOLCALL text per cut per turn
        "fc_res_tokens": fc_res_tokens,
        "fc_res_lens": fc_res_lens,
        "fc_res_times": fc_res_times,
        "fc_res_steps": fc_res_steps,
        # Post-FC response tokens (aligned with tool call index)
        "fc_post_res_tokens": fc_post_res_tokens,
        "fc_post_res_lens": fc_post_res_lens,
        # Extract "assistant response after TOOLRESPONSE" per cut for BLEU-after-tool metric
        "target_text_after_tool_response": [
            _extract_target_text_after_tool_response(cut, output_roles)
            for cut in cuts
        ],
    }

    if debug_fc:
        logging.info(f"[FC] Function calling metadata added to batch: "
                   f"num_cuts={len(metadata)}, has_calls={function_calls is not None}")
        logging.info(f"[FC] System prompt for function calling is in prompt_tokens (from collate_system_prompt)")

    return result


def add_minimal_batch_fc_data(
    out: dict,
    tokenizer: TokenizerSpec,
    frame_length: float,
    model_cfg: dict | None = None,
    fc_drop_info: dict | None = None,
) -> None:
    """Add function calling data to a minimal batch dictionary.
    
    This function populates the minimal batch with function calling data when
    a batch is dropped due to FC constraints (e.g., system+FC tokens > max_fc_lens).
    
    Args:
        out: Dictionary to modify in-place with minimal batch FC data
        tokenizer: Tokenizer to use for encoding text
        frame_length: Frame length for computing timing
        model_cfg: Optional model config dict that may contain minimal_batch_fc_prompt_text
        fc_drop_info: Optional dict with drop information, e.g.:
            {"cut_id": str, "total_prompt_tokens": int, "max_fc_total_tokens": int, "reason": str}
    """
    if fc_drop_info is not None:
        out["fc_drop_info"] = fc_drop_info
    out["metadata"] = [{"audio_filepath": "empty_batch.wav"}]
    tool_call_text = (
        '<TOOLCALL>[{"name": "execute_program", "arguments": {"program_name": "DataAnalysis", '
        '"arguments": ["dataset1", "dataset2"]}}]</TOOLCALL>'
    )
    tool_response_text = (
        '<TOOL_RESPONSE>[{"status": "success", "message": "The program \'DataAnalysis\' has been successfully executed '
        'with the arguments \'dataset1\' and \'dataset2\'. The output file \'AnalysisResult\' has been created."}]</TOOL_RESPONSE>'
    )
    call_ids = tokenizer.text_to_ids(tool_call_text) if tokenizer is not None else []
    resp_ids = tokenizer.text_to_ids(tool_response_text) if tokenizer is not None else []

    minimal_fc_step = 12
    minimal_fc_time = minimal_fc_step * frame_length
    out["num_turns"] = torch.tensor([1], dtype=torch.long)
    out["fc_req_tokens"] = torch.as_tensor([[call_ids]], dtype=torch.long)
    out["fc_req_lens"] = torch.tensor([[len(call_ids)]], dtype=torch.long)
    out["fc_req_times"] = torch.tensor([[minimal_fc_time]], dtype=torch.float)
    out["fc_req_steps"] = torch.tensor([[minimal_fc_step]], dtype=torch.long)
    out["function_call_raw_text"] = [[tool_call_text]]
    out["fc_res_tokens"] = torch.as_tensor([[resp_ids]], dtype=torch.long)
    out["fc_res_lens"] = torch.tensor([[len(resp_ids)]], dtype=torch.long)
    out["fc_res_times"] = torch.tensor([[minimal_fc_time]], dtype=torch.float)
    out["fc_res_steps"] = torch.tensor([[minimal_fc_step]], dtype=torch.long)
    out["function_response_raw_text"] = [[tool_response_text]]

    default_minimal_fc_prompt = (
        '<AVAILABLE_TOOLS>[{"name":"execute_program","description":"Execute a specific program with given arguments",'
        '"parameters":{"type":"dict","properties":{"program_name":{"type":"string","description":"The name of the program to be executed"},'
        '"arguments":{"type":"array","items":{"type":"string"},"description":"The arguments to be passed to the program"}},'
        '"required":["program_name"]}},{"name":"generate_qr_code","description":"Generate a QR code for a given text",'
        '"parameters":{"type":"dict","properties":{"text":{"type":"string","description":"The text to encode in the QR code"}},'
        '"required":["text"]}}]</AVAILABLE_TOOLS>'
    )
    minimal_fc_prompt_text = default_minimal_fc_prompt
    if model_cfg is not None:
        minimal_fc_prompt_text = model_cfg.get("minimal_batch_fc_prompt_text", default_minimal_fc_prompt)

    prompt_ids = []
    if tokenizer is not None and minimal_fc_prompt_text:
        bos = getattr(tokenizer, "bos", None)
        eos = getattr(tokenizer, "eos", None)
        if bos is not None:
            prompt_ids.append(int(bos))
        prompt_ids.extend(tokenizer.text_to_ids(minimal_fc_prompt_text))
        if eos is not None:
            prompt_ids.append(int(eos))
    if len(prompt_ids) == 0:
        out["prompt_tokens"] = torch.empty((1, 0), dtype=torch.long)
        out["prompt_token_lens"] = torch.tensor([0], dtype=torch.long)
    else:
        out["prompt_tokens"] = torch.as_tensor([prompt_ids], dtype=torch.long)
        out["prompt_token_lens"] = torch.tensor([len(prompt_ids)], dtype=torch.long)
    out["target_turn_texts"] = [[{
        "start_time": 0.0,
        "duration": 0.0,
        "role": "assistant",
        "text": "hello",
    }]]
    out["source_turn_texts"] = [[{
        "start_time": 0.0,
        "duration": 0.0,
        "role": "user",
        "text": "hello",
    }]]
    out["system_prompt"] = [minimal_fc_prompt_text]


def get_fc_cut_total_prompt_tokens(
    cut: Cut,
    tokenizer: TokenizerSpec,
    fc_system_prompt: str | None = None,
) -> int:
    """Compute total token count for system prompt + all function-call/response segments.
    Used to decide if we should drop the batch when over threshold.
    """
    if not _is_function_calling_cut(cut) or len(cut.supervisions) == 0:
        return 0
    total = 0
    # System prompt (same logic as collate_system_prompt: BOS + text + EOS)
    prompt_text = cut.supervisions[0].text if cut.supervisions[0].speaker == 'system' else ""
    if prompt_text:
        prompt_text = _get_fc_prompt(prompt_text, fc_system_prompt)
        total += 1 + len(tokenizer.text_to_ids(prompt_text)) + 1
    # All function-call and response segments.
    function_segments = []
    for sup in cut.supervisions[1:]:
        custom = getattr(sup, "custom", None) or {}
        seg_text = (custom.get("function") or sup.text or "").strip()
        if seg_text:
            function_segments.append(seg_text)
    for idx, seg_text in enumerate(function_segments):
        total += len(tokenizer.text_to_ids(seg_text))
        if idx % 2 == 0:
            total += 2
        else:
            total += 1
    return total


def resolve_fc_tokens(
    cfg,
    model_cfg,
    tokenizer: TokenizerSpec,
    agent_bos_id: int,
    agent_eos_id: int,
) -> tuple:
    """Resolve agent_fc_bos_id and agent_fc_eos_id from config.
    Returns (agent_fc_bos_id, agent_fc_eos_id).
    """
    # Default based on model type (mirrors duplex_stt_model.py token setup)
    if model_cfg is not None and 'Nemotron' in model_cfg.get('pretrained_llm', ''):
        default_fc_bos_token = '<SPECIAL_13>'
        default_fc_eos_token = '<SPECIAL_14>'
    elif model_cfg is not None and 'Qwen2.5' in model_cfg.get('pretrained_llm', ''):
        default_fc_bos_token = '<tool_call>'
        default_fc_eos_token = '</tool_call>'
    else:
        default_fc_bos_token = None
        default_fc_eos_token = None

    agent_fc_bos_token = (cfg or {}).get("agent_fc_bos_token", default_fc_bos_token)
    if agent_fc_bos_token is not None:
        token_ids = tokenizer.text_to_ids(agent_fc_bos_token)
        if len(token_ids) != 1:
            raise ValueError(
                f"agent_fc_bos_token '{agent_fc_bos_token}' must tokenize to exactly 1 token, "
                f"but got {len(token_ids)} tokens: {token_ids}. "
                f"Please configure your tokenizer to ensure this token maps to a single ID."
            )
        agent_fc_bos_id = token_ids[0]
    else:
        agent_fc_bos_id = agent_bos_id  # Default to regular agent BOS

    agent_fc_eos_token = (cfg or {}).get("agent_fc_eos_token", default_fc_eos_token)
    if agent_fc_eos_token is not None:
        token_ids = tokenizer.text_to_ids(agent_fc_eos_token)
        if len(token_ids) != 1:
            raise ValueError(
                f"agent_fc_eos_token '{agent_fc_eos_token}' must tokenize to exactly 1 token, "
                f"but got {len(token_ids)} tokens: {token_ids}."
            )
        agent_fc_eos_id = token_ids[0]
    else:
        agent_fc_eos_id = agent_eos_id

    logging.info(
        f"[FC tokens] agent_bos_id={agent_bos_id}, agent_eos_id={agent_eos_id}, "
        f"agent_fc_bos_id={agent_fc_bos_id}, agent_fc_eos_id={agent_fc_eos_id}"
    )

    return agent_fc_bos_id, agent_fc_eos_id


def inject_fc_filler_responses(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    tokenizer: TokenizerSpec,
    fc_bos_id: int,
    fc_eos_id: int,
    pad_id: int,
    fc_filler_response_delay: int,
    fc_filler_responses: list,
    fc_filler_eos_offset: int = 13,
):
    """Inject random filler responses before tool calls in FC data.

    For each sample, finds all agent_fc_bos positions and rewrites each
    tool-call turn with a filler response:
    - Clears the original [agent_fc_bos, agent_eos] at the found position
    - Writes [agent_fc_bos, filler_tok1, ..., filler_tokN] at
      (original_pos + fc_filler_response_delay)
    - Places fc_eos at a fixed offset (fc_filler_eos_offset) after fc_bos,
      corresponding to ~1 sec + 1 frame of speech at 12.5 Hz.
    """
    seq_len = target_tokens.shape[1]

    for i in range(target_tokens.shape[0]):
        fc_bos_positions = (target_tokens[i] == fc_bos_id).nonzero(as_tuple=True)[0].tolist()
        if not fc_bos_positions:
            continue

        for old_pos in fc_bos_positions:
            # Clear original agent_fc_bos
            target_tokens[i, old_pos] = pad_id
            # Clear original agent_fc_eos (expected at old_pos + 1)
            if old_pos + 1 < seq_len and target_tokens[i, old_pos + 1] == fc_eos_id:
                target_tokens[i, old_pos + 1] = pad_id

            # Compute insertion position
            insert_pos = old_pos + fc_filler_response_delay
            if insert_pos >= seq_len:
                logging.warning(
                    f"[_inject_fc_filler] sample {i}: insert_pos={insert_pos} >= seq_len={seq_len}, skipping"
                )
                continue

            # fc_eos at fixed offset from fc_bos (1 sec + 1 frame ≈ 13 frames at 12.5 Hz)
            eos_pos = insert_pos + fc_filler_eos_offset
            if eos_pos >= seq_len:
                logging.warning(
                    f"[_inject_fc_filler] sample {i}: eos_pos={eos_pos} >= seq_len={seq_len}, skipping"
                )
                continue

            # Sample a random filler response and tokenize
            filler_text = random.choice(fc_filler_responses)
            filler_token_ids = tokenizer.text_to_ids(filler_text)
            # Build full turn: [fc_bos] + text tokens
            turn_tokens = [fc_bos_id] + filler_token_ids
            turn_tensor = torch.tensor(turn_tokens, dtype=torch.long)

            # Truncate text tokens to fit before eos_pos (reserve eos_pos for fc_eos)
            max_turn_len = eos_pos - insert_pos
            if len(turn_tensor) > max_turn_len:
                turn_tensor = turn_tensor[:max_turn_len]

            # Write into target_tokens
            target_tokens[i, insert_pos:insert_pos + len(turn_tensor)] = turn_tensor
            # Place fc_eos at fixed offset
            target_tokens[i, eos_pos] = fc_eos_id

            # Update target_token_lens if needed
            new_end = eos_pos + 1
            if new_end > target_token_lens[i].item():
                target_token_lens[i] = min(new_end, seq_len)


def inject_fc_post_response_prefill(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    fc_post_res_tokens: torch.Tensor,
    fc_post_res_lens: torch.Tensor,
    fc_eos_id: int,
    agent_bos_id: int,
    agent_eos_id: int,
    pad_id: int,
    post_fc_delay: int,
    tokenizer,
    prefill_start_id: int,
    prefill_end_id: int,
):
    """Inject post-FC response prefill + repeat into target_tokens.

    For each sample, finds all agent_fc_eos positions and, for the N-th fc_eos,
    writes the following sequence starting at (fc_eos_pos + post_fc_delay):

        <PREFILL_START>, text_tokens..., <PREFILL_END>, agent_bos, text_tokens..., agent_eos

    The prefill portion tells the model what text to produce; the repeat portion
    is the actual prediction target (with normal text loss).

    Args:
        target_tokens: [B, T] target token tensor (modified in-place).
        target_token_lens: [B] lengths (modified in-place).
        fc_post_res_tokens: [B, num_turns, max_len] tokenized post-FC response texts.
        fc_post_res_lens: [B, num_turns] lengths of each response.
        fc_eos_id: Token ID for agent_fc_eos.
        agent_bos_id: Regular agent BOS token ID.
        agent_eos_id: Regular agent EOS token ID.
        pad_id: Pad token ID.
        post_fc_delay: Number of frames to wait after fc_eos before inserting.
        prefill_start_id: Token ID for <PREFILL_START>.
        prefill_end_id: Token ID for <PREFILL_END>.
    """
    if fc_post_res_tokens is None or fc_post_res_lens is None:
        return

    seq_len = target_tokens.shape[1]
    B = target_tokens.shape[0]

    for i in range(B):
        fc_eos_positions = (target_tokens[i] == fc_eos_id).nonzero(as_tuple=True)[0].tolist()
        if not fc_eos_positions:
            continue

        num_turns = fc_post_res_lens.shape[1] if fc_post_res_lens.dim() > 1 else 0
        # Track which turn to consume next. In multi-TOOLCALL scenarios,
        # only the last TOOLCALL in a chain gets fc_eos in target_tokens,
        # but fc_post_res_tokens has entries for all turns (earlier ones
        # empty). So for each fc_eos, consume the next non-empty response.
        next_turn = 0
        for eos_pos in fc_eos_positions:
            # Find the next non-empty post-FC response
            length = 0
            turn_idx = next_turn
            while turn_idx < num_turns:
                length = fc_post_res_lens[i, turn_idx].item()
                if length > 0:
                    break
                logging.debug(
                    f"[FC prefill] sample {i}, turn {turn_idx}: empty post-FC response, skipping to next"
                )
                turn_idx += 1
            next_turn = turn_idx + 1

            if length == 0:
                logging.debug(
                    f"[FC prefill] sample {i}: no non-empty post-FC response left for fc_eos at {eos_pos}"
                )
                continue

            text_toks = fc_post_res_tokens[i, turn_idx, :length]

            # Build: [PREFILL_START, text..., PREFILL_END, agent_bos, text..., agent_eos]
            full_seq = torch.cat([
                torch.tensor([prefill_start_id], dtype=torch.long),
                text_toks,
                torch.tensor([prefill_end_id, agent_bos_id], dtype=torch.long),
                text_toks,
                torch.tensor([agent_eos_id], dtype=torch.long),
            ])

            insert_pos = eos_pos + post_fc_delay
            if insert_pos >= seq_len:
                logging.warning(
                    f"[FC prefill] sample {i}, turn {turn_idx}: "
                    f"insert_pos={insert_pos} >= seq_len={seq_len}, skipping"
                )
                continue

            available = seq_len - insert_pos
            write_len = min(len(full_seq), available)
            if write_len < len(full_seq):
                logging.warning(
                    f"[FC prefill] sample {i}, turn {turn_idx}: "
                    f"truncating prefill+repeat from {len(full_seq)} to {write_len} tokens "
                    f"(insert_pos={insert_pos}, seq_len={seq_len})"
                )
            target_tokens[i, insert_pos:insert_pos + write_len] = full_seq[:write_len]

            new_end = insert_pos + write_len
            if new_end > target_token_lens[i].item():
                target_token_lens[i] = min(new_end, seq_len)

            decoded_text = tokenizer.ids_to_text(text_toks.tolist()) if tokenizer is not None else "<no tokenizer>"
            logging.info(
                f"[FC prefill] sample {i}, turn {turn_idx}: wrote {write_len} tokens at pos {insert_pos} "
                f"(fc_eos={eos_pos} + delay={post_fc_delay}), "
                f"text_len={length}, total_seq_len={len(full_seq)}, "
                f"prefill_text='{decoded_text}'"
            )


def _resolve_single_token(tokenizer, token_or_id, label: str) -> int:
    """Resolve a token string or integer ID to a single token ID."""
    if isinstance(token_or_id, int):
        return token_or_id
    token_ids = tokenizer.text_to_ids(token_or_id)
    if len(token_ids) != 1:
        raise ValueError(
            f"{label} '{token_or_id}' must tokenize to exactly 1 token, "
            f"but got {len(token_ids)} tokens: {token_ids}. "
            f"You can set {label} to an integer token ID directly in the config."
        )
    return token_ids[0]


def resolve_prefill_tokens(
    cfg,
    model_cfg,
    tokenizer,
) -> tuple:
    """Resolve prefill_start_id and prefill_end_id from config.

    Accepts either token strings (e.g. '<SPECIAL_15>') or integer token IDs
    via config keys ``prefill_start_token`` / ``prefill_end_token``.

    Returns (prefill_start_id, prefill_end_id).
    """
    if model_cfg is not None and 'Nemotron' in model_cfg.get('pretrained_llm', ''):
        default_start = '<SPECIAL_15>'
        default_end = '<SPECIAL_16>'
    elif model_cfg is not None and 'Qwen2.5' in model_cfg.get('pretrained_llm', ''):
        # Qwen2.5 extra tokens may not be registered as special tokens in all
        # tokenizer builds.  Fall back to raw IDs (151659 / 151660) which sit
        # right after <tool_call>/<\/tool_call> in the Qwen2.5 vocab.
        default_start = 151659
        default_end = 151660
    else:
        default_start = '<SPECIAL_15>'
        default_end = '<SPECIAL_16>'

    start = (cfg or {}).get("prefill_start_token", default_start)
    end = (cfg or {}).get("prefill_end_token", default_end)

    prefill_start_id = _resolve_single_token(tokenizer, start, "prefill_start_token")
    prefill_end_id = _resolve_single_token(tokenizer, end, "prefill_end_token")

    logging.info(
        f"[FC prefill tokens] prefill_start_id={prefill_start_id} (from '{start}'), "
        f"prefill_end_id={prefill_end_id} (from '{end}')"
    )

    return prefill_start_id, prefill_end_id


def log_fc_target_tokens(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    cuts,
    tokenizer: TokenizerSpec,
    fc_bos_id: int,
    fc_eos_id: int,
    agent_bos_id: int,
    agent_eos_id: int,
    pad_id: int,
):
    """Log the FC turn layout in target_tokens for debugging.

    For each sample, finds all non-pad token regions and decodes them,
    highlighting fc_bos / fc_eos / agent_bos / agent_eos boundaries.
    """
    for i in range(target_tokens.shape[0]):
        seq = target_tokens[i]
        seq_len = target_token_lens[i].item()
        cut_id = cuts[i].id if i < len(cuts) else "?"

        # Find all fc_bos and fc_eos positions
        fc_bos_positions = (seq[:seq_len] == fc_bos_id).nonzero(as_tuple=True)[0].tolist()
        fc_eos_positions = (seq[:seq_len] == fc_eos_id).nonzero(as_tuple=True)[0].tolist()
        agent_bos_positions = (seq[:seq_len] == agent_bos_id).nonzero(as_tuple=True)[0].tolist()
        agent_eos_positions = (seq[:seq_len] == agent_eos_id).nonzero(as_tuple=True)[0].tolist()

        logging.info(
            f"[FC target_tokens] sample {i} (cut={cut_id}): seq_len={seq_len}, "
            f"agent_bos@{agent_bos_positions}, agent_eos@{agent_eos_positions}, "
            f"fc_bos@{fc_bos_positions}, fc_eos@{fc_eos_positions}"
        )

        # For each fc_bos, show the tokens from fc_bos to the next fc_eos (or end)
        for turn_idx, bos_pos in enumerate(fc_bos_positions):
            # Find matching fc_eos (first one after this bos)
            matching_eos = [p for p in fc_eos_positions if p > bos_pos]
            eos_pos = matching_eos[0] if matching_eos else None
            if eos_pos is not None:
                turn_slice = seq[bos_pos:eos_pos + 1].tolist()
                # Decode just the filler content (between bos and eos)
                filler_ids = [t for t in turn_slice[1:-1] if t != pad_id]
                filler_text = tokenizer.ids_to_text(filler_ids) if filler_ids else "(empty)"
                logging.info(
                    f"[FC target_tokens] sample {i}, fc_turn #{turn_idx}: "
                    f"fc_bos@{bos_pos} -> fc_eos@{eos_pos} "
                    f"({eos_pos - bos_pos - 1} frames between), "
                    f"filler='{filler_text}', "
                    f"raw_ids={turn_slice}"
                )
            else:
                logging.warning(
                    f"[FC target_tokens] sample {i}, fc_turn #{turn_idx}: "
                    f"fc_bos@{bos_pos} but NO matching fc_eos found!"
                )

        # For each regular agent turn, show bos->eos and decoded text
        for turn_idx, bos_pos in enumerate(agent_bos_positions):
            matching_eos = [p for p in agent_eos_positions if p > bos_pos]
            eos_pos = matching_eos[0] if matching_eos else None
            if eos_pos is not None:
                content_ids = [t for t in seq[bos_pos + 1:eos_pos].tolist() if t != pad_id]
                content_text = tokenizer.ids_to_text(content_ids) if content_ids else "(empty)"
                logging.info(
                    f"[FC target_tokens] sample {i}, agent_turn #{turn_idx}: "
                    f"bos@{bos_pos} -> eos@{eos_pos}, "
                    f"text='{content_text[:100]}{'...' if len(content_text) > 100 else ''}'"
                )
