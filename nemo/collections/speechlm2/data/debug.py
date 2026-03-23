import torch
from nemo.utils import logging

from nemo.collections.speechlm2.data.function_call import (
    _is_assistant_after_tool_response,
    _TOOLRESPONSE_CLOSING_TAGS,
)


def _decode_target_tokens(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    pad_id: int,
    tokenizer,
    special_names: dict,
    label: str,
):
    """Log decoded target_tokens with special token names for debugging.

    Groups consecutive non-pad regions and decodes them.
    """
    for i in range(target_tokens.shape[0]):
        seq_len = target_token_lens[i].item()
        seq = target_tokens[i, :seq_len]
        non_pad_positions = (seq != pad_id).nonzero(as_tuple=True)[0].tolist()
        if not non_pad_positions:
            logging.info(f"[{label}] sample {i}: (all pad)")
            continue

        # Group consecutive non-pad positions into segments
        segments = []
        seg_start = non_pad_positions[0]
        prev = seg_start
        for p in non_pad_positions[1:]:
            if p != prev + 1:
                segments.append((seg_start, prev + 1))
                seg_start = p
            prev = p
        segments.append((seg_start, prev + 1))

        parts = []
        for s, e in segments:
            toks = seq[s:e].tolist()
            decoded_parts = []
            for tid in toks:
                if tid in special_names:
                    decoded_parts.append(special_names[tid])
                else:
                    decoded_parts.append(tokenizer.ids_to_text([tid]))
            parts.append(f"  [{s}-{e}] {''.join(decoded_parts)}")

        logging.info(
            f"[{label}] sample {i} (seq_len={seq_len}, non_pad_tokens={len(non_pad_positions)}):\n"
            + "\n".join(parts)
        )


def log_target_tokens_after_prefill(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    pad_id: int,
    tokenizer,
    agent_bos_id: int,
    agent_eos_id: int,
    agent_fc_bos_id: int,
    agent_fc_eos_id: int,
    prefill_start_id: int,
    prefill_end_id: int,
):
    """Log only the injected prefill regions in target_tokens after prefill.

    Special tags are printed in ANSI colors for easy visual debugging:
      <PREFILL_START> / <PREFILL_END> = magenta
      <agent_bos> / <agent_eos> = cyan
      <fc_bos> / <fc_eos> = yellow
    """
    # ANSI color codes
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    special_names = {
        agent_bos_id: f"{CYAN}<agent_bos>{RESET}",
        agent_eos_id: f"{CYAN}<agent_eos>{RESET}",
        agent_fc_bos_id: f"{YELLOW}<fc_bos>{RESET}",
        agent_fc_eos_id: f"{YELLOW}<fc_eos>{RESET}",
        prefill_start_id: f"{MAGENTA}<PREFILL_START>{RESET}",
        prefill_end_id: f"{MAGENTA}<PREFILL_END>{RESET}",
    }
    for i in range(target_tokens.shape[0]):
        seq_len = target_token_lens[i].item()
        seq = target_tokens[i, :seq_len]
        # Find PREFILL_START positions and decode from there to the
        # agent_eos that ends the repeat portion
        prefill_starts = (seq == prefill_start_id).nonzero(as_tuple=True)[0].tolist()
        if not prefill_starts:
            logging.info(f"[FC prefill debug AFTER] sample {i}: no prefill region found")
            continue
        parts = []
        for ps in prefill_starts:
            # Find the end: look for agent_eos after PREFILL_START
            end = seq_len
            for j in range(ps + 1, seq_len):
                if seq[j].item() == agent_eos_id:
                    end = j + 1
                    break
            toks = seq[ps:end].tolist()
            decoded_parts = []
            for tid in toks:
                if tid in special_names:
                    decoded_parts.append(special_names[tid])
                else:
                    decoded_parts.append(tokenizer.ids_to_text([tid]))
            parts.append(f"  [{ps}-{end}] {''.join(decoded_parts)}")
        logging.info(
            f"[FC prefill debug AFTER] sample {i} (seq_len={seq_len}):\n"
            + "\n".join(parts)
        )


def log_target_tokens_before_prefill(
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    pad_id: int,
    tokenizer,
    agent_bos_id: int,
    agent_eos_id: int,
    agent_fc_bos_id: int,
    agent_fc_eos_id: int,
):
    """Log decoded target_tokens before prefill injection for debugging."""
    special_names = {
        agent_bos_id: "<agent_bos>",
        agent_eos_id: "<agent_eos>",
        agent_fc_bos_id: "<fc_bos>",
        agent_fc_eos_id: "<fc_eos>",
    }
    _decode_target_tokens(
        target_tokens, target_token_lens, pad_id, tokenizer,
        special_names, "FC prefill debug BEFORE",
    )


def log_build_token_channel_decisions(cut, output_roles):
    """Log what build_token_channel would do for each supervision in a cut.

    Shows which supervisions are kept/skipped and why.
    """
    all_supervisions = list(cut.supervisions)
    logging.info(
        f"[FC build_token_channel debug] cut={cut.id}: "
        f"{len(all_supervisions)} supervisions"
    )
    for idx, sup in enumerate(all_supervisions):
        custom = getattr(sup, 'custom', None) or {}
        function_content = (custom.get('function') or '').strip()
        is_tool_call = function_content != '' and '<TOOLCALL>' in function_content
        is_tool_response = any(
            tag in function_content for tag in _TOOLRESPONSE_CLOSING_TAGS
        ) if function_content else False
        in_output_roles = sup.speaker in output_roles
        skipped_after_tool = _is_assistant_after_tool_response(
            sup, all_supervisions, idx, output_roles
        )
        text_preview = (sup.text or '')[:80]
        func_preview = function_content[:80] if function_content else ''

        included = (
            in_output_roles
            and not skipped_after_tool
            and (function_content == '' or is_tool_call)
        )

        logging.info(
            f"  sup[{idx}] speaker={sup.speaker}, start={sup.start:.2f}, "
            f"in_roles={in_output_roles}, is_toolcall={is_tool_call}, "
            f"is_tool_resp={is_tool_response}, "
            f"skipped_after_tool={skipped_after_tool}, "
            f"INCLUDED={included}, "
            f"text='{text_preview}{'...' if len(sup.text or '') > 80 else ''}'"
            + (f", func='{func_preview}{'...' if len(function_content) > 80 else ''}'" if func_preview else "")
        )
