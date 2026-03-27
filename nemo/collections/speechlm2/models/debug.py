import torch
from nemo.utils import logging
from typing import Optional


def log_slyned_refusal_batch(cuts) -> bool:
    """Log when a batch contains slyned refusal data.

    Detects slyned refusal cuts by their 'refusal_' ID prefix and logs
    batch details with ANSI colored [SLYNED REFUSAL] tags.

    Returns True if any slyned refusal cuts were found.
    """
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    TAG = f"{BOLD}{YELLOW}[SLYNED REFUSAL]{RESET}"

    slyned_cuts = [c for c in cuts if c.id.startswith("refusal_")]
    if not slyned_cuts:
        return False

    logging.info(f"{TAG} Sampled {len(slyned_cuts)}/{len(list(cuts))} slyned refusal cuts")
    for c in slyned_cuts[:3]:  # Log up to 3 examples
        speakers = set()
        toolcall_snippets = []
        for sup in c.supervisions:
            speakers.add(sup.speaker)
            text = sup.text or ""
            if "<TOOLCALL>" in text:
                toolcall_snippets.append(text[:120])
        sys_prompt = ""
        if c.supervisions and c.supervisions[0].speaker == "system":
            sys_prompt = (c.supervisions[0].text or "")[:150]
        logging.info(
            f"{TAG}   cut_id={c.id}, turns={len(c.supervisions)}, "
            f"speakers={speakers}, sys_prompt='{sys_prompt}...'"
        )
        for snippet in toolcall_snippets[:2]:
            logging.info(f"{TAG}     toolcall: {snippet}...")
    return True


def log_prefill_and_repeat_regions(
    text_labels: torch.Tensor,
    loss_scale: torch.Tensor,
    tokenizer,
    prefill_start_id: int,
    prefill_end_id: int,
    text_bos_id: int,
    text_eos_id: int,
    text_pad_id: int,
    text_weight: float = 1.0,
):
    """Log prefill and repeat regions with detokenized text and loss weights.

    For each sample in the batch, finds PREFILL_START...PREFILL_END regions
    and the subsequent agent_bos...agent_eos repeat regions. Logs the
    detokenized text and loss weights with ANSI colors for easy debugging.

    Colors:
      - Prefill region tags/weights: magenta / red
      - Repeat region tags/weights: cyan / green
    """
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"

    for i in range(text_labels.size(0)):
        pf_starts = (text_labels[i] == prefill_start_id).nonzero(as_tuple=True)[0].tolist()
        if not pf_starts:
            continue
        pf_ends = (text_labels[i] == prefill_end_id).nonzero(as_tuple=True)[0].tolist()
        agent_bos_positions = (text_labels[i] == text_bos_id).nonzero(as_tuple=True)[0].tolist()
        agent_eos_positions = (text_labels[i] == text_eos_id).nonzero(as_tuple=True)[0].tolist()

        for ps in pf_starts:
            # Find matching PREFILL_END
            pe = next((p for p in pf_ends if p > ps), None)
            if pe is None:
                logging.warning(
                    f"[FC prefill loss debug] sample {i}: "
                    f"PREFILL_START at {ps} has no matching PREFILL_END"
                )
                continue
            # Find matching agent_bos (first one after PREFILL_END)
            ab = next((p for p in agent_bos_positions if p > pe), None)
            # Find matching agent_eos (first one after agent_bos)
            ae = next((p for p in agent_eos_positions if p > (ab if ab else pe)), None)

            # --- Prefill region ---
            pf_labels = text_labels[i, ps:pe + 1].tolist()
            pf_weights = loss_scale[i, ps:pe + 1, 0].tolist()
            pf_text_ids = [
                t for t in pf_labels
                if t not in (prefill_start_id, prefill_end_id, text_pad_id)
            ]
            pf_text = tokenizer.ids_to_text(pf_text_ids) if pf_text_ids else "(empty)"
            logging.info(
                f"{MAGENTA}[FC prefill loss debug]{RESET} sample {i}, "
                f"prefill region [{ps}-{pe}]: "
                f"text='{pf_text}', "
                f"loss_weights={RED}[{', '.join(f'{w:.2f}' for w in pf_weights[:5])}"
                f"{'...' if len(pf_weights) > 5 else ''}]{RESET} "
                f"(should be all 0.0)"
            )

            # --- Repeat region ---
            # Find repeat end: scan from agent_bos until we hit pad or end of sequence
            # (agent_eos is no longer placed after repeat text; model emits pad instead)
            if ab is not None:
                repeat_end = ab + 1
                seq_len = text_labels.size(1)
                while repeat_end < seq_len and text_labels[i, repeat_end].item() not in (text_pad_id, text_eos_id, prefill_start_id):
                    repeat_end += 1
                # repeat_end now points to the first pad/eos/pf_start after repeat text

                rp_labels = text_labels[i, ab:repeat_end].tolist()
                rp_weights = loss_scale[i, ab:repeat_end, 0].tolist()
                rp_text_ids = [
                    t for t in rp_labels
                    if t not in (text_bos_id, text_eos_id, text_pad_id)
                ]
                rp_text = tokenizer.ids_to_text(rp_text_ids) if rp_text_ids else "(empty)"

                # Show one extra token after repeat to confirm it's pad, not agent_eos
                next_token_id = text_labels[i, repeat_end].item() if repeat_end < seq_len else -1
                next_token_str = tokenizer.ids_to_text([next_token_id]) if next_token_id >= 0 else "OUT_OF_BOUNDS"
                next_token_name = (
                    "pad" if next_token_id == text_pad_id else
                    "agent_eos" if next_token_id == text_eos_id else
                    f"id={next_token_id}('{next_token_str}')"
                )

                logging.info(
                    f"{CYAN}[FC repeat loss debug]{RESET} sample {i}, "
                    f"repeat region [{ab}-{repeat_end - 1}]: "
                    f"text='{rp_text}', "
                    f"loss_weights={GREEN}[{', '.join(f'{w:.2f}' for w in rp_weights[:5])}"
                    f"{'...' if len(rp_weights) > 5 else ''}]{RESET} "
                    f"(should be text_weight={text_weight}), "
                    f"next_token[{repeat_end}]={next_token_name}"
                )
            else:
                logging.warning(
                    f"[FC repeat loss debug] sample {i}: "
                    f"no agent_bos found after PREFILL_END at {pe}"
                )
