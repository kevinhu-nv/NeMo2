########################
# Eval script for turn-taking (TT), user back-channeling (BC), user barge-in (BI), Adapted from Chen Chen script

import torchaudio, torch
from tqdm import tqdm
import tarfile
import gzip
import json, os, re
import argparse
from nemo.collections import asr as nemo_asr

INF_LATENCY = 9999.0

def parse_float_list(arg):
    """
    Parse a string representation of a list of floats.
    Expected format: "[1,2,3]" or "1,2,3"
    """
    if arg.startswith('[') and arg.endswith(']'):
        arg = arg[1:-1]  # Remove brackets
    return [float(x.strip()) for x in arg.split(',')]


def parse_timestamped_text(text_with_timestamps):
    import re
    
    # Parse both BOS and EOS timestamps
    bos_pattern = r'<\|([\d\.]+)\|>'
    eos_pattern = r'<\$([\d\.]+)\$>'
    
    # Find all BOS and EOS markers with their positions
    bos_matches = list(re.finditer(bos_pattern, text_with_timestamps))
    eos_matches = list(re.finditer(eos_pattern, text_with_timestamps))
    
    bos_timestamps = [float(match.group(1)) for match in bos_matches]
    eos_timestamps = [float(match.group(1)) for match in eos_matches]
    
    # Convert timestamps to agent segments with text
    agent_segments = []
    
    # If we have both BOS and EOS timestamps, pair them up
    if bos_timestamps and eos_timestamps:
        for i, start_time in enumerate(bos_timestamps):
            # Find the corresponding EOS timestamp (next EOS after this BOS)
            end_time = None
            eos_idx = None
            for j, eos_time in enumerate(eos_timestamps):
                if eos_time > start_time:
                    end_time = eos_time
                    eos_idx = j
                    break
            
            # Extract text between BOS and EOS markers
            text = ""
            if end_time is not None and i < len(bos_matches) and eos_idx < len(eos_matches):
                bos_end_pos = bos_matches[i].end()
                eos_start_pos = eos_matches[eos_idx].start()
                text = text_with_timestamps[bos_end_pos:eos_start_pos].strip()
            
            if end_time is not None:
                agent_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
            else:
                # No corresponding EOS found, use default duration
                agent_segments.append({
                    'start': start_time,
                    'end': start_time + 5.0,
                    'text': text
                })
    
    # Fallback: if only BOS timestamps are available, add default value
    elif bos_timestamps:
        for i, timestamp in enumerate(bos_timestamps):
            # Extract text between this BOS and next BOS
            text = ""
            if i < len(bos_matches):
                bos_end_pos = bos_matches[i].end()
                if i < len(bos_matches) - 1:
                    next_bos_start_pos = bos_matches[i + 1].start()
                    text = text_with_timestamps[bos_end_pos:next_bos_start_pos].strip()
                else:
                    text = text_with_timestamps[bos_end_pos:].strip()
            
            if i < len(bos_timestamps) - 1:
                # Segment from current timestamp to next timestamp
                agent_segments.append({
                    'start': timestamp,
                    'end': bos_timestamps[i + 1],
                    'text': text
                })
            else:
                # Last segment - assume it lasts for a reasonable duration
                agent_segments.append({
                    'start': timestamp,
                    'end': timestamp + 5.0,
                    'text': text
                })
    
    return agent_segments


def load_jsonl(json_file, field_name='pred_text'):
    output = {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                audio_path = data.get('audio_path', '')
                pred_text = data.get(field_name, '')
                
                if audio_path and pred_text:
                    if '/' in audio_path:
                        filename = audio_path.split('/')[-1]  # Get filename
                        # Remove .wav extension if present
                        if filename.endswith('.wav'):
                            filename = filename[:-4]
                        # Remove "demo_" prefix if present
                        if filename.startswith('demo_'):
                            filename = filename[5:]  # Remove "demo_" (5 characters)
                    else:
                        filename = audio_path
                    
                    output[filename] = pred_text
    return output


def is_stopped_by_backchannel(agent_speech_segments, end_times, delay=0.99):
    """
    Check if agent's speech was interrupted by user backchanneling.
    
    Args:
        agent_speech_segments: List of dicts with 'start' and 'end' times of agent speech
        end_times: List of timestamps where backchanneling ends
        delay: Time delay to check if agent continues speaking (default: 0.99s)
    
    Returns:
        List of bools with same length as end_times, where each bool indicates if agent stopped speaking after that backchannel
    """
    if not end_times or not agent_speech_segments:
        return [False] * len(end_times)
        
    # Pre-sort speech timestamps for faster lookup
    agent_speech_segments = sorted(agent_speech_segments, key=lambda x: x['start'])
    
    bc_failure = []
    for t in end_times:
        if t == 0:  # Skip if no backchannel
            bc_failure.append(False)
            continue
            
        # User interrupts agent if
        # 1) Any end_time falls between a agent segment, and 
        # 2) Agent stops speaking after some delay
        overlapping = False
        for segment in agent_speech_segments:
            if t >= segment['start'] and t <= segment['end']:
                overlapping = True
                
                agent_delayed_stop_time = t + delay
                agent_still_speaking = any(
                    s['start'] <= agent_delayed_stop_time <= s['end'] 
                    for s in agent_speech_segments
                )
                is_interrupted = not agent_still_speaking
                bc_failure.append(is_interrupted)
                break
                
        if not overlapping:
            bc_failure.append(False)
            
    return bc_failure


def find_user_barge_ins(user_turns, agent_turns, threshold_seconds=0.5):
    i, j = 0, 0
    success_barge_ins = []
    failed_barge_ins = []
    
    while i < len(user_turns) and j < len(agent_turns):
        u_start, u_end = user_turns[i]['start'], user_turns[i]['end']
        a_start, a_end = agent_turns[j]['start'], agent_turns[j]['end']

        # Check if user started speaking during agent's turn
        if u_start > a_start and u_start < a_end:
            # overlap_start = max(u_start, a_start)
            # overlap_end = min(u_end, a_end)
            stop_duration_ms = round((a_end - u_start) * 1000)
            
            barge_in_info = {
                'stop_duration_ms': stop_duration_ms,
                'user': user_turns[i],
                'agent': agent_turns[j]
            }
            
            if stop_duration_ms < threshold_seconds * 1000:
                success_barge_ins.append(barge_in_info)
            else:
                failed_barge_ins.append(barge_in_info)

        if u_end < a_end:
            i += 1
        else:
            j += 1
            
    return success_barge_ins, failed_barge_ins


def init_vad_model():
    vad_model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    vad_model = vad_model.to('cuda')
    get_speech_timestamps, _, _, _, _ = utils
    return vad_model, get_speech_timestamps


def init_asr_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    """Initialize ASR model for transcription using NeMo."""
    print(f"Loading ASR model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).cuda()
    print("ASR model loaded successfully.")
    return asr_model


def transcribe_segment(audio, start_time, end_time, sample_rate, asr_model, temp_dir="/tmp"):
    """
    Transcribe a specific segment of audio using NeMo ASR.
    
    Args:
        audio: Audio tensor of shape (channels, samples)
        start_time: Start time in seconds
        end_time: End time in seconds
        sample_rate: Sample rate of the audio
        asr_model: NeMo ASR model
        temp_dir: Temporary directory for saving audio segments
    
    Returns:
        tuple: (transcribed_text, end_timestamp) where end_timestamp is the end time of last spoken word
    """
    import tempfile
    
    try:
        # Extract segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio = audio[:, start_sample:end_sample]
        
        # Skip very short segments
        if segment_audio.shape[1] < 160:  # Less than 10ms at 16kHz
            return "", 0.0
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            segment_audio = torchaudio.functional.resample(segment_audio, sample_rate, 16000)
            sample_rate = 16000
        
        # Convert to mono if stereo
        if segment_audio.shape[0] > 1:
            segment_audio = torch.mean(segment_audio, dim=0, keepdim=True)
        
        # Save to temporary file for NeMo API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_audio.cpu(), sample_rate)
        
        # Transcribe using NeMo API with timestamps
        asr_outputs = asr_model.transcribe([tmp_path], timestamps=True)
        
        # Clean up temporary file
        os.remove(tmp_path)
        
        if not asr_outputs:
            return "", 0.0
        
        # Get the transcription result
        result = asr_outputs[0]
        text = result.text if hasattr(result, 'text') else ""
        
        # Extract end timestamp of the last word
        end_timestamp = 0.0
        if hasattr(result, 'timestamp') and result.timestamp and 'word' in result.timestamp:
            word_timestamps = result.timestamp["word"]
            if word_timestamps and len(word_timestamps) > 0:
                last_word = word_timestamps[-1]
                if isinstance(last_word, dict) and 'end' in last_word:
                    end_timestamp = last_word['end']
        elif hasattr(result, 'end_time'):
            end_timestamp = result.end_time
        
        return text.strip(), end_timestamp
        
    except Exception as e:
        print(f"Error transcribing segment [{start_time:.3f}s - {end_time:.3f}s]: {str(e)}")
        return "", 0.0


def extract_segments_from_binary_audio(audio, sample_rate=16000):
    """
    Extract speech segments from binary audio where 1s indicate agent is active.
    
    Args:
        audio: Tensor of shape (1, num_samples) with values 0 or 1
        sample_rate: Sample rate of the audio (default: 16000)
    
    Returns:
        List of dicts with 'start' and 'end' times in seconds
    """
    # Flatten audio to 1D array
    audio_flat = audio.squeeze().cpu().numpy()
    
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, val in enumerate(audio_flat):
        if val > 0.5 and not in_segment:  # Start of active segment
            in_segment = True
            start_idx = i
        elif val <= 0.5 and in_segment:  # End of active segment
            in_segment = False
            segments.append({
                'start': start_idx / sample_rate,
                'end': i / sample_rate
            })
    
    # Handle case where audio ends while in a segment
    if in_segment:
        segments.append({
            'start': start_idx / sample_rate,
            'end': len(audio_flat) / sample_rate
        })
    
    return segments


def compute_barge_in_metrics(success_barge_ins, failed_barge_ins):
    """
    Compute barge-in metrics including success rate and counts.
    
    Args:
        success_barge_ins: List of successful barge-in events
        failed_barge_ins: List of failed barge-in events
    
    Returns:
        dict: Dictionary containing metrics including:
            - total_count: Total number of barge-ins
            - success_count: Number of successful barge-ins
            - success_rate: Success rate as percentage
            - has_barge_ins: Boolean indicating if any barge-ins were detected
            - avg_latency_ms: Average latency of successful barge-ins in milliseconds
    """
    total_barge_ins = len(success_barge_ins) + len(failed_barge_ins)
    success_count = len(success_barge_ins)
    
    metrics = {
        'total_count': total_barge_ins,
        'success_count': success_count,
        'has_barge_ins': total_barge_ins > 0
    }
    
    if metrics['has_barge_ins']:
        metrics['success_rate'] = (success_count / total_barge_ins) * 100
        
    if success_count > 0:
        metrics['avg_latency_ms'] = sum(bi['stop_duration_ms'] for bi in success_barge_ins) / success_count
    
    return metrics


def print_metrics(metrics_dict, verbose=False):
    """
    Print all evaluation metrics for a conversation.
    
    Args:
        metrics_dict: Dictionary containing all metrics including:
            - item_id: ID of the conversation
            - tt_latency: Turn-taking latency in seconds
            - tt_accuracy: Boolean indicating if turn-taking was accurate
            - barge_in_metrics: Dictionary containing barge-in related metrics
            - bc_result: List of backchanneling results
            - user_segments: List of user speech segments (if verbose=True)
            - agent_segments: List of agent speech segments (if verbose=True)
            - success_barge_ins: List of successful barge-in events (if verbose=True)
            - failed_barge_ins: List of failed barge-in events (if verbose=True)
            - user_transcripts: Dict mapping (start, end) to transcript (optional)
            - agent_transcripts: Dict mapping (start, end) to transcript (optional)
        verbose: If True, print detailed segment information
    """
    if not verbose:
        return
    # Build barge-in statistics section
    barge_in_stats = []
    if metrics_dict['barge_in_metrics']['has_barge_ins']:
        barge_in_stats.append(f"   - Barge-in success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})")
        if 'avg_latency_ms' in metrics_dict['barge_in_metrics']:
            barge_in_stats.append(f"   - Average barge-in latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f} ms")
    else:
        barge_in_stats.append("   - No barge-ins detected")

    # Build segment information section if verbose
    segment_info = ""
    if verbose:
        # Get transcripts if available
        user_transcripts = metrics_dict.get('user_transcripts', {})
        agent_transcripts = metrics_dict.get('agent_transcripts', {})
        
        # General speech segments - combine and sort by start time
        all_segments = []
        for seg in metrics_dict['user_segments']:
            all_segments.append({'type': 'User', 'start': seg['start'], 'end': seg['end']})
        for seg in metrics_dict['agent_segments']:
            all_segments.append({'type': 'Agent', 'start': seg['start'], 'end': seg['end']})
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Calculate response latencies for each user segment
        user_to_agent_latencies = {}
        for user_seg in metrics_dict['user_segments']:
            # Find the next agent segment that starts after this user segment ends
            next_agent = None
            min_latency = float('inf')
            for agent_seg in metrics_dict['agent_segments']:
                if agent_seg['start'] >= user_seg['end']:
                    latency = agent_seg['start'] - user_seg['end']
                    if latency < min_latency:
                        min_latency = latency
                        next_agent = agent_seg
            
            if next_agent:
                # Use a tuple of (start, end) as key to uniquely identify the segment
                user_to_agent_latencies[(user_seg['start'], user_seg['end'])] = min_latency
        
        # Format segments in chronological order with colors and latencies
        # ANSI color codes: \033[94m = Blue (User), \033[92m = Green (Agent), \033[0m = Reset
        def format_segment(seg):
            seg_key = (seg['start'], seg['end'])
            transcript = ""
            
            if seg['type'] == 'User':
                if seg_key in user_transcripts:
                    transcript = f" ({user_transcripts[seg_key]})"
                
                if seg_key in user_to_agent_latencies:
                    latency = user_to_agent_latencies[seg_key]
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s], \033[93m{latency:.3f}s\033[0m{transcript}"
                else:
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"
            else:  # Agent
                if seg_key in agent_transcripts:
                    transcript = f" ({agent_transcripts[seg_key]})"
                return f"   \033[92m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"
        
        segments_str = "\n".join(format_segment(seg) for seg in all_segments)
        
        # Also keep the old format for backwards compatibility
        user_segments_str = ", ".join(f"   [{seg['start']:.3f}s - {seg['end']:.3f}s]" for seg in metrics_dict['user_segments'])
        agent_segments_str = ", ".join(f"   [{seg['start']:.3f}s - {seg['end']:.3f}s]" for seg in metrics_dict['agent_segments'])
        
        # Barge-in segments
        barge_in_segments = []
        if metrics_dict['barge_in_metrics']['has_barge_ins']:
            if metrics_dict['success_barge_ins']:
                barge_in_segments.append("   Successful barge-ins:")
                for bi in metrics_dict['success_barge_ins']:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")
            
            if metrics_dict['failed_barge_ins']:
                barge_in_segments.append("   Failed barge-ins:")
                for bi in metrics_dict['failed_barge_ins']:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")

        segment_info = f"""
4. Speech segments (chronological order):
{segments_str}
5. Barge-in details:
{chr(10).join(barge_in_segments) if barge_in_segments else "   No barge-ins detected"}"""

    # Build the complete output string
    output = f"""
Evaluation metrics for conversation {metrics_dict['item_id']}:
1. Turn-taking metrics:
   - Average latency: {metrics_dict['tt_latency']:.3f} seconds
   - Precision: {metrics_dict['tt_precision']:.3f}
   - Recall: {metrics_dict['tt_recall']:.3f}
   - F1: {metrics_dict['tt_f1']:.3f}
2. Barge-in statistics:
{chr(10).join(barge_in_stats)}
3. Backchanneling failures: {metrics_dict['bc_failure']}{segment_info}
{'-' * 50}"""

    print(output)

def get_pred_audio_path(pred_audio_dir, item_id, dataset_name):
    # Find a file in pred_audio_dir where item_id is a substring of the filename and the filename ends with .wav
    for fname in os.listdir(pred_audio_dir):
        if item_id in fname and fname.endswith('.wav'):
            # Remove the .wav suffix from the filename
            fname_no_wav = fname[:-4] if fname.lower().endswith('.wav') else fname
            return os.path.join(pred_audio_dir, f"{dataset_name}_{item_id}.wav").strip()
    return None


def get_filtered_wav_keys(pred_audio_dir, validation_set_name):
    """
    Returns a set of keys for wav files in pred_audio_dir that start with validation_set_name.
    The key is the filename with the prefix and '.wav' removed.
    """
    wav_files = [f for f in os.listdir(pred_audio_dir) if f.startswith(validation_set_name) and f.endswith('.wav')]
    prefix_len = len(validation_set_name)
    filtered_wav_keys = set()
    for fname in wav_files:
        # Remove prefix
        name_wo_prefix = fname[prefix_len:]
        name_wo_prefix = name_wo_prefix[1:] if name_wo_prefix[0] == '_' else name_wo_prefix
        # Remove .wav suffix
        if name_wo_prefix.lower().endswith('.wav'):
            name_wo_prefix = name_wo_prefix[:-4]
        # Remove leading underscores or dashes if present
        if name_wo_prefix.startswith("_"):
            name_wo_prefix = name_wo_prefix[1:]
        elif name_wo_prefix.startswith("-"):
            name_wo_prefix = name_wo_prefix[1:]
        filtered_wav_keys.add(name_wo_prefix)
    return filtered_wav_keys    


def compute_turn_taking_metrics(agent_segments, user_segments, tt_latency_threshold_sec, tt_precision_buffer_sec, tt_recall_buffer_sec):
    """
    Compute turn-taking metrics using precision and recall.
    
    Args:
        agent_segments: List of dicts with 'start' and 'end' times of agent speech
        user_segments: List of dicts with 'start' and 'end' times of user speech
        tt_latency_threshold_sec: Threshold in seconds for considering turn-taking to be accurate
        tt_precision_buffer_sec: Buffer time in seconds for precision calculation (agent segments)
        tt_recall_buffer_sec: Buffer time in seconds for recall calculation (user segments)
    Returns:
        dict: Contains precision, recall, f1, and latency metrics
    """
    if not agent_segments or not user_segments:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'avg_latency': INF_LATENCY,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    # Calculate true positives (tp), false positives (fp), and false negatives (fn)
    tp = 0
    fp = 0
    fn = 0
    tp_latencies = []

    # For each agent segment, check if it follows any user segment end within the threshold
    for agent_seg in agent_segments:
        found_tp = False
        min_latency = INF_LATENCY
        
        for user_seg in user_segments:
            gap = agent_seg['start'] - user_seg['end']
            if (gap >= -tt_precision_buffer_sec and 
                gap <= tt_latency_threshold_sec and 
                agent_seg['start'] >= user_seg['start']):
                found_tp = True
                min_latency = min(min_latency, max(gap, 0))
        
        if found_tp:
            tp += 1
            tp_latencies.append(min_latency)
        else:
            fp += 1

    # For each user segment, check if there is any agent segment following its end within the threshold
    for user_seg in user_segments:
        found_tp = False
        for agent_seg in agent_segments:
            gap = agent_seg['start'] - user_seg['end']
            if gap >= -tt_recall_buffer_sec and gap <= tt_latency_threshold_sec:
                found_tp = True
                break
        if not found_tp:
            fn += 1

    # Compute precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average latency for true positives
    avg_latency = sum(tp_latencies) / len(tp_latencies) if tp_latencies else INF_LATENCY

    # Sanity checks
    num_user_segments = len(user_segments)
    num_agent_segments = len(agent_segments)
    
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"User segments: {num_user_segments}, Agent segments: {num_agent_segments}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"Average latency for TPs: {avg_latency:.3f}s")
    
    # Sanity check: tp + fn should equal number of user segments
    if tp + fn != num_user_segments:
        print(f"WARNING: tp + fn ({tp + fn}) != user segments ({num_user_segments})")
    
    # Sanity check: tp + fp should equal number of agent segments  
    if tp + fp != num_agent_segments:
        print(f"WARNING: tp + fp ({tp + fp}) != agent segments ({num_agent_segments})")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_latency': avg_latency,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def main(args):
    vad_model, get_speech_timestamps = init_vad_model()
    
    # Initialize ASR model for transcription if requested
    asr_model = None
    if args.enable_transcription:
        asr_model = init_asr_model(args.asr_model_name)

    manifest_dir = args.manifest_dir
    pred_audio_dir= args.pred_audio_dir

    # Load timestamped predictions if provided
    timestamped_preds = None
    if args.jsonl_with_timestamp:
        print(f"Loading timestamped predictions from: {args.jsonl_with_timestamp}")
        timestamped_preds = load_jsonl(args.jsonl_with_timestamp, field_name='pred_text')
        print(f"Loaded {len(timestamped_preds)} timestamped predictions")

    validation_set_names = getattr(args, "validation_set_names", None)
    if isinstance(validation_set_names, str):
        val_set_names = [name.strip() for name in validation_set_names.split(",") if name.strip()]

    for val_set_name in val_set_names:
        count = 0
        # Lists to store metrics for averaging
        all_tt_latencies = []
        all_tt_accuracies = []
        all_tt_precisions = []
        all_tt_recalls = []
        all_tt_f1s = []
        all_barge_in_success_rates = []
        all_barge_in_latencies = []
        all_bc_accuracies = []

        # Eval a specific validation set
        filtered_wav_keys = get_filtered_wav_keys(pred_audio_dir, val_set_name)

        for filtered_wav_key in filtered_wav_keys:
            pred_audio_file = get_pred_audio_path(pred_audio_dir, filtered_wav_key, val_set_name)
            if not os.path.exists(pred_audio_file):
                print(f"File not found: {pred_audio_file}")
                continue

            # Read user and agent audios
            if getattr(args, "is_stereo", False):
                # Load stereo audio: first channel is user, second is agent
                audio, audio_sr = torchaudio.load(pred_audio_file)
                if audio.shape[0] < 2:
                    raise ValueError(f"Expected stereo audio with 2 channels in {pred_audio_file}, but got {audio.shape[0]} channel(s).")
                user_audio, agent_audio = audio[0:1, :], audio[1:2, :]
                user_audio_sr = audio_sr
                agent_audio_sr = audio_sr
            else:
                # Assuming only agent audio, then read user audio from shar
                cuts_files = [f for f in os.listdir(manifest_dir) if f.startswith("cuts.") and f.endswith(".jsonl.gz")]
                recording_files = [f for f in os.listdir(manifest_dir) if f.startswith("recording.") and f.endswith(".tar")]
                cuts_files.sort()
                recording_files.sort()

                for cuts_file, recording_file in zip(cuts_files, recording_files):
                    cuts_path = os.path.join(manifest_dir, cuts_file)
                    recording_tar_path = os.path.join(manifest_dir, recording_file)

                    with tarfile.open(recording_tar_path, 'r') as recording_tar:
                        with gzip.open(cuts_path, 'rt', encoding='utf-8') as f:
                            # load from pred_audio_path
                            pred_audio_file = get_pred_audio_path(pred_audio_dir, filtered_wav_key, val_set_name)
                            if not os.path.exists(pred_audio_file):
                                print(f"File not found: {pred_audio_file}")
                                continue

                            # load from recording.000000.tar
                            json_file_name = f"{filtered_wav_key}.json"
                            flac_file_name = f"{filtered_wav_key}.flac"

                            if json_file_name in recording_tar.getnames() and flac_file_name in recording_tar.getnames():
                                json_member = recording_tar.getmember(json_file_name)
                                flac_member = recording_tar.getmember(flac_file_name)
                                with recording_tar.extractfile(flac_member) as flac_file:
                                    user_audio, user_audio_sr = torchaudio.load(flac_file)   # weqing 1
                            else:
                                print(f"Missing files for ID: {filtered_wav_key} in tar archive")                  

                            agent_audio, agent_audio_sr = torchaudio.load(pred_audio_file)

                            # Write user_audio out for debug
                            debug_user_audio_dir = os.path.join(os.path.dirname(pred_audio_file), "recordings")
                            os.makedirs(debug_user_audio_dir, exist_ok=True)
                            debug_user_audio_path = os.path.join(debug_user_audio_dir, f"{filtered_wav_key}.user.wav")
                            torchaudio.save(debug_user_audio_path, user_audio, 16000)

            # Run VAD to get the speech segments
            # Here min_silence_duration_ms is a important metric to control the tolerance of silence duration "---" in xxxxx---xxxxx, where xxxxx is the speech segment
            user_audio = torchaudio.functional.resample(user_audio, user_audio_sr, 16000)
            agent_audio = torchaudio.functional.resample(agent_audio, agent_audio_sr, 16000)
            
            # Use timestamped predictions for agent segments if available, otherwise use VAD or binary audio
            # Find the full key that contains filtered_wav_key as a substring
            matching_key = next((k for k in timestamped_preds.keys() if filtered_wav_key in k), None) if timestamped_preds else None
            
            if matching_key:
                print(f"Using timestamped text predictions for {matching_key}")
                timestamped_text = timestamped_preds[matching_key]
                agent_segments = parse_timestamped_text(timestamped_text)
                print(f"Parsed {len(agent_segments)} agent segments from timestamped text")
            elif args.agent_binary_audio:
                # Extract segments from binary audio (0s and 1s)
                print(f"Extracting agent segments from binary audio for {filtered_wav_key}")
                agent_segments = extract_segments_from_binary_audio(agent_audio, sample_rate=16000)
                print(f"Extracted {len(agent_segments)} agent segments from binary audio")
            else:
                # Fallback to VAD-based segmentation
                agent_vad_results = get_speech_timestamps(agent_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=args.vad_min_silence_duration_ms)
                agent_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in agent_vad_results]

            user_vad_results = get_speech_timestamps(user_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=args.vad_min_silence_duration_ms)
            user_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in user_vad_results]

            #################
            # Compute eval Metrics 
            # So far, we mesaure three types of conversation behaviors: Turn-taking, barge-in, and user backchanneling

            # Turn-taking
            tt_metrics = compute_turn_taking_metrics(agent_segments, user_segments, args.tt_latency_threshold_sec, args.tt_precision_buffer_sec, args.tt_recall_buffer_sec)
            tt_latency = tt_metrics['avg_latency']
            tt_accuracy = tt_metrics['f1']  # Use F1 as overall accuracy metric

            # Barge-in: find the overlap and return the overlap duration and segments
            success_barge_ins, failed_barge_ins = find_user_barge_ins(user_segments, agent_segments, args.barge_in_threshold_sec)
            barge_in_metrics = compute_barge_in_metrics(success_barge_ins, failed_barge_ins)

            # Backchaneling: find the backchanneling, end_time is the timestamp of backchannel end point
            end_time = args.end_time  # end time of backchanneling, note that this is predefined when creating backchanneling data
            if end_time is not None:
                bc_failure = is_stopped_by_backchannel(agent_segments, end_time, args.barge_in_threshold_sec)
            else:
                bc_failure = []

            # Store metrics for averaging
            all_tt_latencies.append(tt_latency)
            all_tt_accuracies.append(1 if tt_accuracy else 0)
            all_tt_precisions.append(tt_metrics['precision'])
            all_tt_recalls.append(tt_metrics['recall'])
            all_tt_f1s.append(tt_metrics['f1'])
            
            if barge_in_metrics['has_barge_ins']:
                all_barge_in_success_rates.append(barge_in_metrics['success_rate'])
                if 'avg_latency_ms' in barge_in_metrics:
                    all_barge_in_latencies.append(barge_in_metrics['avg_latency_ms'])
            
            # Calculate backchannel accuracy (percentage of successful backchannels)
            bc_accuracy = sum(1 for x in bc_failure if not x) / len(bc_failure) if bc_failure else 0
            all_bc_accuracies.append(bc_accuracy)

            # Transcribe/extract text for segments
            user_transcripts = {}
            agent_transcripts = {}
            
            # Extract agent text from timestamped predictions if available
            if timestamped_preds and matching_key:
                timestamped_text = timestamped_preds[matching_key]
                agent_segments_with_text = parse_timestamped_text(timestamped_text)
                # Convert list to dict with (start, end) tuples as keys
                agent_transcripts = {(seg['start'], seg['end']): seg.get('text', '') for seg in agent_segments_with_text}
            
            # Transcribe user segments with ASR if enabled
            if asr_model is not None:
                print(f"Transcribing {len(user_segments)} user segments...")
                
                # Transcribe user segments
                for seg in user_segments:
                    transcript, end_ts = transcribe_segment(
                        user_audio, seg['start'], seg['end'], 
                        16000, asr_model
                    )
                    user_transcripts[(seg['start'], seg['end'])] = transcript

            # Collect all metrics in a dictionary
            metrics_dict = {
                'item_id': filtered_wav_key,
                'tt_latency': tt_latency,
                'tt_accuracy': tt_accuracy,
                'tt_precision': tt_metrics['precision'],
                'tt_recall': tt_metrics['recall'],
                'tt_f1': tt_metrics['f1'],
                'barge_in_metrics': barge_in_metrics,
                'bc_failure': bc_failure,
                'user_segments': user_segments,
                'agent_segments': agent_segments,
                'success_barge_ins': success_barge_ins,
                'failed_barge_ins': failed_barge_ins,
                'user_transcripts': user_transcripts,
                'agent_transcripts': agent_transcripts
            }

            # Print all metrics
            print_metrics(metrics_dict, verbose=args.verbose)
            
            count += 1

        # Compute and print average metrics
        _valid_tt_latencies = [x for x in all_tt_latencies if x != INF_LATENCY]
        avg_metrics = {
            'avg_tt_latency': sum(_valid_tt_latencies) / len(_valid_tt_latencies) if _valid_tt_latencies else 0,
            'avg_tt_accuracy': sum(all_tt_accuracies) / len(all_tt_accuracies) * 100 if all_tt_accuracies else 0,
            'avg_tt_precision': sum(all_tt_precisions) / len(all_tt_precisions) * 100 if all_tt_precisions else 0,
            'avg_tt_recall': sum(all_tt_recalls) / len(all_tt_recalls) * 100 if all_tt_recalls else 0,
            'avg_tt_f1': sum(all_tt_f1s) / len(all_tt_f1s) * 100 if all_tt_f1s else 0,
            'avg_barge_in_success_rate': sum(all_barge_in_success_rates) / len(all_barge_in_success_rates) if all_barge_in_success_rates else 0,
            'avg_barge_in_latency': sum(all_barge_in_latencies) / len(all_barge_in_latencies) if all_barge_in_latencies else 0,
            'avg_bc_accuracy': sum(all_bc_accuracies) / len(all_bc_accuracies) * 100 if all_bc_accuracies else 0,
            'num_audios_evaluated': count,
        }

        avg_metrics_str = f"""
    {'=' * 50}
    Average Metrics for \033[92m{val_set_name}\033[0m:
    1. Turn-taking:
    - Average latency: {avg_metrics['avg_tt_latency'] * 1000:.1f} ms
    - Precision: {avg_metrics['avg_tt_precision']:.1f}%
    - Recall: {avg_metrics['avg_tt_recall']:.1f}%
    - F1: {avg_metrics['avg_tt_f1']:.1f}%
    2. User barge-in:
    - Average success rate: {avg_metrics['avg_barge_in_success_rate']:.1f}%
    - Average latency: {avg_metrics['avg_barge_in_latency']:.1f} ms
    3. Back-channeling:
    - Average accuracy: {avg_metrics['avg_bc_accuracy']:.1f}%
    4. Number of audios evaluated: {avg_metrics['num_audios_evaluated']}
    {'=' * 50}"""

        print(avg_metrics_str)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_audio_dir", type=str, default="/lustre/fsw/portfolios/convai/users/cchen1/results/s2s_rl/uc_samples")
    parser.add_argument("--manifest_dir", type=str, default=None, required=False, help="Path to manifest directory. Optional.")
    parser.add_argument("--barge_in_threshold_sec", type=float, default=1.5, help="Buffering time for the agent to stop after user barges in.")
    parser.add_argument("--tt_latency_threshold_sec", type=float, default=0.64, help="Threshold in seconds for considering a turn-taking to be accurate.")
    parser.add_argument("--tt_precision_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for precision calculation (agent segments).")
    parser.add_argument("--tt_recall_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for recall calculation (user segments).")
    parser.add_argument(
        "--end_time",
        type=lambda x: None if x is None or x.lower() == "none" else parse_float_list(x),
        default=[1, 10, 15, 20],
        help="End time of backchanneling. Format: '[1,10,15,20]' or '1,10,15,20' or None"
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed segment information")
    parser.add_argument(
        "--validation_set_names",
        type=str,
        default=None,
        help="Prefix of the wav file in the pred_audio_dir to filter for a specific validation set."
    )
    parser.add_argument("--is_stereo", action="store_true", default=True, help="Whether the audio is stereo.")
    parser.add_argument("--jsonl_with_timestamp", type=str, default=None, help="Path to JSONL file with timestamped text predictions. Each line should have 'pred_text' field with text containing <|timestamp|> markers.")
    parser.add_argument("--vad_min_silence_duration_ms", type=int, default=1500, help="Minimum silence duration in milliseconds for VAD.")
    parser.add_argument("--agent_binary_audio", action="store_true", default=False, help="Whether the agent audio is binary (0s and 1s) indicating active/inactive segments instead of actual speech audio.")
    parser.add_argument("--enable_transcription", action="store_true", default=True, help="Enable transcription of user segments using ASR model. Agent text is automatically extracted from --jsonl_with_timestamp if provided.")
    parser.add_argument("--asr_model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="Name of the ASR model to use for transcription of user segments.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)