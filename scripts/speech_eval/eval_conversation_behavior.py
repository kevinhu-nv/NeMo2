########################
# Eval script for turn-taking (TT), user back-channeling (BC), user barge-in (BI), Adapted from Chen Chen script

import torchaudio, torch
from tqdm import tqdm
import tarfile
import gzip
import json, os, re
import argparse
import traceback
import tempfile
from nemo.collections import asr as nemo_asr
from openai import OpenAI
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer
from lhotse import CutSet

INF_LATENCY = 9999.0

def parse_float_list(arg):
    """
    Parse a string representation of a list of floats.
    Expected format: "[1,2,3]" or "1,2,3"
    """
    if arg.startswith('[') and arg.endswith(']'):
        arg = arg[1:-1]  # Remove brackets
    return [float(x.strip()) for x in arg.split(',')]


def remove_special_symbols(text):
    """
    Remove special symbols like <SPECIAL_12> from text.
    """
    import re
    # Remove patterns like <SPECIAL_12>, <SPECIAL_1>, etc.
    text = re.sub(r'<SPECIAL_\d+>', '', text)
    return text.strip()


def clean_predicted_text(text):
    """
    Clean predicted text by removing timestamps and special markers.
    """
    import re
    # Remove all caret characters
    text = text.replace('^', '')
    # Remove all occurrences of <|...|>
    text = re.sub(r'<\|.*?\|>', '', text)
    # Remove all occurrences of <$...$>
    text = re.sub(r'<\$.*?\$>', '', text)
    # Remove all occurrences of <SPECIAL_12>
    text = text.replace('<SPECIAL_12>', '')
    return text.strip()


def compute_wer_for_text(reference, hypothesis, normalizer=None):
    """
    Compute WER between reference and hypothesis text.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalizer: Optional text normalizer (e.g., EnglishTextNormalizer)
    
    Returns:
        WER as a float (0.0 to 1.0+)
    """
    if not reference or not hypothesis:
        return 0.0
    
    try:
        # Normalize texts if normalizer is provided
        if normalizer:
            reference = normalizer(reference)
            hypothesis = normalizer(hypothesis)
        
        wer_score = wer(reference, hypothesis)
        return wer_score
    except Exception as e:
        print(f"Error computing WER: {e}")
        return 0.0


def parse_user_timestamped_text(text_with_timestamps):
    """
    Parse user timestamped text to extract segments.
    
    Args:
        text_with_timestamps: Text with <|timestamp|> BOS and <$timestamp$> EOS markers for user speech
                              Example: ' <|1.12|> hello what is the best way <$6.96$>  <|11.36|> can you provide <$15.84$>'
    
    Returns:
        List of user segments with 'start', 'end', and 'text' fields
    """
    import re
    
    # Parse both BOS and EOS timestamps for user speech
    bos_pattern = r'<\|([\d\.]+)\|>'
    eos_pattern = r'<\$([\d\.]+)\$>'
    
    # Find all BOS and EOS markers with their positions
    bos_matches = list(re.finditer(bos_pattern, text_with_timestamps))
    eos_matches = list(re.finditer(eos_pattern, text_with_timestamps))
    
    bos_timestamps = [float(match.group(1)) for match in bos_matches]
    eos_timestamps = [float(match.group(1)) for match in eos_matches]
    
    user_segments = []
    
    # Pair up BOS and EOS timestamps
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
                user_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
    
    return user_segments


def parse_timestamped_text(text_with_timestamps, estimate_sec_per_word=0.3, cutoff_threshold=2.0, user_segments=None, gap_threshold=2.0):
    """
    Parse timestamped text and detect potentially cut-off agent segments.
    
    Args:
        text_with_timestamps: Text with <|timestamp|> BOS and <$timestamp$> EOS markers
        estimate_sec_per_word: Estimated seconds per word for duration calculation (default: 0.3)
        cutoff_threshold: Threshold in seconds to mark segment as cut off 
                          If estimated_duration - actual_duration > cutoff_threshold, mark as cut off
        user_segments: List of user segments with 'start' and 'end' times (optional)
        gap_threshold: Minimum gap in seconds between estimated end and next user start to consider cutoff (default: 0.5)
    
    Returns:
        List of agent segments with 'start', 'end', 'text', 'estimated_duration', and 'is_cutoff' fields
    """
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
            
            # Calculate estimated duration based on word count
            cleaned_text = remove_special_symbols(text)
            words = cleaned_text.split()
            estimated_duration = len(words) * estimate_sec_per_word if words else 0.0
            
            if end_time is not None:
                actual_duration = end_time - start_time
                estimated_end = start_time + estimated_duration
                
                # Condition 1: Estimated duration is significantly longer than actual duration
                is_cutoff_by_duration = estimated_duration - actual_duration > cutoff_threshold if estimated_duration > 0 else False
                
                # Condition 2: Estimated end time has a big gap to the following user start time
                # This means agent was not barged in by user, but was cut off by the end of the audio
                is_cutoff_by_gap = False
                if user_segments and estimated_end > end_time:
                    # Find the next user segment after this agent segment start time
                    next_user_segments = [u for u in user_segments if u['start'] > start_time]
                    if next_user_segments:
                        next_user_start = min(u['start'] for u in next_user_segments)
                        # If estimated end + gap_threshold < next user start, agent had room to continue
                        if estimated_end + gap_threshold < next_user_start:
                            is_cutoff_by_gap = True
                
                is_cutoff = is_cutoff_by_duration and is_cutoff_by_gap
                
                agent_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'estimated_duration': estimated_duration,
                    'is_cutoff': is_cutoff,
                    'is_cutoff_by_duration': is_cutoff_by_duration,
                    'is_cutoff_by_gap': is_cutoff_by_gap
                })
            else:
                # No corresponding EOS found, use default duration
                agent_segments.append({
                    'start': start_time,
                    'end': start_time + 5.0,
                    'text': text,
                    'estimated_duration': estimated_duration,
                    'is_cutoff': False,  # Can't determine if cut off without EOS
                    'is_cutoff_by_duration': False,
                    'is_cutoff_by_gap': False
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
            
            # Calculate estimated duration based on word count
            cleaned_text = remove_special_symbols(text)
            words = cleaned_text.split()
            estimated_duration = len(words) * estimate_sec_per_word if words else 0.0
            
            if i < len(bos_timestamps) - 1:
                # Segment from current timestamp to next timestamp
                end_time = bos_timestamps[i + 1]
                actual_duration = end_time - timestamp
                estimated_end = timestamp + estimated_duration
                
                # Condition 1: Estimated duration is significantly longer than actual duration
                is_cutoff_by_duration = estimated_duration - actual_duration > cutoff_threshold if estimated_duration > 0 else False
                
                # Condition 2: Estimated end time has a big gap to the following user start time
                is_cutoff_by_gap = False
                if user_segments and estimated_end > end_time:
                    next_user_segments = [u for u in user_segments if u['start'] > end_time]
                    if next_user_segments:
                        next_user_start = min(u['start'] for u in next_user_segments)
                        if estimated_end + gap_threshold < next_user_start:
                            is_cutoff_by_gap = True
                
                is_cutoff = is_cutoff_by_duration or is_cutoff_by_gap
                
                agent_segments.append({
                    'start': timestamp,
                    'end': end_time,
                    'text': text,
                    'estimated_duration': estimated_duration,
                    'is_cutoff': is_cutoff,
                    'is_cutoff_by_duration': is_cutoff_by_duration,
                    'is_cutoff_by_gap': is_cutoff_by_gap
                })
            else:
                # Last segment - assume it lasts for a reasonable duration
                agent_segments.append({
                    'start': timestamp,
                    'end': timestamp + 5.0,
                    'text': text,
                    'estimated_duration': estimated_duration,
                    'is_cutoff': False,  # Can't determine if cut off without next segment
                    'is_cutoff_by_duration': False,
                    'is_cutoff_by_gap': False
                })
    
    return agent_segments


def load_jsonl(json_file, field_name='pred_text', additional_fields=None):
    """
    Load JSONL file and extract specified fields.
    
    Args:
        json_file: Path to JSONL file
        field_name: Primary field to extract (default: 'pred_text')
        additional_fields: List of additional field names to extract (optional)
    
    Returns:
        If additional_fields is None: dict mapping filename to field value
        If additional_fields is provided: dict mapping filename to dict of {field_name: value, additional_field1: value1, ...}
    """
    output = {}
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                audio_path = data.get('audio_path', '')
                id = data.get('id', '')
                pred_text = data.get(field_name, '')
                
                if audio_path or id:
                    if '/' in audio_path:
                        filename = audio_path.split('/')[-1]  # Get filename
                        # Remove .wav extension if present
                        if filename.endswith('.wav'):
                            filename = filename[:-4]
                        # Remove "demo_" prefix if present
                        if filename.startswith('demo_'):
                            filename = filename[5:]  # Remove "demo_" (5 characters)
                    elif id:
                        filename = id
                    else:
                        filename = audio_path
                    
                    # If additional fields are requested, store as dict
                    if additional_fields:
                        output[filename] = {field_name: pred_text}
                        for field in additional_fields:
                            output[filename][field] = data.get(field, '')
                    else:
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


def print_detailed_utterance(metrics_dict):
    """
    Print detailed information for a single utterance including all segments and metrics.
    
    Args:
        metrics_dict: Dictionary containing all metrics for the utterance
    """
    user_transcripts = metrics_dict.get('user_transcripts', {})
    agent_transcripts = metrics_dict.get('agent_transcripts', {})
    predicted_user_transcripts = metrics_dict.get('predicted_user_transcripts', {})
    user_wer_scores = metrics_dict.get('user_wer_scores', {})
    
    # Build header
    print(f"\n{'=' * 80}")
    print(f"Utterance: {metrics_dict['item_id']}")
    print(f"{'=' * 80}")
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  Turn-taking:")
    print(f"    - Precision: {metrics_dict['tt_precision']:.3f}")
    print(f"    - Recall: {metrics_dict['tt_recall']:.3f}")
    print(f"    - F1: {metrics_dict['tt_f1']:.3f}")
    print(f"    - Latency: {metrics_dict['tt_latency']:.3f}s ({metrics_dict['tt_latency']*1000:.1f}ms)")
    
    if metrics_dict['barge_in_metrics']['has_barge_ins']:
        print(f"  Barge-in:")
        print(f"    - Success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})")
        if 'avg_latency_ms' in metrics_dict['barge_in_metrics']:
            print(f"    - Average latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f}ms")
    else:
        print(f"  Barge-in: No barge-ins detected")
    
    if 'cutoff_count' in metrics_dict and metrics_dict['total_agent_segments_with_text'] > 0:
        print(f"  Cutoff detection:")
        print(f"    - Cutoff rate: {metrics_dict['cutoff_rate']:.1f}% ({metrics_dict['cutoff_count']}/{metrics_dict['total_agent_segments_with_text']})")
    
    if 'gpt_scores' in metrics_dict and metrics_dict['gpt_scores']:
        gpt_scores = metrics_dict['gpt_scores']
        avg_gpt_score = sum(s.get('avg_score', 0.0) for s in gpt_scores) / len(gpt_scores)
        print(f"  GPT Quality Score:")
        print(f"    - Average: {avg_gpt_score:.2f}/5.0 ({len(gpt_scores)} pairs)")
    
    if 'user_eou_metrics' in metrics_dict and metrics_dict['user_eou_metrics']:
        user_eou = metrics_dict['user_eou_metrics']
        print(f"  User EOU Detection:")
        print(f"    - Precision: {user_eou['precision']:.3f}")
        print(f"    - Recall: {user_eou['recall']:.3f}")
        print(f"    - F1: {user_eou['f1']:.3f}")
        print(f"    - Avg EOU latency: {user_eou['avg_eou_latency']:.3f}s ({user_eou['avg_eou_latency']*1000:.1f}ms)")
        print(f"    - Early/Late/On-time: {user_eou['early_eou_count']}/{user_eou['late_eou_count']}/{user_eou['true_positives'] - user_eou['early_eou_count'] - user_eou['late_eou_count']}")
    
    # Print all segments in chronological order
    print(f"\nConversation flow:")
    all_segments = []
    for seg in metrics_dict['user_segments']:
        all_segments.append({'type': 'User', 'start': seg['start'], 'end': seg['end']})
    for seg in metrics_dict['agent_segments']:
        all_segments.append({'type': 'Agent', 'start': seg['start'], 'end': seg['end']})
    
    all_segments.sort(key=lambda x: x['start'])
    
    # Create a lookup for GPT scores by agent segment
    gpt_scores_by_agent = {}
    if 'gpt_scores' in metrics_dict:
        for gpt_score_info in metrics_dict['gpt_scores']:
            agent_seg = gpt_score_info['agent_segment']
            agent_key = (agent_seg['start'], agent_seg['end'])
            gpt_scores_by_agent[agent_key] = gpt_score_info
    
    for seg in all_segments:
        seg_key = (seg['start'], seg['end'])
        duration = seg['end'] - seg['start']
        
        if seg['type'] == 'User':
            transcript = user_transcripts.get(seg_key, '')
            predicted_transcript = predicted_user_transcripts.get(seg_key, '')
            wer_score = user_wer_scores.get(seg_key, None)
            
            wer_info = ""
            if wer_score is not None:
                color_code = "\033[91m" if wer_score > 0.3 else "\033[93m" if wer_score > 0.1 else "\033[92m"
                wer_info = f" {color_code}[WER: {wer_score:.1%}]\033[0m"
            
            if transcript:
                transcript_info = f": GT: '{transcript}'"
                if predicted_transcript:
                    transcript_info += f" | Pred: '{predicted_transcript}'"
                print(f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s){wer_info}{transcript_info}")
            else:
                print(f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s){wer_info}")
        else:  # Agent
            seg_info = agent_transcripts.get(seg_key, None)
            gpt_score_info = gpt_scores_by_agent.get(seg_key, None)
            
            if seg_info:
                # seg_info can be either a dict (new format) or string (old format for backward compatibility)
                if isinstance(seg_info, dict):
                    text = seg_info.get('text', '')
                    cleaned_text = remove_special_symbols(text)
                    estimated_dur = seg_info.get('estimated_duration', 0.0)
                    is_cutoff = seg_info.get('is_cutoff', False)
                    is_cutoff_by_duration = seg_info.get('is_cutoff_by_duration', False)
                    is_cutoff_by_gap = seg_info.get('is_cutoff_by_gap', False)
                    
                    cutoff_warning = ""
                    if is_cutoff:
                        if is_cutoff_by_duration and is_cutoff_by_gap:
                            cutoff_warning = " \033[91m[CUTOFF: duration+gap]\033[0m"
                        elif is_cutoff_by_duration:
                            cutoff_warning = " \033[91m[CUTOFF: duration]\033[0m"
                        elif is_cutoff_by_gap:
                            cutoff_warning = " \033[91m[CUTOFF: gap]\033[0m"
                    est_info = f" [est. {estimated_dur:.2f}s]" if estimated_dur > 0 else ""
                    gpt_info = ""
                    if gpt_score_info:
                        avg_score = gpt_score_info.get('avg_score', 0.0)
                        user_seg = gpt_score_info.get('user_segment', {})
                        user_start = user_seg.get('start', 0.0)
                        # Use red color for scores <= 2, yellow for others
                        color_code = "\033[91m" if avg_score <= 2.0 else "\033[93m"
                        gpt_info = f" {color_code}[GPT: {avg_score:.2f}/5.0 from User: {user_start:.3f}s]\033[0m"
                    print(f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s){est_info}{cutoff_warning}{gpt_info}: {cleaned_text}")
                else:
                    # Backward compatibility: seg_info is just the text string
                    cleaned_text = remove_special_symbols(seg_info)
                    gpt_info = ""
                    if gpt_score_info:
                        avg_score = gpt_score_info.get('avg_score', 0.0)
                        user_seg = gpt_score_info.get('user_segment', {})
                        user_start = user_seg.get('start', 0.0)
                        # Use red color for scores <= 2, yellow for others
                        color_code = "\033[91m" if avg_score <= 2.0 else "\033[93m"
                        gpt_info = f" {color_code}[GPT: {avg_score:.2f}/5.0 from User: {user_start:.3f}s]\033[0m"
                    print(f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s){gpt_info}: {cleaned_text}")
            else:
                print(f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s)")
    
    # Print barge-in details if any
    if metrics_dict['barge_in_metrics']['has_barge_ins']:
        print(f"\nBarge-in events:")
        if metrics_dict['success_barge_ins']:
            print(f"  Successful ({len(metrics_dict['success_barge_ins'])}):")
            for bi in metrics_dict['success_barge_ins']:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent stopped in {bi['stop_duration_ms']:.1f}ms")
        
        if metrics_dict['failed_barge_ins']:
            print(f"  Failed ({len(metrics_dict['failed_barge_ins'])}):")
            for bi in metrics_dict['failed_barge_ins']:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent took {bi['stop_duration_ms']:.1f}ms to stop (too slow)")
    
    print(f"{'=' * 80}\n")


def print_bottom_percentile_utterances(all_metrics, percentile=5):
    """
    Print utterances in the bottom percentile for barge-in accuracy and turn-taking recall.
    
    Args:
        all_metrics: List of metrics dictionaries for all utterances
        percentile: The percentile threshold (default 5 for bottom 5%)
    """
    import numpy as np
    
    if not all_metrics:
        print("No metrics to analyze.")
        return
    
    print(f"\n{'#' * 80}")
    print(f"BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")
    
    # Extract metrics for percentile calculation
    # For barge-in accuracy, only consider utterances that have barge-ins
    utterances_with_barge_ins = [m for m in all_metrics if m['barge_in_metrics']['has_barge_ins']]
    barge_in_rates = [m['barge_in_metrics']['success_rate'] for m in utterances_with_barge_ins]
    
    # For turn-taking recall, consider all utterances
    tt_recalls = [m['tt_recall'] for m in all_metrics]
    
    # Calculate percentile thresholds
    if barge_in_rates:
        barge_in_threshold = np.percentile(barge_in_rates, percentile)
        print(f"Barge-in success rate {percentile}th percentile threshold: {barge_in_threshold:.1f}%")
        print(f"  (Based on {len(utterances_with_barge_ins)} utterances with barge-ins)\n")
    else:
        barge_in_threshold = None
        print(f"No utterances with barge-ins detected.\n")
    
    tt_recall_threshold = np.percentile(tt_recalls, percentile)
    print(f"Turn-taking recall {percentile}th percentile threshold: {tt_recall_threshold:.3f}")
    print(f"  (Based on {len(all_metrics)} utterances)\n")
    
    # Find utterances below thresholds
    low_barge_in_utterances = []
    if barge_in_threshold is not None:
        low_barge_in_utterances = [
            m for m in utterances_with_barge_ins 
            if m['barge_in_metrics']['success_rate'] < barge_in_threshold
        ]
        # Sort by barge-in success rate (lowest first)
        low_barge_in_utterances.sort(key=lambda m: m['barge_in_metrics']['success_rate'])
    
    low_tt_recall_utterances = [
        m for m in all_metrics 
        if m['tt_recall'] < tt_recall_threshold
    ]
    # Sort by turn-taking recall (lowest first)
    low_tt_recall_utterances.sort(key=lambda m: m['tt_recall'])
    
    # Print low barge-in accuracy utterances
    if low_barge_in_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW BARGE-IN SUCCESS RATE (< {barge_in_threshold:.1f}%)")
        print(f"Found {len(low_barge_in_utterances)} utterance(s)")
        print(f"{'-' * 80}")
        
        for m in low_barge_in_utterances:
            print_detailed_utterance(m)
    
    # Print low turn-taking recall utterances
    if low_tt_recall_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW TURN-TAKING RECALL (< {tt_recall_threshold:.3f})")
        print(f"Found {len(low_tt_recall_utterances)} utterance(s)")
        print(f"{'-' * 80}")
        
        for m in low_tt_recall_utterances:
            print_detailed_utterance(m)
    
    print(f"\n{'#' * 80}")
    print(f"END OF BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")


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
        predicted_user_transcripts = metrics_dict.get('predicted_user_transcripts', {})
        user_wer_scores = metrics_dict.get('user_wer_scores', {})
        
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
        
        # Create GPT score lookup by agent segment
        gpt_scores_by_agent = {}
        if 'gpt_scores' in metrics_dict:
            for gpt_score_info in metrics_dict['gpt_scores']:
                agent_seg = gpt_score_info['agent_segment']
                agent_key = (agent_seg['start'], agent_seg['end'])
                gpt_scores_by_agent[agent_key] = gpt_score_info
        
        # Format segments in chronological order with colors and latencies
        # ANSI color codes: \033[94m = Blue (User), \033[92m = Green (Agent), \033[0m = Reset
        def format_segment(seg):
            seg_key = (seg['start'], seg['end'])
            transcript = ""
            
            if seg['type'] == 'User':
                wer_score = user_wer_scores.get(seg_key, None)
                wer_info = ""
                if wer_score is not None:
                    color_code = "\033[91m" if wer_score > 0.3 else "\033[93m" if wer_score > 0.1 else "\033[92m"
                    wer_info = f" {color_code}[WER: {wer_score:.1%}]\033[0m"
                
                if seg_key in user_transcripts:
                    transcript = f" (GT: {user_transcripts[seg_key]})"
                    if seg_key in predicted_user_transcripts:
                        transcript += f" (Pred: {predicted_user_transcripts[seg_key]})"
                
                if seg_key in user_to_agent_latencies:
                    latency = user_to_agent_latencies[seg_key]
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s], \033[93m{latency:.3f}s\033[0m{wer_info}{transcript}"
                else:
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{wer_info}{transcript}"
            else:  # Agent
                gpt_score_info = gpt_scores_by_agent.get(seg_key, None)
                gpt_info = ""
                if gpt_score_info:
                    avg_score = gpt_score_info.get('avg_score', 0.0)
                    user_seg = gpt_score_info.get('user_segment', {})
                    user_start = user_seg.get('start', 0.0)
                    # Use red color for scores <= 2, yellow for others
                    color_code = "\033[91m" if avg_score <= 2.0 else "\033[93m"
                    gpt_info = f" {color_code}[GPT: {avg_score:.2f}/5.0 from User: {user_start:.3f}s]\033[0m"
                
                if seg_key in agent_transcripts:
                    seg_info = agent_transcripts[seg_key]
                    # seg_info can be either a dict (new format) or string (old format for backward compatibility)
                    if isinstance(seg_info, dict):
                        text = seg_info.get('text', '')
                        cleaned_text = remove_special_symbols(text)
                        estimated_duration = seg_info.get('estimated_duration', 0.0)
                        is_cutoff = seg_info.get('is_cutoff', False)
                        
                        cutoff_warning = " \033[91m[CUTOFF]\033[0m" if is_cutoff else ""
                        transcript = f" ({cleaned_text}) [\033[95mest. {estimated_duration:.2f}s\033[0m]{cutoff_warning}{gpt_info}"
                    else:
                        # Backward compatibility: seg_info is just the text string
                        cleaned_text = remove_special_symbols(seg_info)
                        words = cleaned_text.split()
                        estimate_sec_per_word = 0.3
                        estimated_duration = len(words) * estimate_sec_per_word
                        transcript = f" ({cleaned_text}) [\033[95mest. {estimated_duration:.2f}s\033[0m]{gpt_info}"
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

        # EOU metrics
        eou_info = ""
        if 'user_eou_metrics' in metrics_dict and metrics_dict['user_eou_metrics']:
            user_eou = metrics_dict['user_eou_metrics']
            on_time_count = user_eou['true_positives'] - user_eou['early_eou_count'] - user_eou['late_eou_count']
            eou_info = f"""
6. User EOU Detection:
   - Precision: {user_eou['precision']:.3f}
   - Recall: {user_eou['recall']:.3f}
   - F1: {user_eou['f1']:.3f}
   - Avg EOU latency: {user_eou['avg_eou_latency']:.3f}s ({user_eou['avg_eou_latency']*1000:.1f}ms)
   - Early/Late/On-time: {user_eou['early_eou_count']}/{user_eou['late_eou_count']}/{on_time_count}"""

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
3. Backchanneling failures: {metrics_dict['bc_failure']}{segment_info}{eou_info}
{'-' * 50}"""

    print(output)

def get_pred_audio_path(pred_audio_dir, item_id, dataset_name):
    # Find a file in pred_audio_dir where item_id is a substring of the filename and the filename ends with .wav
    for fname in os.listdir(pred_audio_dir):
        if item_id in fname and fname.endswith('.wav') and 'dup' not in fname:
            # Remove the .wav suffix from the filename
            fname_no_wav = fname[:-4] if fname.lower().endswith('.wav') else fname
            return os.path.join(pred_audio_dir, f"{dataset_name}_{item_id}.wav").strip()
        elif 'dup' in fname:
            print(f"Skip duplicate file: {fname}")
    return None


def get_filtered_wav_keys(pred_audio_dir, validation_set_name):
    """
    Returns a set of keys for wav files in pred_audio_dir that start with validation_set_name.
    The key is the filename with the prefix and '.wav' removed.
    """
    import re
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


def get_filtered_keys_from_shar(shar_input_dir, validation_set_name=None):
    """
    Returns a set of keys (cut IDs) from shar cuts files.
    The validation_set_name parameter is kept for API compatibility but not used for filtering,
    as shar cut IDs don't have validation set prefixes.
    """
    cuts_files = sorted([f for f in os.listdir(shar_input_dir) if f.startswith("cuts.") and f.endswith(".jsonl.gz")])
    filtered_keys = set()
    
    for cuts_file in cuts_files:
        cuts_path = os.path.join(shar_input_dir, cuts_file)
        try:
            # Load the CutSet from shar
            cutset = CutSet.from_jsonl_lazy(cuts_path)
            
            for cut in cutset:
                # Add the cut ID directly without filtering by validation_set_name
                # since shar cut IDs don't have validation set prefixes
                filtered_keys.add(cut.id)
        except Exception as e:
            print(f"Error reading cuts from {cuts_file}: {e}")
            continue
    
    return filtered_keys    


def mark(prompt, client, model="gpt-4o-mini"):
    """
    Score a response using GPT.
    
    Args:
        prompt: The prompt to send to GPT
        client: OpenAI client instance
        model: Model name to use (default: gpt-4o-mini)
    
    Returns:
        List of scores from GPT (3 attempts)
    """
    try:
        scores = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=0.5,
            top_p=0.95,
            n=3,
        )
    except Exception as e:
        print(f"Error in mark function: {e}")
        return [0, 0, 0]
    else:
        return [choice.message.content for choice in scores.choices]


def compute_gpt_scores_for_turn_pairs(user_segments, agent_segments, user_transcripts, agent_transcripts, openai_client, gpt_model="gpt-4o-mini"):
    """
    Compute GPT scores for user-agent turn pairs.
    
    Args:
        user_segments: List of dicts with 'start' and 'end' times of user speech
        agent_segments: List of dicts with 'start' and 'end' times of agent speech
        user_transcripts: Dict mapping (start, end) tuples to user transcripts
        agent_transcripts: Dict mapping (start, end) tuples to agent transcripts or segment info dicts
        openai_client: OpenAI client for GPT scoring
        gpt_model: Model name to use for GPT scoring (default: gpt-4o-mini)
    
    Returns:
        List of dicts containing GPT scores for each user-agent turn pair
    """
    gpt_scores = []
    
    # Prompt template for GPT evaluation
    prompt_template = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.

Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response].

Please evaluate the response on a scale of 1 to 5:

1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.

2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.

3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.

4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.

5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user's instruction and models' response:

### [Instruction]

{question}

### [Response]

{answer}

After evaluating, please output the score only without anything else.

You don't need to provide any explanations.
""".strip()
    
    # Create user-agent turn pairs (user turn followed by the next agent turn)
    # Sort segments by start time
    sorted_user_segments = sorted(user_segments, key=lambda x: x['start'])
    sorted_agent_segments = sorted(agent_segments, key=lambda x: x['start'])
    
    for user_seg in sorted_user_segments:
        # Find the next agent segment that starts after this user segment ends
        next_agent_seg = None
        for agent_seg in sorted_agent_segments:
            if agent_seg['start'] >= user_seg['start']:
                next_agent_seg = agent_seg
                break
        
        if next_agent_seg is not None:
            # Get transcripts
            user_key = (user_seg['start'], user_seg['end'])
            agent_key = (next_agent_seg['start'], next_agent_seg['end'])
            
            user_text = user_transcripts.get(user_key, "")
            
            # Extract agent text (handle both dict and string formats)
            agent_info = agent_transcripts.get(agent_key, "")
            if isinstance(agent_info, dict):
                agent_text = remove_special_symbols(agent_info.get('text', ''))
            else:
                agent_text = remove_special_symbols(agent_info)
            
            # Only compute GPT score if both transcripts are available
            if user_text.strip() and agent_text.strip():
                prompt = prompt_template.format(question=user_text, answer=agent_text)
                try:
                    scores = mark(prompt, openai_client, model=gpt_model)
                    # Parse scores from the responses
                    parsed_scores = []
                    for score_text in scores:
                        try:
                            # Extract numeric score from response
                            score_value = float(score_text.strip())
                            parsed_scores.append(score_value)
                        except ValueError:
                            print(f"Warning: Could not parse score from GPT response: {score_text}")
                            parsed_scores.append(0.0)
                    
                    avg_score = sum(parsed_scores) / len(parsed_scores) if parsed_scores else 0.0
                    gpt_scores.append({
                        'user_segment': user_seg,
                        'agent_segment': next_agent_seg,
                        'user_text': user_text,
                        'agent_text': agent_text,
                        'scores': parsed_scores,
                        'avg_score': avg_score
                    })
                    print(f"GPT score for pair [User: {user_seg['start']:.3f}s -> Agent: {next_agent_seg['start']:.3f}s]: {parsed_scores} (avg: {avg_score:.2f})")
                except Exception as e:
                    print(f"Error computing GPT score for pair [User: {user_seg['start']:.3f}s -> Agent: {next_agent_seg['start']:.3f}s]: {e}")
    
    return gpt_scores


def compute_user_eou_metrics(predicted_user_segments, gt_user_segments, match_threshold_sec=2.0):
    """
    Compute user EOU (End-of-Utterance) detection metrics by comparing predicted segments with ground truth.
    
    Args:
        predicted_user_segments: List of predicted user segments with 'start' and 'end' times
        gt_user_segments: List of ground truth user segments (from VAD) with 'start' and 'end' times
        match_threshold_sec: Threshold in seconds for matching start and end times (default: 2.0)
    
    Returns:
        dict: Contains precision, recall, f1, matched segments info, and EOU latency statistics
    """
    if not predicted_user_segments or not gt_user_segments:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'eou_latencies': [],
            'avg_eou_latency': 0.0,
            'early_eou_count': 0,
            'late_eou_count': 0,
            'matched_pairs': []
        }
    
    # Track which segments have been matched
    matched_pred_indices = set()
    matched_gt_indices = set()
    matched_pairs = []
    eou_latencies = []
    
    # For each predicted segment, try to find a matching GT segment
    for i, pred_seg in enumerate(predicted_user_segments):
        best_match_idx = None
        best_match_score = float('inf')
        
        for j, gt_seg in enumerate(gt_user_segments):
            if j in matched_gt_indices:
                continue
            
            # Check if start and end times are within threshold
            start_diff = abs(pred_seg['start'] - gt_seg['start'])
            end_diff = abs(pred_seg['end'] - gt_seg['end'])
            
            if start_diff <= match_threshold_sec and end_diff <= match_threshold_sec:
                # Score based on combined difference (lower is better)
                match_score = start_diff + end_diff
                if match_score < best_match_score:
                    best_match_score = match_score
                    best_match_idx = j
        
        # If found a match, record it
        if best_match_idx is not None:
            matched_pred_indices.add(i)
            matched_gt_indices.add(best_match_idx)
            
            gt_seg = gt_user_segments[best_match_idx]
            eou_latency = pred_seg['end'] - gt_seg['end']  # Positive = late, Negative = early
            eou_latencies.append(eou_latency)
            
            matched_pairs.append({
                'predicted': pred_seg,
                'ground_truth': gt_seg,
                'eou_latency': eou_latency,
                'start_diff': abs(pred_seg['start'] - gt_seg['start']),
                'end_diff': abs(pred_seg['end'] - gt_seg['end'])
            })
    
    # Calculate metrics
    tp = len(matched_pred_indices)
    fp = len(predicted_user_segments) - tp
    fn = len(gt_user_segments) - len(matched_gt_indices)
    
    # Visualize segment matching
    print("\n" + "="*80)
    print("SEGMENT MATCHING VISUALIZATION")
    print("="*80)
    
    # Show matched pairs (True Positives)
    if matched_pairs:
        print(f"\n✓ MATCHED SEGMENTS (True Positives: {tp}):")
        print("-" * 80)
        for i, pair in enumerate(matched_pairs, 1):
            pred = pair['predicted']
            gt = pair['ground_truth']
            print(f"  Match #{i}:")
            print(f"    Predicted:     [{pred['start']:.2f}s - {pred['end']:.2f}s] (duration: {pred['end']-pred['start']:.2f}s)")
            print(f"    Ground Truth:  [{gt['start']:.2f}s - {gt['end']:.2f}s] (duration: {gt['end']-gt['start']:.2f}s)")
            print(f"    Differences:   start_diff={pair['start_diff']:.3f}s, end_diff={pair['end_diff']:.3f}s, eou_latency={pair['eou_latency']:.3f}s")
            print()
    
    # Show false positives (predicted but no match)
    fp_segments = [seg for i, seg in enumerate(predicted_user_segments) if i not in matched_pred_indices]
    if fp_segments:
        print(f"\n✗ FALSE POSITIVES (Predicted but no GT match: {fp}):")
        print("-" * 80)
        for i, seg in enumerate(fp_segments, 1):
            print(f"  FP #{i}: [{seg['start']:.2f}s - {seg['end']:.2f}s] (duration: {seg['end']-seg['start']:.2f}s)")
        print()
    
    # Show false negatives (GT but no match)
    fn_segments = [seg for i, seg in enumerate(gt_user_segments) if i not in matched_gt_indices]
    if fn_segments:
        print(f"\n✗ FALSE NEGATIVES (GT but no prediction match: {fn}):")
        print("-" * 80)
        for i, seg in enumerate(fn_segments, 1):
            print(f"  FN #{i}: [{seg['start']:.2f}s - {seg['end']:.2f}s] (duration: {seg['end']-seg['start']:.2f}s)")
        print()
    
    print("="*80 + "\n")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate EOU latency statistics
    avg_eou_latency = sum(eou_latencies) / len(eou_latencies) if eou_latencies else 0.0
    early_eou_count = sum(1 for lat in eou_latencies if lat <= -2.0)
    late_eou_count = sum(1 for lat in eou_latencies if lat > 0)
    
    print(f"User EOU Detection - TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"User EOU Detection - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"User EOU Detection - Avg EOU Latency: {avg_eou_latency:.3f}s ({avg_eou_latency*1000:.1f}ms)")
    print(f"User EOU Detection - Early: {early_eou_count}, Late: {late_eou_count}, On-time: {len(eou_latencies) - early_eou_count - late_eou_count}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'eou_latencies': eou_latencies,
        'avg_eou_latency': avg_eou_latency,
        'early_eou_count': early_eou_count,
        'late_eou_count': late_eou_count,
        'matched_pairs': matched_pairs
    }


def compute_turn_taking_metrics(agent_segments, user_segments, tt_latency_threshold_sec, tt_precision_buffer_sec, tt_recall_buffer_sec, user_transcripts=None, agent_transcripts=None, openai_client=None, gpt_model="gpt-4o-mini"):
    """
    Compute turn-taking metrics using precision and recall, and optionally compute GPT scores for user-agent turn pairs.
    
    Args:
        agent_segments: List of dicts with 'start' and 'end' times of agent speech
        user_segments: List of dicts with 'start' and 'end' times of user speech
        tt_latency_threshold_sec: Threshold in seconds for considering turn-taking to be accurate
        tt_precision_buffer_sec: Buffer time in seconds for precision calculation (agent segments)
        tt_recall_buffer_sec: Buffer time in seconds for recall calculation (user segments)
        user_transcripts: Optional dict mapping (start, end) tuples to user transcripts
        agent_transcripts: Optional dict mapping (start, end) tuples to agent transcripts or segment info dicts
        openai_client: Optional OpenAI client for GPT scoring
        gpt_model: Model name to use for GPT scoring (default: gpt-4o-mini)
    Returns:
        dict: Contains precision, recall, f1, latency metrics, and optionally gpt_scores
    """
    if not agent_segments or not user_segments:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'avg_latency': INF_LATENCY,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'gpt_scores': []
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
    
    # Compute GPT scores for user-agent turn pairs if transcripts and client are provided
    gpt_scores = []
    if user_transcripts is not None and agent_transcripts is not None and openai_client is not None:
        gpt_scores = compute_gpt_scores_for_turn_pairs(
            user_segments, 
            agent_segments, 
            user_transcripts, 
            agent_transcripts, 
            openai_client, 
            gpt_model
        )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_latency': avg_latency,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'gpt_scores': gpt_scores
    }


def main(args):
    vad_model, get_speech_timestamps = init_vad_model()
    
    # Initialize ASR model for transcription if requested
    asr_model = None
    if args.enable_transcription:
        asr_model = init_asr_model(args.asr_model_name)
    
    # Initialize text normalizer for WER computation
    text_normalizer = EnglishTextNormalizer() if args.enable_transcription else None

    manifest_dir = args.manifest_dir
    pred_audio_dir= args.pred_audio_dir

    # Load timestamped predictions if provided
    timestamped_preds = None
    if args.jsonl_with_timestamp:
        print(f"Loading timestamped predictions from: {args.jsonl_with_timestamp}")
        
        # If compute_user_eou is enabled, also load pred_src_text field
        if args.compute_user_eou:
            timestamped_preds = load_jsonl(args.jsonl_with_timestamp, field_name='pred_text', additional_fields=['pred_src_text'])
            print(f"Loaded {len(timestamped_preds)} timestamped predictions (including pred_src_text for user EOU)")
        else:
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
        
        # Lists for user EOU metrics
        all_user_eou_precisions = []
        all_user_eou_recalls = []
        all_user_eou_f1s = []
        all_user_eou_latencies = []
        
        # List for user WER metrics
        all_user_wer_scores = []
        
        # List to store all metrics dictionaries for percentile analysis
        all_metrics_dicts = []

        # Eval a specific validation set
        # Get keys from shar input if provided, otherwise from pred_audio_dir
        if args.shar_input_dir is not None:
            filtered_wav_keys = get_filtered_keys_from_shar(args.shar_input_dir, val_set_name)
            print(f"Found {len(filtered_wav_keys)} keys from shar input directory")
        else:
            filtered_wav_keys = get_filtered_wav_keys(pred_audio_dir, val_set_name)
            print(f"Found {len(filtered_wav_keys)} keys from pred_audio_dir")

        for filtered_wav_key in filtered_wav_keys:

            # Read user and agent audios
            if args.shar_input_dir is not None:
                # Load user audio from lhotse shar format
                print(f"Loading user audio from shar: {args.shar_input_dir}")
                
                user_audio = None
                user_audio_sr = None
                
                try:
                    # Load the entire CutSet from shar directory (this handles both cuts and tar files)
                    cutset = CutSet.from_shar(
                        in_dir=args.shar_input_dir,
                        shuffle_shards=False
                    )
                    
                    # Find the cut with matching ID
                    # The filtered_wav_key might have suffixes like _rank0, so we need to match the base ID
                    filtered_wav_key_id = filtered_wav_key.split('_rank')[0] if '_rank' in filtered_wav_key else filtered_wav_key
                    
                    matching_cut = None
                    for cut in cutset:
                        if cut.id == filtered_wav_key_id or filtered_wav_key_id in cut.id:
                            matching_cut = cut
                            break
                    
                    if matching_cut is not None:
                        print(f"Found matching cut with ID: {matching_cut.id}")
                        # Load audio from the cut
                        audio_array = matching_cut.load_audio()
                        user_audio = torch.from_numpy(audio_array).float()
                        if user_audio.dim() == 1:
                            user_audio = user_audio.unsqueeze(0)  # Add channel dimension
                        user_audio_sr = matching_cut.sampling_rate
                except Exception as e:
                    print(f"Error loading from shar: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                
                if user_audio is None:
                    print(f"Could not find user audio for ID: {filtered_wav_key} in shar")
                    continue
                
                # Write user_audio out to temp dir for VAD processing
                debug_user_audio_dir = os.path.join(tempfile.gettempdir(), "shar_user_recordings")
                os.makedirs(debug_user_audio_dir, exist_ok=True)
                debug_user_audio_path = os.path.join(debug_user_audio_dir, f"{filtered_wav_key}.user.wav")
                # Resample to 16kHz if needed before saving
                if user_audio_sr != 16000:
                    user_audio_to_save = torchaudio.functional.resample(user_audio, user_audio_sr, 16000)
                    torchaudio.save(debug_user_audio_path, user_audio_to_save, 16000)
                else:
                    torchaudio.save(debug_user_audio_path, user_audio, 16000)
                
            elif getattr(args, "is_stereo", False):
                pred_audio_file = get_pred_audio_path(pred_audio_dir, filtered_wav_key, val_set_name)
                if pred_audio_file is None or not os.path.exists(pred_audio_file):
                    print(f"File not found: {pred_audio_file}")
                    continue
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
                            if pred_audio_file is None or not os.path.exists(pred_audio_file):
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
            if args.shar_input_dir is None:
                agent_audio = torchaudio.functional.resample(agent_audio, agent_audio_sr, 16000)
            
            # Use timestamped predictions for agent segments if available, otherwise use VAD or binary audio
            # Find the full key that contains filtered_wav_key as a substring
            filtered_wav_key_id = filtered_wav_key.split('_rank')[0] if '_rank' in filtered_wav_key else filtered_wav_key
            matching_key = next((k for k in timestamped_preds.keys() if filtered_wav_key_id in k), None) if timestamped_preds else None

            print("Eval audio: ", filtered_wav_key)
            
            # Extract user segments first (needed for agent cutoff detection)
            user_vad_results = get_speech_timestamps(user_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=args.vad_min_silence_duration_ms)
            user_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in user_vad_results]
            
            # Extract predicted user segments from pred_src_text if compute_user_eou is enabled
            predicted_user_segments = None
            user_eou_metrics = None
            predicted_user_transcripts = {}
            user_wer_scores = {}
            
            if args.compute_user_eou and matching_key and isinstance(timestamped_preds.get(matching_key), dict):
                pred_src_text = timestamped_preds[matching_key].get('pred_src_text', '')
                if pred_src_text:
                    print(f"Parsing predicted user segments from pred_src_text for {matching_key}")
                    predicted_user_segments = parse_user_timestamped_text(pred_src_text)
                    print(f"Parsed {len(predicted_user_segments)} predicted user segments")
                    
                    # Store predicted user transcripts
                    for seg in predicted_user_segments:
                        seg_key = (seg['start'], seg['end'])
                        predicted_user_transcripts[seg_key] = seg['text']
                    
                    # Compute user EOU metrics
                    user_eou_metrics = compute_user_eou_metrics(
                        predicted_user_segments,
                        user_segments,
                        match_threshold_sec=args.user_eou_match_threshold_sec
                    )
            
            # Extract agent segments (with cutoff detection if using timestamped text)
            if matching_key:
                print(f"Using timestamped text predictions for {matching_key}")
                # Handle both dict and string formats for timestamped_preds
                if isinstance(timestamped_preds[matching_key], dict):
                    timestamped_text = timestamped_preds[matching_key]['pred_text']
                else:
                    timestamped_text = timestamped_preds[matching_key]
                
                agent_segments = parse_timestamped_text(
                    timestamped_text, 
                    estimate_sec_per_word=args.estimate_sec_per_word,
                    cutoff_threshold=args.cutoff_duration_threshold_sec,
                    user_segments=user_segments,
                    gap_threshold=args.cutoff_gap_threshold_sec
                )
                print(f"Parsed {len(agent_segments)} agent segments from timestamped text")
            elif args.agent_binary_audio:
                # Extract segments from binary audio (0s and 1s)
                print(f"Extracting agent segments from binary audio for {filtered_wav_key}")
                agent_segments = extract_segments_from_binary_audio(agent_audio, sample_rate=16000)
                print(f"Extracted {len(agent_segments)} agent segments from binary audio")
            else:
                continue
                # Fallback to VAD-based segmentation
                # agent_vad_results = get_speech_timestamps(agent_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=args.vad_min_silence_duration_ms)
                # agent_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in agent_vad_results]

            #################
            # Compute eval Metrics 
            # So far, we mesaure three types of conversation behaviors: Turn-taking, barge-in, and user backchanneling

            # Barge-in: find the overlap and return the overlap duration and segments
            success_barge_ins, failed_barge_ins = find_user_barge_ins(user_segments, agent_segments, args.barge_in_threshold_sec)
            barge_in_metrics = compute_barge_in_metrics(success_barge_ins, failed_barge_ins)

            # Backchaneling: find the backchanneling, end_time is the timestamp of backchannel end point
            end_time = args.end_time  # end time of backchanneling, note that this is predefined when creating backchanneling data
            if end_time is not None:
                bc_failure = is_stopped_by_backchannel(agent_segments, end_time, args.barge_in_threshold_sec)
            else:
                bc_failure = []

            # Transcribe/extract text for segments (assuming user transcription is ready by default)
            user_transcripts = {}
            agent_transcripts = {}
            
            # Extract agent text from timestamped predictions if available
            if timestamped_preds and matching_key:
                # Handle both dict and string formats for timestamped_preds
                if isinstance(timestamped_preds[matching_key], dict):
                    timestamped_text = timestamped_preds[matching_key]['pred_text']
                else:
                    timestamped_text = timestamped_preds[matching_key]
                
                agent_segments_with_text = parse_timestamped_text(
                    timestamped_text,
                    estimate_sec_per_word=args.estimate_sec_per_word,
                    cutoff_threshold=args.cutoff_duration_threshold_sec,
                    user_segments=user_segments,
                    gap_threshold=args.cutoff_gap_threshold_sec
                )
                # Convert list to dict with (start, end) tuples as keys, storing full segment info
                agent_transcripts = {(seg['start'], seg['end']): seg for seg in agent_segments_with_text}
            
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
                
                # Compute WER for matched predicted and transcribed user segments
                # Use the matched pairs from EOU computation to ensure consistency
                if predicted_user_transcripts and user_transcripts and user_eou_metrics:
                    matched_pairs = user_eou_metrics.get('matched_pairs', [])
                    print(f"Computing WER for {len(matched_pairs)} matched user segments (from EOU pairs)...")
                    
                    for pair in matched_pairs:
                        gt_seg = pair['ground_truth']
                        pred_seg = pair['predicted']
                        
                        gt_key = (gt_seg['start'], gt_seg['end'])
                        pred_key = (pred_seg['start'], pred_seg['end'])
                        
                        gt_text = user_transcripts.get(gt_key, '')
                        pred_text_raw = predicted_user_transcripts.get(pred_key, '')
                        
                        if gt_text and pred_text_raw:
                            pred_text = clean_predicted_text(pred_text_raw)
                            wer_score = compute_wer_for_text(gt_text, pred_text, normalizer=text_normalizer)
                            user_wer_scores[gt_key] = wer_score
                            print(f"  User segment GT:[{gt_seg['start']:.3f}s - {gt_seg['end']:.3f}s] Pred:[{pred_seg['start']:.3f}s - {pred_seg['end']:.3f}s]: WER = {wer_score:.1%}")
                            if text_normalizer:
                                normalized_gt = text_normalizer(gt_text)
                                normalized_pred = text_normalizer(pred_text)
                                print(f"    GT (raw):        '{gt_text}'")
                                print(f"    Pred (raw):      '{pred_text}'")
                                print(f"    GT (norm):       '{normalized_gt}'")
                                print(f"    Pred (norm):     '{normalized_pred}'")
                            else:
                                print(f"    GT:   '{gt_text}'")
                                print(f"    Pred: '{pred_text}'")

            # Initialize OpenAI client for GPT scoring if API key is available
            openai_client = None
            if args.openai_key:
                try:
                    openai_client = OpenAI(api_key=args.openai_key)
                except Exception as e:
                    print(f"Warning: Could not initialize OpenAI client: {e}")
            
            # Turn-taking metrics - compute once with optional GPT scoring if transcripts are available
            tt_metrics = compute_turn_taking_metrics(
                agent_segments, 
                user_segments, 
                args.tt_latency_threshold_sec, 
                args.tt_precision_buffer_sec, 
                args.tt_recall_buffer_sec,
                user_transcripts=user_transcripts if user_transcripts else None,
                agent_transcripts=agent_transcripts if agent_transcripts else None,
                openai_client=openai_client,
                gpt_model=args.gpt_model
            )
            tt_latency = tt_metrics['avg_latency']
            tt_accuracy = tt_metrics['f1']  # Use F1 as overall accuracy metric
            gpt_scores = tt_metrics.get('gpt_scores', [])
            
            if gpt_scores:
                print(f"Computed {len(gpt_scores)} GPT scores")

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
            
            # Store user EOU metrics if computed
            if user_eou_metrics is not None:
                all_user_eou_precisions.append(user_eou_metrics['precision'])
                all_user_eou_recalls.append(user_eou_metrics['recall'])
                all_user_eou_f1s.append(user_eou_metrics['f1'])
                if user_eou_metrics['eou_latencies']:
                    all_user_eou_latencies.extend(user_eou_metrics['eou_latencies'])
            
            # Store user WER scores
            if user_wer_scores:
                all_user_wer_scores.extend(user_wer_scores.values())

            # Compute cutoff metrics from agent_transcripts
            cutoff_segments = []
            total_agent_segments_with_text = 0
            for seg_info in agent_transcripts.values():
                if isinstance(seg_info, dict):
                    total_agent_segments_with_text += 1
                    if seg_info.get('is_cutoff', False):
                        cutoff_segments.append(seg_info)
            
            cutoff_rate = (len(cutoff_segments) / total_agent_segments_with_text * 100) if total_agent_segments_with_text > 0 else 0.0
            
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
                'agent_transcripts': agent_transcripts,
                'predicted_user_transcripts': predicted_user_transcripts,
                'user_wer_scores': user_wer_scores,
                'cutoff_count': len(cutoff_segments),
                'cutoff_rate': cutoff_rate,
                'total_agent_segments_with_text': total_agent_segments_with_text,
                'gpt_scores': gpt_scores,
                'user_eou_metrics': user_eou_metrics
            }

            # Store metrics for percentile analysis
            all_metrics_dicts.append(metrics_dict)

            # Print all metrics
            print_metrics(metrics_dict, verbose=args.verbose)
            
            count += 1

        # Compute and print average metrics
        _valid_tt_latencies = [x for x in all_tt_latencies if x != INF_LATENCY]
        
        # Compute cutoff statistics
        total_cutoffs = sum(m.get('cutoff_count', 0) for m in all_metrics_dicts)
        total_agent_segs_with_text = sum(m.get('total_agent_segments_with_text', 0) for m in all_metrics_dicts)
        avg_cutoff_rate = (total_cutoffs / total_agent_segs_with_text * 100) if total_agent_segs_with_text > 0 else 0.0
        
        # Compute GPT score statistics
        all_gpt_avg_scores = []
        for m in all_metrics_dicts:
            gpt_scores = m.get('gpt_scores', [])
            for score_info in gpt_scores:
                all_gpt_avg_scores.append(score_info.get('avg_score', 0.0))
        avg_gpt_score = sum(all_gpt_avg_scores) / len(all_gpt_avg_scores) if all_gpt_avg_scores else 0.0
        
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
            'total_cutoffs': total_cutoffs,
            'total_agent_segments_with_text': total_agent_segs_with_text,
            'avg_cutoff_rate': avg_cutoff_rate,
            'avg_gpt_score': avg_gpt_score,
            'num_gpt_scores': len(all_gpt_avg_scores),
            'avg_user_eou_precision': sum(all_user_eou_precisions) / len(all_user_eou_precisions) * 100 if all_user_eou_precisions else 0,
            'avg_user_eou_recall': sum(all_user_eou_recalls) / len(all_user_eou_recalls) * 100 if all_user_eou_recalls else 0,
            'avg_user_eou_f1': sum(all_user_eou_f1s) / len(all_user_eou_f1s) * 100 if all_user_eou_f1s else 0,
            'avg_user_eou_latency': sum(all_user_eou_latencies) / len(all_user_eou_latencies) if all_user_eou_latencies else 0,
            'num_user_eou_evaluated': len(all_user_eou_precisions),
            'avg_user_wer': sum(all_user_wer_scores) / len(all_user_wer_scores) if all_user_wer_scores else 0,
            'num_user_wer_evaluated': len(all_user_wer_scores),
        }

        gpt_score_str = ""
        if avg_metrics['num_gpt_scores'] > 0:
            gpt_score_str = f"""
    5. GPT Quality Score:
    - Average score: {avg_metrics['avg_gpt_score']:.2f}/5.0
    - Number of scored pairs: {avg_metrics['num_gpt_scores']}"""
        
        user_eou_str = ""
        if avg_metrics['num_user_eou_evaluated'] > 0:
            user_eou_str = f"""
    6. User EOU Detection:
    - Precision: {avg_metrics['avg_user_eou_precision']:.1f}%
    - Recall: {avg_metrics['avg_user_eou_recall']:.1f}%
    - F1: {avg_metrics['avg_user_eou_f1']:.1f}%
    - Average EOU latency: {avg_metrics['avg_user_eou_latency'] * 1000:.1f} ms
    - Number of samples: {avg_metrics['num_user_eou_evaluated']}"""
        
        user_wer_str = ""
        if avg_metrics['num_user_wer_evaluated'] > 0:
            wer_section_num = 7 if user_eou_str else 6
            user_wer_str = f"""
    {wer_section_num}. User Speech Recognition (WER):
    - Average WER: {avg_metrics['avg_user_wer']:.1%}
    - Number of segments evaluated: {avg_metrics['num_user_wer_evaluated']}"""
        
        num_section = 8 if (user_eou_str and user_wer_str) else (7 if (user_eou_str or user_wer_str) else 6)
        
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
    4. Agent cutoff detection:
    - Cutoff rate: {avg_metrics['avg_cutoff_rate']:.1f}% ({avg_metrics['total_cutoffs']}/{avg_metrics['total_agent_segments_with_text']}){gpt_score_str}{user_eou_str}{user_wer_str}
    {num_section}. Number of audios evaluated: {avg_metrics['num_audios_evaluated']}
    {'=' * 50}"""

        print(avg_metrics_str)
        
        # Print bottom 5% percentile utterances
        if args.show_bottom_percentile:
            print_bottom_percentile_utterances(all_metrics_dicts, percentile=args.percentile_threshold)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_audio_dir", type=str, default="/lustre/fsw/portfolios/convai/users/cchen1/results/s2s_rl/uc_samples")
    parser.add_argument("--manifest_dir", type=str, default=None, nargs='?', help="Path to manifest directory. Optional.")
    parser.add_argument("--shar_input_dir", type=str, default=None, help="Path to lhotse shar directory containing cuts.*.jsonl.gz and recording.*.tar files for user audio input. If provided, user audio will be loaded from this shar instead of stereo or manifest_dir.")
    parser.add_argument("--barge_in_threshold_sec", type=float, default=1.5, help="Buffering time for the agent to stop after user barges in.")
    parser.add_argument("--tt_latency_threshold_sec", type=float, default=0.64, help="Threshold in seconds for considering a turn-taking to be accurate.")
    parser.add_argument("--tt_precision_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for precision calculation (agent segments).")
    parser.add_argument("--tt_recall_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for recall calculation (user segments).")
    
    # Read default OpenAI API key from file
    default_openai_key = None
    try:
        with open('/lustre/fsw/portfolios/llmservice/users/kevinhu/HFCACHE/zh_openai_key.txt', 'r') as f:
            default_openai_key = f.read().strip()
    except Exception:
        pass
    
    parser.add_argument("--openai_key", type=str, default=default_openai_key, help="API key for OpenAI authentication. Defaults to reading from zh_openai_key.txt file.")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="GPT model name to use for scoring user-agent turn pairs. Works with --openai_key.")
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
    parser.add_argument("--enable_transcription", action="store_true", default=False, help="Enable transcription of user segments using ASR model. Agent text is automatically extracted from --jsonl_with_timestamp if provided.")
    parser.add_argument("--asr_model_name", type=str, default="nvidia/parakeet-tdt-0.6b-v2", help="Name of the ASR model to use for transcription of user segments.")
    parser.add_argument("--show_bottom_percentile", action="store_true", default=True, help="Show detailed analysis of utterances in the bottom percentile for barge-in accuracy and turn-taking recall.")
    parser.add_argument("--percentile_threshold", type=float, default=5.0, help="Percentile threshold for identifying low-quality utterances (default: 5.0 for bottom 5%%).")
    parser.add_argument("--cutoff_duration_threshold_sec", type=float, default=2, help="Threshold in seconds for marking agent segment as cut off based on duration mismatch (estimated vs actual).")
    parser.add_argument("--cutoff_gap_threshold_sec", type=float, default=2, help="Minimum gap in seconds between estimated agent end and next user start to consider agent segment as cut off.")
    parser.add_argument("--estimate_sec_per_word", type=float, default=0.3, help="Estimated seconds per word for duration calculation when detecting cutoffs.")
    parser.add_argument("--compute_user_eou", action="store_true", default=False, help="Compute user EOU (End-of-Utterance) detection metrics by comparing predicted user segments from pred_src_text with ground truth VAD segments.")
    parser.add_argument("--user_eou_match_threshold_sec", type=float, default=2.0, help="Threshold in seconds for matching predicted and ground truth user segment start/end times when computing EOU metrics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)