########################
# Eval script for turn-taking (TT), user back-channeling (BC), user barge-in (BI), Adapted from Chen Chen script

import torchaudio, torch
from tqdm import tqdm
import tarfile
import gzip
import json, os, re
import argparse


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
        if u_start < a_end and u_start > a_start:
            overlap_start = max(u_start, a_start)
            overlap_end = min(u_end, a_end)
            overlap_duration = round((overlap_end - overlap_start) * 1000)
            
            barge_in_info = {
                'overlap_ms': overlap_duration,
                'user': user_turns[i],
                'agent': agent_turns[j]
            }
            
            if overlap_duration < threshold_seconds * 1000:
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
        metrics['avg_latency_ms'] = sum(bi['overlap_ms'] for bi in success_barge_ins) / success_count
    
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
        verbose: If True, print detailed segment information
    """
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
        # General speech segments
        user_segments_str = ", ".join(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s]" for seg in metrics_dict['user_segments'])
        agent_segments_str = ", ".join(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s]" for seg in metrics_dict['agent_segments'])
        
        # Barge-in segments
        barge_in_segments = []
        if metrics_dict['barge_in_metrics']['has_barge_ins']:
            if metrics_dict['success_barge_ins']:
                barge_in_segments.append("   Successful barge-ins:")
                for bi in metrics_dict['success_barge_ins']:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.1f}s - {bi['user']['end']:.1f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.1f}s - {bi['agent']['end']:.1f}s]")
                    barge_in_segments.append(f"     Overlap: {bi['overlap_ms']:.1f} ms")
            
            if metrics_dict['failed_barge_ins']:
                barge_in_segments.append("   Failed barge-ins:")
                for bi in metrics_dict['failed_barge_ins']:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.1f}s - {bi['user']['end']:.1f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.1f}s - {bi['agent']['end']:.1f}s]")
                    barge_in_segments.append(f"     Overlap: {bi['overlap_ms']:.1f} ms")

        segment_info = f"""
4. User speech segments:
{user_segments_str}
5. Agent speech segments:
{agent_segments_str}
6. Barge-in details:
{chr(10).join(barge_in_segments) if barge_in_segments else "   No barge-ins detected"}"""

    # Build the complete output string
    output = f"""
Evaluation metrics for conversation {metrics_dict['item_id']}:
1. 1st turn-taking latency: {metrics_dict['tt_latency']:.3f} seconds
   - 1st turn-taking accuracy: {'Accurate' if metrics_dict['tt_accuracy'] else 'Inaccurate'}
2. Barge-in statistics:
{chr(10).join(barge_in_stats)}
3. Backchanneling failures: {metrics_dict['bc_failure']}{segment_info}
{'-' * 50}"""

    print(output)


def main(args):
    vad_model, get_speech_timestamps = init_vad_model()

    manifest_dir = args.manifest_dir
    pred_audio_path= args.pred_audio_path
    cuts_files = [f for f in os.listdir(manifest_dir) if f.startswith("cuts.") and f.endswith(".jsonl.gz")]
    recording_files = [f for f in os.listdir(manifest_dir) if f.startswith("recording.") and f.endswith(".tar")]
    cuts_files.sort()
    recording_files.sort()

    # Lists to store metrics for averaging
    all_tt_latencies = []
    all_tt_accuracies = []
    all_barge_in_success_rates = []
    all_barge_in_latencies = []
    all_bc_accuracies = []

    count = 0
    for cuts_file, recording_file in zip(cuts_files, recording_files):
        cuts_path = os.path.join(manifest_dir, cuts_file)
        recording_tar_path = os.path.join(manifest_dir, recording_file)

        with tarfile.open(recording_tar_path, 'r') as recording_tar:
            with gzip.open(cuts_path, 'rt', encoding='utf-8') as f:

                for line in tqdm(f, desc=f"Processing {cuts_file}"):

                    item = json.loads(line.strip())
                    item_id = item['id']

                    # load from pred_audio_path
                    pred_audio_file = os.path.join(pred_audio_path, f"{item_id}.gen.wav").strip()
                    if not os.path.exists(pred_audio_file):
                        print(f"File not found: {pred_audio_file}")
                        continue

                    # load from recording.000000.tar
                    json_file_name = f"{item_id}.json"
                    flac_file_name = f"{item_id}.flac"

                    if json_file_name in recording_tar.getnames() and flac_file_name in recording_tar.getnames():
                        json_member = recording_tar.getmember(json_file_name)
                        flac_member = recording_tar.getmember(flac_file_name)
                        with recording_tar.extractfile(flac_member) as flac_file:
                            user_audio, user_audio_sr = torchaudio.load(flac_file)   # weqing 1
                    else:
                        print(f"Missing files for ID: {item_id} in tar archive")

                    agent_audio, agent_audio_sr = torchaudio.load(pred_audio_file)

                    agent_audio = torchaudio.functional.resample(agent_audio, agent_audio_sr, 16000)
                    user_audio = torchaudio.functional.resample(user_audio, user_audio_sr, 16000)

                    agent_vad_results = get_speech_timestamps(agent_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=1500)
                    # Here min_silence_duration_ms is a important metric to control the tolerance of "---" in =====---=====

                    agent_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in agent_vad_results]

                    user_vad_results = get_speech_timestamps(user_audio.to('cuda'), vad_model, sampling_rate=16000, min_silence_duration_ms=1500)
                    user_segments = [{'start': s['start'] / 16000, 'end': s['end'] / 16000} for s in user_vad_results]


                    #################
                    # Eval Metrics 

                    # 1st turn-taking latency: The latency between the first agent turn and the first user turn
                    tt_latency = agent_segments[0]['start'] - user_segments[0]['end']
                    tt_accuracy = tt_latency <= args.tt_accuracy_threshold_sec

                    # Barge-in: find the overlap and return the overlap duration and segments
                    success_barge_ins, failed_barge_ins = find_user_barge_ins(user_segments, agent_segments, args.barge_in_threshold_sec)                    
                    barge_in_metrics = compute_barge_in_metrics(success_barge_ins, failed_barge_ins)

                    # Backchaneling: find the backchanneling, end_time is the timestamp of backchannel end point
                    end_time = args.end_time # end time of backchanneling
                    bc_failure = is_stopped_by_backchannel(agent_segments, end_time)

                    # Store metrics for averaging
                    all_tt_latencies.append(tt_latency)
                    all_tt_accuracies.append(1 if tt_accuracy else 0)
                    
                    if barge_in_metrics['has_barge_ins']:
                        all_barge_in_success_rates.append(barge_in_metrics['success_rate'])
                        if 'avg_latency_ms' in barge_in_metrics:
                            all_barge_in_latencies.append(barge_in_metrics['avg_latency_ms'])
                    
                    # Calculate backchannel accuracy (percentage of successful backchannels)
                    bc_accuracy = sum(1 for x in bc_failure if not x) / len(bc_failure) if bc_failure else 0
                    all_bc_accuracies.append(bc_accuracy)

                    # Collect all metrics in a dictionary
                    metrics_dict = {
                        'item_id': item_id,
                        'tt_latency': tt_latency,
                        'tt_accuracy': tt_accuracy,
                        'barge_in_metrics': barge_in_metrics,
                        'bc_failure': bc_failure,
                        'user_segments': user_segments,
                        'agent_segments': agent_segments,
                        'success_barge_ins': success_barge_ins,
                        'failed_barge_ins': failed_barge_ins
                    }

                    # Print all metrics
                    print_metrics(metrics_dict, verbose=args.verbose)
                    
                    count += 1
                    # if count > 3:
                    #     break

    # Compute and print average metrics
    avg_metrics = {
        'avg_tt_latency': sum(all_tt_latencies) / len(all_tt_latencies) if all_tt_latencies else 0,
        'avg_tt_accuracy': sum(all_tt_accuracies) / len(all_tt_accuracies) * 100 if all_tt_accuracies else 0,
        'avg_barge_in_success_rate': sum(all_barge_in_success_rates) / len(all_barge_in_success_rates) if all_barge_in_success_rates else 0,
        'avg_barge_in_latency': sum(all_barge_in_latencies) / len(all_barge_in_latencies) if all_barge_in_latencies else 0,
        'avg_bc_accuracy': sum(all_bc_accuracies) / len(all_bc_accuracies) * 100 if all_bc_accuracies else 0
    }

    avg_metrics_str = f"""
{'=' * 50}
Average Metrics Across All Conversations:
1. Turn-taking:
   - Average latency: {avg_metrics['avg_tt_latency']:.3f} seconds
   - Accuracy: {avg_metrics['avg_tt_accuracy']:.1f}%
2. Barge-in:
   - Average success rate: {avg_metrics['avg_barge_in_success_rate']:.1f}%
   - Average latency: {avg_metrics['avg_barge_in_latency']:.1f} ms
3. Back-channeling:
   - Average accuracy: {avg_metrics['avg_bc_accuracy']:.1f}%
{'=' * 50}"""

    print(avg_metrics_str)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_audio_path", type=str, default="/lustre/fsw/portfolios/convai/users/cchen1/results/s2s_rl/uc_samples")
    parser.add_argument("--manifest_dir", type=str, default="/lustre/fsw/portfolios/convai/users/cchen1/data/ultrachat_200_0")
    parser.add_argument("--barge_in_threshold_sec", type=float, default=1.5, help="Buffering time for the agent to stop after user barges in.")
    parser.add_argument("--end_time", type=list, default=[1,10,15,20], help="End time of backchanneling.")
    parser.add_argument("--tt_accuracy_threshold_sec", type=float, default=0.64, help="Threshold in seconds for considering a turn-taking to be accurate.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed segment information")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)