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
from collections import defaultdict
import torch
from nemo.utils import logging



def compute_turn_taking_metrics(source_tokens, pred_tokens, eos_token_id, bos_token_id, tolerance=13, latency_multiplier=0.08):
    """
    Computes turn taking accuracy and latency.
    
    Args:
        source_tokens (torch.Tensor): Batch of source sequences (batch_size, seq_len) - user speech
        pred_tokens (torch.Tensor): Batch of predicted sequences (batch_size, seq_len) - agent speech
        eos_token_id (int): End of speech token ID for user speech
        bos_token_id (int): Beginning of speech token ID for agent speech
        tolerance (int): Allowed index difference for successful turn taking
        latency_multiplier (float): Multiplier to convert index difference to latency (default: 0.08)
    
    Returns:
        tuple: (accuracy, average_latency) - both as Python floats
    """
    # Convert to CPU and Python lists to avoid any symbolic tensor issues
    source_tokens = source_tokens.cpu().numpy()
    pred_tokens = pred_tokens.cpu().numpy()
    
    batch_size = source_tokens.shape[0]
    
    successful_turns = 0
    total_turns = 0
    successful_latencies = []
    
    for b in range(batch_size):
        # Find first EOS in source tokens (user speech end) using numpy operations
        eos_positions = (source_tokens[b] == eos_token_id).nonzero()[0]
        if len(eos_positions) == 0:
            continue  # No user speech end found, skip
        
        # Find first BOS in predicted tokens (agent speech start) using numpy operations
        bos_positions = (pred_tokens[b] == bos_token_id).nonzero()[0]
        if len(bos_positions) == 0:
            total_turns += 1
            continue  # No agent speech start found, failed turn taking
        
        # Calculate difference between user speech end and agent speech start
        user_eos_pos = int(eos_positions[0])
        agent_bos_pos = int(bos_positions[0])
        diff = agent_bos_pos - user_eos_pos
        
        total_turns += 1
        
        # Check if within tolerance
        if abs(diff) <= tolerance:
            successful_turns += 1
            latency = diff * latency_multiplier
            successful_latencies.append(latency)
    
    # Calculate metrics as Python floats
    accuracy = successful_turns / total_turns if total_turns > 0 else 0.0
    avg_latency = sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0.0
    
    return float(accuracy), float(avg_latency)




class TurnTakingMetrics:
    """
    Computes turn taking accuracy and latency metrics.
    Following the same pattern as BLEU metrics to ensure multi-node compatibility.
    """
    
    def __init__(self, eos_token_id: int, bos_token_id: int, tolerance: int = 13, latency_multiplier: float = 0.08):
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tolerance = tolerance
        self.latency_multiplier = latency_multiplier
        # Store Python lists like BLEU does, not tensor values
        self.accuracies = defaultdict(list)
        self.latencies = defaultdict(list)
    
    def reset(self):
        self.accuracies.clear()
        self.latencies.clear()
        return self
    
    def update(self, name: str, source_tokens: torch.Tensor, pred_tokens: torch.Tensor) -> None:
        """
        Update metrics with a batch of samples.
        
        Args:
            name (str): Dataset name
            source_tokens (torch.Tensor): User speech tokens (batch_size, seq_len)
            pred_tokens (torch.Tensor): Agent speech tokens (batch_size, seq_len)
        """
        accuracy, avg_latency = compute_turn_taking_metrics(
            source_tokens, pred_tokens, 
            self.eos_token_id, self.bos_token_id,
            self.tolerance, self.latency_multiplier
        )
        
        # Store Python floats, just like BLEU stores Python strings
        self.accuracies[name].append(accuracy)
        if avg_latency > 0:  # Only add latency if there were successful cases
            self.latencies[name].append(avg_latency)

    
    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final metrics across all updates.
        Following the same pattern as BLEU.compute()
        
        Returns:
            dict: Dictionary with turn_taking_acc and turn_taking_latency metrics
        """
        corpus_metrics = {}
        
        # Get all dataset names to ensure consistent metric structure across all ranks
        all_names = set(self.accuracies.keys()) | set(self.latencies.keys())
        
        # Compute accuracy metrics - same pattern as BLEU
        for name in all_names:
            if self.accuracies[name]:
                # Calculate mean accuracy and create tensor from Python float (like BLEU does)
                acc_mean = sum(self.accuracies[name]) / len(self.accuracies[name])
                corpus_metrics[f"turn_taking_acc_{name}"] = torch.tensor(acc_mean)
            else:
                # Ensure consistent metric structure even if no data
                corpus_metrics[f"turn_taking_acc_{name}"] = torch.tensor(0.0)
        
        # Compute latency metrics - CRITICAL: always include latency metrics for all datasets
        # to ensure consistent metric structure across all ranks in distributed training
        for name in all_names:
            if self.latencies[name]:
                # Calculate mean latency and create tensor from Python float (like BLEU does)
                latency_mean = sum(self.latencies[name]) / len(self.latencies[name])
                corpus_metrics[f"turn_taking_latency_{name}"] = torch.tensor(latency_mean)
            else:
                # CRITICAL: Even if no successful latencies, return 0.0 to maintain structure consistency
                corpus_metrics[f"turn_taking_latency_{name}"] = torch.tensor(0.0)
        
        # Clear stored values
        self.accuracies.clear()
        self.latencies.clear()
        
        return corpus_metrics


def compute_token_level_latency(source_tokens, pred_tokens, pad_token_id):
    """
    Computes token-level latency between source and predicted tokens.
    Uses dynamic programming alignment (similar to edit distance) to match tokens,
    then computes position differences for matched tokens only.
    
    Args:
        source_tokens (torch.Tensor): Batch of source sequences (batch_size, seq_len)
        pred_tokens (torch.Tensor): Batch of predicted sequences (batch_size, seq_len)
        pad_token_id (int): Token ID to ignore in both sequences
    
    Returns:
        tuple: (latencies_with_tokens, mean_latency, median_latency) where:
               - latencies_with_tokens: list of dicts with 'latency' and 'token_id' for debugging
               - mean_latency: average latency across all matched tokens
               - median_latency: median latency across all matched tokens
    """
    import numpy as np
    
    # Convert to CPU and numpy for processing
    source_tokens = source_tokens.cpu().numpy()
    pred_tokens = pred_tokens.cpu().numpy()
    
    batch_size = source_tokens.shape[0]
    all_latencies = []
    
    for b in range(batch_size):
        src = source_tokens[b]
        pred = pred_tokens[b]
        
        # Filter out pad tokens and get non-pad positions
        src_nonpad_mask = src != pad_token_id
        pred_nonpad_mask = pred != pad_token_id
        
        src_nonpad_tokens = src[src_nonpad_mask]
        pred_nonpad_tokens = pred[pred_nonpad_mask]
        
        src_nonpad_positions = np.where(src_nonpad_mask)[0]
        pred_nonpad_positions = np.where(pred_nonpad_mask)[0]
        
        if len(src_nonpad_tokens) == 0 or len(pred_nonpad_tokens) == 0:
            continue
        
        # Use dynamic programming to find the alignment (similar to edit distance)
        # dp[i][j] = (cost, backpointer) where cost is the edit distance
        m, n = len(src_nonpad_tokens), len(pred_nonpad_tokens)
        dp = [[(float('inf'), None) for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize base cases
        dp[0][0] = (0, None)
        for i in range(1, m + 1):
            dp[i][0] = (i, ('del', i-1, -1))  # deletion
        for j in range(1, n + 1):
            dp[0][j] = (j, ('ins', -1, j-1))  # insertion
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/substitution
                match_cost = 0 if src_nonpad_tokens[i-1] == pred_nonpad_tokens[j-1] else 1
                match_option = (dp[i-1][j-1][0] + match_cost, ('match' if match_cost == 0 else 'sub', i-1, j-1))
                
                # Deletion (from source)
                del_option = (dp[i-1][j][0] + 1, ('del', i-1, -1))
                
                # Insertion (to pred)
                ins_option = (dp[i][j-1][0] + 1, ('ins', -1, j-1))
                
                # Take minimum cost
                dp[i][j] = min(match_option, del_option, ins_option, key=lambda x: x[0])
        
        # Backtrack to find alignment
        alignments = []
        i, j = m, n
        while i > 0 or j > 0:
            if dp[i][j][1] is None:
                break
            op, src_idx, pred_idx = dp[i][j][1]
            if op == 'match':
                # Only record latency for matches
                alignments.append((src_idx, pred_idx))
                i -= 1
                j -= 1
            elif op == 'sub':
                # Substitution - not a match, skip
                i -= 1
                j -= 1
            elif op == 'del':
                i -= 1
            elif op == 'ins':
                j -= 1
        
        # Compute latencies for matched tokens
        # Sort alignments by source position to get first token
        alignments_sorted = sorted(alignments, key=lambda x: x[0])
        
        for idx, (src_idx, pred_idx) in enumerate(alignments_sorted):
            src_pos = int(src_nonpad_positions[src_idx])
            pred_pos = int(pred_nonpad_positions[pred_idx])
            latency = pred_pos - src_pos
            token_id = int(src_nonpad_tokens[src_idx])
            all_latencies.append({
                'latency': latency, 
                'token_id': token_id,
                'is_first_token': idx == 0  # Mark the first token
            })
    
    if not all_latencies:
        return [], 0.0, 0.0, None
    
    # Extract just latency values for statistics
    latency_values = [item['latency'] for item in all_latencies]
    mean_latency = float(np.mean(latency_values))
    median_latency = float(np.median(latency_values))
    
    # Extract first token latency (from the first batch item that has a match)
    first_token_latency = None
    for item in all_latencies:
        if item.get('is_first_token', False):
            first_token_latency = item['latency']
            break
    
    return all_latencies, mean_latency, median_latency, first_token_latency


class TokenLevelLatency:
    """
    Computes token-level latency metrics by aligning source and predicted tokens.
    Only computes latency for matched tokens (ignoring substitutions, insertions, deletions).
    Following the same pattern as TurnTakingMetrics to ensure multi-node compatibility.
    """
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        # Store Python lists like BLEU does, not tensor values
        self.all_latencies = defaultdict(list)
        self.mean_latencies = defaultdict(list)
        self.median_latencies = defaultdict(list)
        self.first_token_latencies = defaultdict(list)
    
    def reset(self):
        self.all_latencies.clear()
        self.mean_latencies.clear()
        self.median_latencies.clear()
        self.first_token_latencies.clear()
        return self
    
    def update(self, name: str, source_tokens: torch.Tensor, pred_tokens: torch.Tensor) -> None:
        """
        Update metrics with a batch of samples.
        
        Args:
            name (str): Dataset name
            source_tokens (torch.Tensor): Source tokens (batch_size, seq_len)
            pred_tokens (torch.Tensor): Predicted tokens (batch_size, seq_len)
        """
        latencies_list, mean_latency, median_latency, first_token_latency = compute_token_level_latency(
            source_tokens, pred_tokens, self.pad_token_id
        )
        
        # Store Python floats and lists
        if latencies_list:  # Only add if there were valid matches
            self.all_latencies[name].extend(latencies_list)
            self.mean_latencies[name].append(mean_latency)
            self.median_latencies[name].append(median_latency)
            if first_token_latency is not None:
                self.first_token_latencies[name].append(first_token_latency)
    
    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final metrics across all updates.
        Following the same pattern as TurnTakingMetrics.compute()
        
        Returns:
            dict: Dictionary with token_level_latency_mean, token_level_latency_median, 
                  and first_token_latency_mean metrics
        """
        import numpy as np
        corpus_metrics = {}
        
        # Get all dataset names to ensure consistent metric structure across all ranks
        all_names = set(self.all_latencies.keys()) | set(self.first_token_latencies.keys())
        
        # Compute mean and median across all accumulated latencies
        for name in all_names:
            if self.all_latencies[name]:
                # Extract latency values from dicts for statistics
                latency_values = [item['latency'] for item in self.all_latencies[name]]
                # Compute statistics across all latencies
                overall_mean = float(np.mean(latency_values))
                overall_median = float(np.median(latency_values))
                corpus_metrics[f"token_level_latency_mean_{name}"] = torch.tensor(overall_mean)
                corpus_metrics[f"token_level_latency_median_{name}"] = torch.tensor(overall_median)
            else:
                # Ensure consistent metric structure even if no data
                corpus_metrics[f"token_level_latency_mean_{name}"] = torch.tensor(0.0)
                corpus_metrics[f"token_level_latency_median_{name}"] = torch.tensor(0.0)
            
            # Compute first token latency statistics
            if self.first_token_latencies[name]:
                first_token_mean = float(np.mean(self.first_token_latencies[name]))
                corpus_metrics[f"first_token_latency_mean_{name}"] = torch.tensor(first_token_mean)
            else:
                corpus_metrics[f"first_token_latency_mean_{name}"] = torch.tensor(0.0)
        
        # Clear stored values
        self.all_latencies.clear()
        self.mean_latencies.clear()
        self.median_latencies.clear()
        self.first_token_latencies.clear()
        
        return corpus_metrics


def compute_timestamp_accuracy(source_tokens, pred_tokens, pad_token_id, user_bos_id, user_eos_id, tokenizer):
    """
    Computes word-level timestamp accuracy between source and predicted tokens.
    Extracts words from both sequences, aligns them, and computes start/end timestamp differences.
    
    For streaming ASR where tokens are frame-aligned:
    - Start timestamp: position of last pad before word starts
    - End timestamp: position of first pad after word ends
    
    Args:
        source_tokens (torch.Tensor): Batch of source sequences (batch_size, seq_len) - ground truth
        pred_tokens (torch.Tensor): Batch of predicted sequences (batch_size, seq_len) - predictions
        pad_token_id (int): Token ID to ignore (also used to find word boundaries)
        user_bos_id (int): User beginning of speech token ID
        user_eos_id (int): User end of speech token ID
        tokenizer: Tokenizer to decode tokens to text
    
    Returns:
        tuple: (word_timestamp_diffs, mean_start_diff, mean_end_diff, accuracy_within_threshold) where:
               - word_timestamp_diffs: list of dicts with 'start_diff', 'end_diff', 'word'
               - mean_start_diff: average start timestamp difference
               - mean_end_diff: average end timestamp difference
               - accuracy_within_threshold: percentage of words with both start and end within 5 frames
    """
    import numpy as np
    
    # Convert to CPU and numpy
    source_tokens = source_tokens.cpu().numpy()
    pred_tokens = pred_tokens.cpu().numpy()
    
    batch_size = source_tokens.shape[0]
    all_timestamp_diffs = []
    
    for b in range(batch_size):
        src = source_tokens[b]
        pred = pred_tokens[b]
        
        # Extract user text tokens between user_bos and user_eos
        def extract_user_tokens(tokens):
            """Extract tokens between user_bos and user_eos, return list of (token_id, position)"""
            user_tokens = []
            in_user_text = False
            for pos, token_id in enumerate(tokens):
                if token_id == user_bos_id:
                    in_user_text = True
                elif token_id == user_eos_id:
                    in_user_text = False
                elif in_user_text and token_id != pad_token_id:
                    user_tokens.append((int(token_id), int(pos)))
            return user_tokens
        
        src_user_tokens = extract_user_tokens(src)
        pred_user_tokens = extract_user_tokens(pred)
        
        if len(src_user_tokens) == 0 or len(pred_user_tokens) == 0:
            continue
        
        # Convert token positions to word-level timestamps
        # In streaming ASR, tokens are frame-aligned and pads indicate boundaries
        def tokens_to_words_with_timestamps(token_list, tokenizer, all_tokens, pad_token_id):
            """
            Convert list of (token_id, position) to list of (word, start_pos, end_pos).
            
            For streaming ASR:
            - Start timestamp: position of last pad before word starts
            - End timestamp: position of first pad after word ends
            
            Args:
                token_list: list of (token_id, position) for non-pad user tokens
                tokenizer: tokenizer to decode tokens
                all_tokens: full token sequence (includes pads) for finding boundaries
                pad_token_id: ID of pad token
            """
            if not token_list:
                return []
            
            # Decode each token individually to track word boundaries
            words_with_timestamps = []
            current_word = ""
            current_token_positions = []  # positions of tokens in current word
            
            for idx, (token_id, pos) in enumerate(token_list):
                # Decode single token
                try:
                    token_text = tokenizer.ids_to_text([token_id])
                    # Check if this token starts a new word (has leading space)
                    if token_text.startswith(' ') and current_word:
                        # Save previous word with timestamps
                        if current_word.strip() and current_token_positions:
                            start_pos = current_token_positions[0]
                            end_pos = current_token_positions[-1]
                            
                            # Find last pad before first token (start timestamp)
                            for p in range(start_pos - 1, -1, -1):
                                if all_tokens[p] == pad_token_id:
                                    start_pos = p
                                    break
                                elif all_tokens[p] != pad_token_id:
                                    # Hit another non-pad token, stop
                                    break
                            
                            # Find first pad after last token (end timestamp)
                            for p in range(end_pos + 1, len(all_tokens)):
                                if all_tokens[p] == pad_token_id:
                                    end_pos = p
                                    break
                                elif all_tokens[p] != pad_token_id:
                                    # Hit another non-pad token, stop
                                    break
                            
                            words_with_timestamps.append((current_word.strip(), start_pos, end_pos))
                        
                        # Start new word
                        current_word = token_text
                        current_token_positions = [pos]
                    else:
                        current_word += token_text
                        current_token_positions.append(pos)
                except:
                    # If decoding fails, treat as continuation
                    current_token_positions.append(pos)
            
            # Add last word with timestamps
            if current_word.strip() and current_token_positions:
                start_pos = current_token_positions[0]
                end_pos = current_token_positions[-1]
                
                # Find last pad before first token (start timestamp)
                for p in range(start_pos - 1, -1, -1):
                    if all_tokens[p] == pad_token_id:
                        start_pos = p
                        break
                    elif all_tokens[p] != pad_token_id:
                        break
                
                # Find first pad after last token (end timestamp)
                for p in range(end_pos + 1, len(all_tokens)):
                    if all_tokens[p] == pad_token_id:
                        end_pos = p
                        break
                    elif all_tokens[p] != pad_token_id:
                        break
                
                words_with_timestamps.append((current_word.strip(), start_pos, end_pos))
            
            return words_with_timestamps
        
        src_words = tokens_to_words_with_timestamps(src_user_tokens, tokenizer, src, pad_token_id)
        pred_words = tokens_to_words_with_timestamps(pred_user_tokens, tokenizer, pred, pad_token_id)
        
        if len(src_words) == 0 or len(pred_words) == 0:
            continue
        
        # Align words using dynamic programming (similar to token alignment)
        src_word_texts = [w[0] for w in src_words]
        pred_word_texts = [w[0] for w in pred_words]
        
        m, n = len(src_word_texts), len(pred_word_texts)
        dp = [[(float('inf'), None) for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize base cases
        dp[0][0] = (0, None)
        for i in range(1, m + 1):
            dp[i][0] = (i, ('del', i-1, -1))
        for j in range(1, n + 1):
            dp[0][j] = (j, ('ins', -1, j-1))
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/substitution
                match_cost = 0 if src_word_texts[i-1].lower() == pred_word_texts[j-1].lower() else 1
                match_option = (dp[i-1][j-1][0] + match_cost, 
                               ('match' if match_cost == 0 else 'sub', i-1, j-1))
                
                # Deletion
                del_option = (dp[i-1][j][0] + 1, ('del', i-1, -1))
                
                # Insertion
                ins_option = (dp[i][j-1][0] + 1, ('ins', -1, j-1))
                
                dp[i][j] = min(match_option, del_option, ins_option, key=lambda x: x[0])
        
        # Backtrack to find alignments
        alignments = []
        i, j = m, n
        while i > 0 or j > 0:
            if dp[i][j][1] is None:
                break
            op, src_idx, pred_idx = dp[i][j][1]
            if op == 'match':
                # Record timestamp difference for matched words
                # src_words and pred_words now have format: (word_text, start_pos, end_pos)
                src_start = src_words[src_idx][1]
                src_end = src_words[src_idx][2]
                pred_start = pred_words[pred_idx][1]
                pred_end = pred_words[pred_idx][2]
                
                # Compute differences for both start and end timestamps
                start_diff = abs(pred_start - src_start)
                end_diff = abs(pred_end - src_end)
                
                # Store both differences (we'll compute statistics on both)
                all_timestamp_diffs.append({
                    'start_diff': start_diff,
                    'end_diff': end_diff,
                    'word': src_words[src_idx][0]
                })
                i -= 1
                j -= 1
            elif op == 'sub':
                i -= 1
                j -= 1
            elif op == 'del':
                i -= 1
            elif op == 'ins':
                j -= 1
    
    if not all_timestamp_diffs:
        return [], 0.0, 0.0, 0.0
    
    # Extract start and end differences
    start_diffs = [item['start_diff'] for item in all_timestamp_diffs]
    end_diffs = [item['end_diff'] for item in all_timestamp_diffs]
    
    # Compute mean for start and end timestamps
    mean_start_diff = float(np.mean(start_diffs))
    mean_end_diff = float(np.mean(end_diffs))
    
    # Accuracy within threshold (5 frames = 0.4 seconds)
    # A word is considered accurate if BOTH start and end are within threshold
    threshold = 5
    within_threshold = sum(1 for item in all_timestamp_diffs 
                          if item['start_diff'] <= threshold and item['end_diff'] <= threshold)
    accuracy = within_threshold / len(all_timestamp_diffs) if all_timestamp_diffs else 0.0
    
    return all_timestamp_diffs, mean_start_diff, mean_end_diff, float(accuracy)


class TimestampAccuracy:
    """
    Computes word-level timestamp accuracy between ground truth and predicted user text.
    Tracks both start and end timestamps separately (mean differences only).
    Following the same pattern as TokenLevelLatency to ensure multi-node compatibility.
    """
    
    def __init__(self, pad_token_id: int, user_bos_id: int, user_eos_id: int, tokenizer):
        self.pad_token_id = pad_token_id
        self.user_bos_id = user_bos_id
        self.user_eos_id = user_eos_id
        self.tokenizer = tokenizer
        # Store Python lists
        self.all_diffs = defaultdict(list)
        self.accuracies = defaultdict(list)
    
    def reset(self):
        self.all_diffs.clear()
        self.accuracies.clear()
        return self
    
    def update(self, name: str, source_tokens: torch.Tensor, pred_tokens: torch.Tensor) -> None:
        """
        Update metrics with a batch of samples.
        
        Args:
            name (str): Dataset name
            source_tokens (torch.Tensor): Ground truth user tokens (batch_size, seq_len)
            pred_tokens (torch.Tensor): Predicted user tokens (batch_size, seq_len)
        """
        diffs_list, mean_start_diff, mean_end_diff, accuracy = compute_timestamp_accuracy(
            source_tokens, pred_tokens, 
            self.pad_token_id, self.user_bos_id, self.user_eos_id,
            self.tokenizer
        )
        
        # Store Python floats and lists
        if diffs_list:  # Only add if there were valid matches
            self.all_diffs[name].extend(diffs_list)
            self.accuracies[name].append(accuracy)
    
    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute final metrics across all updates.
        
        Returns:
            dict: Dictionary with metrics for start and end timestamps:
                  - timestamp_start_diff_mean_{name}: mean start timestamp difference
                  - timestamp_end_diff_mean_{name}: mean end timestamp difference
                  - timestamp_accuracy_{name}: % of words with both start and end within threshold
        """
        import numpy as np
        corpus_metrics = {}
        
        # Get all dataset names to ensure consistent metric structure
        all_names = set(self.all_diffs.keys()) | set(self.accuracies.keys())
        
        # Compute metrics for each dataset
        for name in all_names:
            if self.all_diffs[name]:
                # Extract start and end differences from all matched words
                start_diffs = [item['start_diff'] for item in self.all_diffs[name]]
                end_diffs = [item['end_diff'] for item in self.all_diffs[name]]
                
                # Compute mean for start and end timestamps
                start_mean = float(np.mean(start_diffs))
                end_mean = float(np.mean(end_diffs))
                corpus_metrics[f"timestamp_start_diff_mean_{name}"] = torch.tensor(start_mean)
                corpus_metrics[f"timestamp_end_diff_mean_{name}"] = torch.tensor(end_mean)
            else:
                # Ensure consistent metric structure even if no data
                corpus_metrics[f"timestamp_start_diff_mean_{name}"] = torch.tensor(0.0)
                corpus_metrics[f"timestamp_end_diff_mean_{name}"] = torch.tensor(0.0)
            
            # Compute accuracy (both start and end within threshold)
            if self.accuracies[name]:
                accuracy_mean = float(np.mean(self.accuracies[name]))
                corpus_metrics[f"timestamp_accuracy_{name}"] = torch.tensor(accuracy_mean)
            else:
                corpus_metrics[f"timestamp_accuracy_{name}"] = torch.tensor(0.0)
        
        # Clear stored values
        self.all_diffs.clear()
        self.accuracies.clear()
        
        return corpus_metrics

