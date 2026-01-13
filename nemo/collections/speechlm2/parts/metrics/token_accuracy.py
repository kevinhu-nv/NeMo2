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

