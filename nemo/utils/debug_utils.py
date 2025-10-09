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

import os
import torch


def debug_breakpoint(rank=0, message=None):
    """
    A simple debug function that works in both distributed and non-distributed environments.
    
    Args:
        rank (int): The rank to pause on (default: 0)
        message (str, optional): Optional message to print before breaking
    
    Usage:
        from nemo.utils.debug_utils import debug_breakpoint
        
        # Simple usage
        debug_breakpoint()
        
        # With custom rank
        debug_breakpoint(rank=1)
        
        # With message
        debug_breakpoint(message="Check tensor shapes here")
    """
    import torch.distributed as dist
    import sys, os
    
    if message:
        print(f"[DEBUG] {message}")
    
    if dist.is_initialized():
        # Method 1: Use torch.distributed.breakpoint() (recommended for distributed)
        dist.breakpoint(rank=rank)
    else:
        # Method 2: Use conditional breakpoint with rank check
        current_rank = int(os.environ.get("RANK", 0))
        if current_rank == rank:
            try:
                from IPython import embed
                embed()

                # from ipdb.__main__ import Pdb
                # tty = open('/dev/tty')
                # Pdb(stdin=tty, stdout=sys.__stdout__).set_trace(sys._getframe())

                # print("isatty:", sys.stdin.isatty())
                # print("stdin repr:", repr(sys.stdin))
                # print("stdout repr:", repr(sys.stdout))
                # import sys, os
                # sys.stdin = open('/dev/tty')
                # import ipdb; ipdb.set_trace()
            except ImportError:
                breakpoint()


def debug_print(message, rank=0, force=False):
    """
    Print debug messages only on specified rank or force print on all ranks.
    
    Args:
        message (str): Message to print
        rank (int): Only print on this rank (default: 0)
        force (bool): Force print on all ranks regardless of rank (default: False)
    
    Usage:
        from nemo.utils.debug_utils import debug_print
        
        debug_print("Processing batch", rank=0)
        debug_print("All ranks see this", force=True)
    """
    import torch.distributed as dist
    
    if force:
        print(f"[DEBUG] {message}")
        return
    
    if dist.is_initialized():
        current_rank = dist.get_rank()
        if current_rank == rank:
            print(f"[DEBUG-R{rank}] {message}")
    else:
        current_rank = int(os.environ.get("RANK", 0))
        if current_rank == rank:
            print(f"[DEBUG-R{rank}] {message}")


def debug_tensor_info(tensor, name="tensor", rank=0):
    """
    Print tensor information for debugging.
    
    Args:
        tensor: PyTorch tensor to inspect
        name (str): Name of the tensor for the output
        rank (int): Only print on this rank (default: 0)
    
    Usage:
        from nemo.utils.debug_utils import debug_tensor_info
        
        debug_tensor_info(my_tensor, "source_audio")
    """
    import torch.distributed as dist
    
    if dist.is_initialized():
        current_rank = dist.get_rank()
        if current_rank == rank:
            print(f"[DEBUG-R{rank}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            if tensor.numel() > 0:
                print(f"[DEBUG-R{rank}] {name}: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.float().mean().item():.4f}")
    else:
        current_rank = int(os.environ.get("RANK", 0))
        if current_rank == rank:
            print(f"[DEBUG-R{rank}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            if tensor.numel() > 0:
                print(f"[DEBUG-R{rank}] {name}: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.float().mean().item():.4f}")


# Convenience alias for the main debug function
debug = debug_breakpoint 