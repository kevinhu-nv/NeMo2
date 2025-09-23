#!/bin/bash

# Environment setup for safer distributed debugging

echo "Setting up environment for safer distributed debugging..."

# CUDA debugging options
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# Python debugging
export PYTHONBREAKPOINT=ipdb.set_trace
export PYTHONUNBUFFERED=1

# Install debugging tools
pip install ipdb ipython --quiet

echo "Environment variables set:"
echo "  CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "  TORCH_USE_CUDA_DSA=$TORCH_USE_CUDA_DSA"
echo "  PYTHONBREAKPOINT=$PYTHONBREAKPOINT"
echo "  TORCH_DISTRIBUTED_DEBUG=$TORCH_DISTRIBUTED_DEBUG"

echo ""
echo "Usage:"
echo "1. For regular debugging: from nemo.utils.debug_utils import debug"
echo "2. For safer debugging: from nemo.utils.debug_utils import debug_safe"
echo ""
echo "The debug_safe function avoids distributed barrier issues." 