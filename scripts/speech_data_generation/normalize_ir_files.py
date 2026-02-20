#!/usr/bin/env python3
"""
Normalize impulse response files to prevent signal attenuation during convolution.
This ensures that convolution preserves signal energy.
"""

import glob
import os
import sys
import numpy as np
import soundfile as sf

def normalize_ir_files(ir_folder, output_folder=None, method='peak', target_gain_db=0.0):
    """
    Normalize IR files to prevent signal attenuation.
    
    Args:
        ir_folder: Input folder with IR files
        output_folder: Output folder (if None, overwrites originals)
        method: 'peak' (normalize to peak=1.0) or 'energy' (preserve energy)
        target_gain_db: Target gain in dB (0 = no change, positive = amplify)
    """
    
    if not os.path.exists(ir_folder):
        print(f"ERROR: Folder not found: {ir_folder}")
        return
    
    ir_files = glob.glob(os.path.join(ir_folder, "*.wav"))
    
    if not ir_files:
        print(f"ERROR: No .wav files found in {ir_folder}")
        return
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Normalizing {len(ir_files)} IR files from {ir_folder} to {output_folder}")
    else:
        print(f"Normalizing {len(ir_files)} IR files in-place in {ir_folder}")
        output_folder = ir_folder
    
    print(f"Method: {method}, Target gain: {target_gain_db} dB")
    print("=" * 80)
    
    for ir_path in ir_files:
        try:
            ir, sr = sf.read(ir_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(ir.shape) > 1:
                ir = np.mean(ir, axis=1)
            
            original_peak = np.abs(ir).max()
            original_rms = np.sqrt(np.mean(ir ** 2))
            
            # Normalize based on method
            if method == 'peak':
                # Normalize so peak = 1.0, then apply target gain
                if original_peak > 0:
                    ir_normalized = ir / original_peak
                else:
                    ir_normalized = ir
            
            elif method == 'energy':
                # Normalize to preserve energy (sum of squares = 1.0)
                energy = np.sum(ir ** 2)
                if energy > 0:
                    ir_normalized = ir / np.sqrt(energy)
                else:
                    ir_normalized = ir
            
            elif method == 'rms':
                # Normalize to target RMS
                target_rms = 0.1  # Reasonable RMS for IR
                if original_rms > 0:
                    ir_normalized = ir * (target_rms / original_rms)
                else:
                    ir_normalized = ir
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Apply target gain
            if target_gain_db != 0.0:
                gain_linear = 10 ** (target_gain_db / 20)
                ir_normalized = ir_normalized * gain_linear
            
            # Ensure no clipping
            ir_normalized = np.clip(ir_normalized, -1.0, 1.0)
            
            new_peak = np.abs(ir_normalized).max()
            new_rms = np.sqrt(np.mean(ir_normalized ** 2))
            
            # Save
            output_path = os.path.join(output_folder, os.path.basename(ir_path))
            sf.write(output_path, ir_normalized, sr, subtype='PCM_16')
            
            print(f"✓ {os.path.basename(ir_path):50s} | "
                  f"peak: {original_peak:.6f} → {new_peak:.6f}, "
                  f"rms: {original_rms:.6f} → {new_rms:.6f}")
        
        except Exception as e:
            print(f"❌ ERROR processing {os.path.basename(ir_path)}: {e}")
    
    print("=" * 80)
    print(f"✓ Normalization complete! Files saved to: {output_folder}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize IR files to prevent signal attenuation")
    parser.add_argument("input_folder", help="Folder containing IR wav files")
    parser.add_argument("--output", "-o", default=None, help="Output folder (default: overwrite originals)")
    parser.add_argument("--method", "-m", choices=['peak', 'energy', 'rms'], default='peak',
                        help="Normalization method (default: peak)")
    parser.add_argument("--gain", "-g", type=float, default=0.0,
                        help="Additional gain in dB (default: 0.0)")
    
    args = parser.parse_args()
    
    normalize_ir_files(args.input_folder, args.output, args.method, args.gain)

