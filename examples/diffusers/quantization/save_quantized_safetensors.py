#!/usr/bin/env python3
"""
Helper function to save quantized models directly to SafeTensors with ComfyUI metadata.
Import and use this instead of relying on post-processing conversion.
"""

import json
import logging
from pathlib import Path
import torch
from safetensors.torch import save_file


def save_quantized_safetensors(
    backbone: torch.nn.Module,
    output_path: Path,
    quant_format: str = "nvfp4",
    quant_algo: str = "svdquant",
    logger: logging.Logger = None
):
    """
    Save quantized model directly to SafeTensors with ComfyUI-compatible metadata.

    Args:
        backbone: Quantized model backbone
        output_path: Path to save .safetensors file
        quant_format: Quantization format ('nvfp4' or 'float8_e4m3fn')
        quant_algo: Quantization algorithm used
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # PROOF OF FIX - Added by Claude 2026-01-14
    logger.info("=" * 80)
    logger.info("🔥 USING FIXED SAFETENSORS SAVER - Claude's Fix Applied! 🔥")
    logger.info("   Version: 2026-01-14-v2 (Direct SafeTensors Export)")
    logger.info("   This proves the new code is running!")
    logger.info("=" * 80)

    logger.info(f"📋 Extracting quantized state dict from backbone...")

    # Get state dict with quantized weights
    state_dict = backbone.state_dict()

    # Check what we got
    logger.info(f"   Total keys in state dict: {len(state_dict)}")

    # Verify we have quantization data
    quant_layers = {}
    weight_scale_keys = []
    input_scale_keys = []

    for key in state_dict.keys():
        if '.weight_scale' in key:
            weight_scale_keys.append(key)
            layer_name = key.replace('.weight_scale', '')
            if layer_name not in quant_layers:
                quant_layers[layer_name] = {"format": quant_format}

        if '.input_scale' in key:
            input_scale_keys.append(key)
            layer_name = key.replace('.input_scale', '')
            if layer_name not in quant_layers:
                quant_layers[layer_name] = {"format": quant_format}

    logger.info(f"   Found {len(weight_scale_keys)} weight_scale keys")
    logger.info(f"   Found {len(input_scale_keys)} input_scale keys")
    logger.info(f"   Total quantized layers: {len(quant_layers)}")

    if not quant_layers:
        logger.error("❌ ERROR: No quantized layers found in state dict!")
        logger.error("   The model may not be properly quantized.")
        logger.error("   State dict keys sample:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            logger.error(f"     {key}: dtype={state_dict[key].dtype}")
        raise ValueError("No quantized layers found - model is not quantized!")

    # Add ComfyUI-compatible .comfy_quant metadata to state dict
    logger.info(f"🔧 Adding ComfyUI-compatible quantization metadata...")
    for layer_name, layer_config in quant_layers.items():
        comfy_quant_key = f"{layer_name}.comfy_quant"
        comfy_quant_value = json.dumps(layer_config).encode('utf-8')
        state_dict[comfy_quant_key] = torch.tensor(list(comfy_quant_value), dtype=torch.uint8)

    logger.info(f"   Added .comfy_quant keys for {len(quant_layers)} layers")

    # Build SafeTensors header metadata
    metadata = {
        "_quantization_metadata": json.dumps({
            "layers": quant_layers,
            "format": quant_format,
            "quant_algo": quant_algo,
            "version": "1.0"
        })
    }

    # Save to SafeTensors
    logger.info(f"💾 Saving quantized model to SafeTensors: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    if output_path.exists():
        file_size_gb = output_path.stat().st_size / (1024**3)
        logger.info(f"✅ SafeTensors file saved successfully!")
        logger.info(f"   Path: {output_path}")
        logger.info(f"   Size: {file_size_gb:.2f} GB")
        logger.info(f"   Format: {quant_format}")
        logger.info(f"   Algorithm: {quant_algo}")
        logger.info(f"   Quantized layers: {len(quant_layers)}")
        return True
    else:
        logger.error(f"❌ Failed to save SafeTensors file!")
        return False
