#!/usr/bin/env python3
"""
Helper function to save quantized models directly to SafeTensors with ComfyUI metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor, QTensorWrapper


def _is_quantizer_key(key: str) -> bool:
    return any(
        pattern in key
        for pattern in (
            ".weight_quantizer.",
            ".input_quantizer.",
            ".output_quantizer.",
            ".q_bmm_quantizer.",
            ".k_bmm_quantizer.",
            ".v_bmm_quantizer.",
            ".softmax_quantizer.",
            ".bmm2_output_quantizer.",
        )
    )


def _get_quant_format(weight_quantizer) -> str | None:
    num_bits = getattr(weight_quantizer, "num_bits", None)
    block_sizes = getattr(weight_quantizer, "block_sizes", None) or {}
    if num_bits == (2, 1) and block_sizes.get("scale_bits") == (4, 3):
        return "nvfp4"
    if num_bits == (4, 3):
        return "float8_e4m3fn"
    return None


def _get_input_scale(input_quantizer):
    if input_quantizer is None or not getattr(input_quantizer, "is_enabled", False):
        return None
    amax = None
    if hasattr(input_quantizer, "export_amax"):
        amax = input_quantizer.export_amax()
    if amax is None and hasattr(input_quantizer, "amax"):
        amax = input_quantizer.amax
    if amax is None:
        return None
    maxbound = getattr(input_quantizer, "maxbound", None)
    if maxbound is None:
        return None
    scale = amax.float() / (maxbound * 448.0)
    if scale.numel() == 1:
        scale = scale.squeeze()
    return scale


def _quantize_weight(weight, weight_quantizer, layer_name: str, fmt: str):
    if isinstance(weight, QTensorWrapper):
        q_weight = weight.data
        if fmt == "nvfp4":
            w_scale = getattr(weight_quantizer, "_scale", None)
            w_scale_2 = getattr(weight_quantizer, "_double_scale", None)
            if w_scale is None or w_scale_2 is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            return q_weight, w_scale, w_scale_2
        if fmt == "float8_e4m3fn":
            w_scale = getattr(weight_quantizer, "_scale", None)
            if w_scale is None:
                raise ValueError(f"Missing FP8 scale for layer {layer_name}")
            return q_weight, w_scale, None
        raise ValueError(f"Unsupported quant format: {fmt}")

    if fmt == "nvfp4":
        block_sizes = getattr(weight_quantizer, "block_sizes", {}) or {}
        block_size = block_sizes.get(-1) or block_sizes.get(weight.dim() - 1)
        if block_size is None:
            raise ValueError(f"Missing NVFP4 block size for layer {layer_name}")
        weight_scale_2 = None
        if hasattr(weight_quantizer, "amax") and weight_quantizer.amax is not None:
            weight_scale_2 = weight_quantizer.amax.float() / (448.0 * 6.0)
        q_tensor, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(
            weight,
            block_size,
            weights_scaling_factor_2=weight_scale_2,
            keep_high_precision=False,
            try_tensorrt=False,
        )
        return q_tensor._quantized_data, weight_scale, weight_scale_2

    if fmt == "float8_e4m3fn":
        scales = None
        if hasattr(weight_quantizer, "amax") and weight_quantizer.amax is not None:
            scales = weight_quantizer.amax.float() / 448.0
        q_tensor, weight_scale = FP8QTensor.quantize(
            weight,
            scales=scales,
            axis=getattr(weight_quantizer, "axis", None),
            block_sizes=getattr(weight_quantizer, "block_sizes", None),
        )
        return q_tensor._quantized_data, weight_scale, None

    raise ValueError(f"Unsupported quant format: {fmt}")


def _build_comfy_state_dict(backbone: torch.nn.Module, logger: logging.Logger):
    raw_state_dict = backbone.state_dict()
    state_dict = {k: v for k, v in raw_state_dict.items() if not _is_quantizer_key(k)}

    quant_layers = {}
    skipped_layers = 0

    for layer_name, module in backbone.named_modules():
        if not layer_name:
            continue
        weight = getattr(module, "weight", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if weight is None or weight_quantizer is None:
            continue
        if not getattr(weight_quantizer, "is_enabled", False):
            continue
        if weight.ndim != 2:
            continue

        fmt = _get_quant_format(weight_quantizer)
        if fmt is None:
            continue

        try:
            q_weight, weight_scale, weight_scale_2 = _quantize_weight(
                weight.detach(), weight_quantizer, layer_name, fmt
            )
        except Exception as exc:
            skipped_layers += 1
            logger.warning(f"Skipping layer {layer_name}: {exc}")
            continue

        state_dict[f"{layer_name}.weight"] = q_weight
        bias = getattr(module, "bias", None)
        if bias is not None:
            state_dict[f"{layer_name}.bias"] = bias

        if fmt == "nvfp4":
            state_dict[f"{layer_name}.weight_scale"] = weight_scale.to(
                dtype=torch.float8_e4m3fn
            )
            state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.to(
                dtype=torch.float32
            )
        else:
            state_dict[f"{layer_name}.weight_scale"] = weight_scale.to(
                dtype=torch.float32
            )

        input_quantizer = getattr(module, "input_quantizer", None)
        input_scale = _get_input_scale(input_quantizer)
        if input_scale is not None:
            state_dict[f"{layer_name}.input_scale"] = input_scale.to(dtype=torch.float32)

        quant_layers[layer_name] = {"format": fmt}

    return state_dict, quant_layers, skipped_layers, len(raw_state_dict)


def save_quantized_safetensors(
    backbone: torch.nn.Module,
    output_path: Path,
    quant_format: str = "nvfp4",
    quant_algo: str = "svdquant",
    logger: logging.Logger | None = None,
):
    """
    Save quantized model directly to SafeTensors with ComfyUI-compatible metadata.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("USING FIXED SAFETENSORS SAVER - ComfyUI export enabled")
    logger.info("Version: 2026-01-14-v3 (Quantized weights + metadata)")
    logger.info("=" * 80)

    logger.info("Extracting quantized weights and scales from backbone...")
    state_dict, quant_layers, skipped_layers, raw_key_count = _build_comfy_state_dict(
        backbone, logger
    )
    logger.info(
        f"   Raw keys: {raw_key_count} | Exported keys: {len(state_dict)}"
    )
    logger.info(
        f"   Quantized layers: {len(quant_layers)} | Skipped: {skipped_layers}"
    )

    if not quant_layers:
        sample_keys = list(backbone.state_dict().keys())[:10]
        logger.error("ERROR: No quantized layers found after export.")
        logger.error(f"State dict sample: {sample_keys}")
        raise ValueError("No quantized layers found - export aborted.")

    logger.info("Adding ComfyUI-compatible quantization metadata...")
    for layer_name, layer_config in quant_layers.items():
        comfy_quant_key = f"{layer_name}.comfy_quant"
        comfy_quant_value = json.dumps(layer_config).encode("utf-8")
        state_dict[comfy_quant_key] = torch.tensor(
            list(comfy_quant_value), dtype=torch.uint8
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "layers": quant_layers,
                "format": quant_format,
                "quant_algo": quant_algo,
            }
        )
    }

    logger.info(f"Saving quantized model to SafeTensors: {output_path}")
    save_file(state_dict, str(output_path), metadata=metadata)

    if output_path.exists():
        file_size_gb = output_path.stat().st_size / (1024**3)
        logger.info("SafeTensors file saved successfully.")
        logger.info(f"Path: {output_path}")
        logger.info(f"Size: {file_size_gb:.2f} GB")
        logger.info(f"Quantized layers: {len(quant_layers)}")
        return True

    logger.error("Failed to save SafeTensors file.")
    return False
