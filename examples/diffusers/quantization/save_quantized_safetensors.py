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


def _count_blocks(state_dict_keys: list[str], prefix_template: str) -> int:
    count = 0
    while True:
        prefix = prefix_template.format(count)
        if any(k.startswith(prefix) for k in state_dict_keys):
            count += 1
            continue
        return count


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _is_flux_diffusers_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return (
        "x_embedder.weight" in state_dict
        and "transformer_blocks.0.attn.to_q.weight" in state_dict
        and "single_transformer_blocks.0.attn.to_q.weight" in state_dict
    )


def _build_flux_key_map(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    state_dict_keys = list(state_dict.keys())
    depth = _count_blocks(state_dict_keys, "transformer_blocks.{}.")
    depth_single_blocks = _count_blocks(state_dict_keys, "single_transformer_blocks.{}.")
    if "x_embedder.bias" in state_dict:
        hidden_size = state_dict["x_embedder.bias"].shape[0]
    else:
        hidden_size = state_dict["x_embedder.weight"].shape[0]

    key_map: dict[str, object] = {}

    for index in range(depth):
        prefix_from = f"transformer_blocks.{index}"
        prefix_to = f"double_blocks.{index}"

        for end in ("weight", "bias"):
            k = f"{prefix_from}.attn."
            qkv = f"{prefix_to}.img_attn.qkv.{end}"
            key_map[f"{k}to_q.{end}"] = (qkv, (0, 0, hidden_size))
            key_map[f"{k}to_k.{end}"] = (qkv, (0, hidden_size, hidden_size))
            key_map[f"{k}to_v.{end}"] = (qkv, (0, hidden_size * 2, hidden_size))

            qkv = f"{prefix_to}.txt_attn.qkv.{end}"
            key_map[f"{k}add_q_proj.{end}"] = (qkv, (0, 0, hidden_size))
            key_map[f"{k}add_k_proj.{end}"] = (qkv, (0, hidden_size, hidden_size))
            key_map[f"{k}add_v_proj.{end}"] = (qkv, (0, hidden_size * 2, hidden_size))

        block_map = {
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_out.0.bias": "img_attn.proj.bias",
            "norm1.linear.weight": "img_mod.lin.weight",
            "norm1.linear.bias": "img_mod.lin.bias",
            "norm1_context.linear.weight": "txt_mod.lin.weight",
            "norm1_context.linear.bias": "txt_mod.lin.bias",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "attn.to_add_out.bias": "txt_attn.proj.bias",
            "ff.net.0.proj.weight": "img_mlp.0.weight",
            "ff.net.0.proj.bias": "img_mlp.0.bias",
            "ff.net.2.weight": "img_mlp.2.weight",
            "ff.net.2.bias": "img_mlp.2.bias",
            "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
            "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
            "ff_context.net.2.weight": "txt_mlp.2.weight",
            "ff_context.net.2.bias": "txt_mlp.2.bias",
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        }

        for k, v in block_map.items():
            key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

    for index in range(depth_single_blocks):
        prefix_from = f"single_transformer_blocks.{index}"
        prefix_to = f"single_blocks.{index}"

        for end in ("weight", "bias"):
            k = f"{prefix_from}.attn."
            qkv = f"{prefix_to}.linear1.{end}"
            key_map[f"{k}to_q.{end}"] = (qkv, (0, 0, hidden_size))
            key_map[f"{k}to_k.{end}"] = (qkv, (0, hidden_size, hidden_size))
            key_map[f"{k}to_v.{end}"] = (qkv, (0, hidden_size * 2, hidden_size))
            key_map[f"{prefix_from}.proj_mlp.{end}"] = (
                qkv,
                (0, hidden_size * 3, hidden_size * 4),
            )

        block_map = {
            "norm.linear.weight": "modulation.lin.weight",
            "norm.linear.bias": "modulation.lin.bias",
            "proj_out.weight": "linear2.weight",
            "proj_out.bias": "linear2.bias",
            "attn.norm_q.weight": "norm.query_norm.scale",
            "attn.norm_k.weight": "norm.key_norm.scale",
        }

        for k, v in block_map.items():
            key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

    map_basic = {
        ("final_layer.linear.bias", "proj_out.bias"),
        ("final_layer.linear.weight", "proj_out.weight"),
        ("img_in.bias", "x_embedder.bias"),
        ("img_in.weight", "x_embedder.weight"),
        ("time_in.in_layer.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
        ("time_in.in_layer.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("time_in.out_layer.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
        ("time_in.out_layer.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
        ("txt_in.bias", "context_embedder.bias"),
        ("txt_in.weight", "context_embedder.weight"),
        ("vector_in.in_layer.bias", "time_text_embed.text_embedder.linear_1.bias"),
        ("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight"),
        ("vector_in.out_layer.bias", "time_text_embed.text_embedder.linear_2.bias"),
        ("vector_in.out_layer.weight", "time_text_embed.text_embedder.linear_2.weight"),
        ("guidance_in.in_layer.bias", "time_text_embed.guidance_embedder.linear_1.bias"),
        ("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight"),
        ("guidance_in.out_layer.bias", "time_text_embed.guidance_embedder.linear_2.bias"),
        ("guidance_in.out_layer.weight", "time_text_embed.guidance_embedder.linear_2.weight"),
        ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", _swap_scale_shift),
        ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", _swap_scale_shift),
        ("pos_embed_input.bias", "controlnet_x_embedder.bias"),
        ("pos_embed_input.weight", "controlnet_x_embedder.weight"),
    }

    for k in map_basic:
        if len(k) > 2:
            key_map[k[1]] = (k[0], None, k[2])
        else:
            key_map[k[1]] = k[0]

    return key_map


def _convert_flux_state_dict(
    state_dict: dict[str, torch.Tensor], key_map: dict[str, object]
) -> dict[str, torch.Tensor]:
    out_sd: dict[str, torch.Tensor] = {}
    remaining = dict(state_dict)

    for key, target in key_map.items():
        weight = remaining.pop(key, None)
        if weight is None:
            continue

        if isinstance(target, tuple):
            target_key = target[0]
            offset = target[1] if len(target) > 1 else None
            fun = target[2] if len(target) > 2 else None
            if fun is not None:
                weight = fun(weight)

            if offset is not None:
                old_weight = out_sd.get(target_key)
                if old_weight is None:
                    old_weight = torch.empty_like(weight)
                if old_weight.shape[offset[0]] < offset[1] + offset[2]:
                    expanded = list(weight.shape)
                    expanded[offset[0]] = offset[1] + offset[2]
                    new_weight = torch.empty(
                        expanded, device=weight.device, dtype=weight.dtype
                    )
                    new_weight[: old_weight.shape[0]] = old_weight
                    old_weight = new_weight
                view = old_weight.narrow(offset[0], offset[1], offset[2])
                view.copy_(weight)
                out_sd[target_key] = old_weight
            else:
                out_sd[target_key] = weight
        else:
            out_sd[target] = weight

    return out_sd


def _map_layer_name(layer_name: str, key_map: dict[str, object]) -> str | None:
    key = f"{layer_name}.weight"
    target = key_map.get(key)
    if target is None:
        return None
    target_key = target[0] if isinstance(target, tuple) else target
    if target_key.endswith(".weight"):
        return target_key[: -len(".weight")]
    if target_key.endswith(".bias"):
        return target_key[: -len(".bias")]
    return None


def _collect_quant_targets(backbone: torch.nn.Module, key_map: dict[str, object]):
    quant_targets: dict[str, dict[str, object]] = {}
    input_scales: dict[str, torch.Tensor] = {}
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

        target_layer = _map_layer_name(layer_name, key_map)
        if target_layer is None:
            skipped_layers += 1
            continue

        block_size = None
        if fmt == "nvfp4":
            block_sizes = getattr(weight_quantizer, "block_sizes", None) or {}
            block_size = block_sizes.get(-1) or block_sizes.get(weight.dim() - 1) or 16

        existing = quant_targets.get(target_layer)
        if existing is None:
            quant_targets[target_layer] = {"format": fmt, "block_size": block_size}
        else:
            if existing.get("format") != fmt:
                continue

        input_quantizer = getattr(module, "input_quantizer", None)
        input_scale = _get_input_scale(input_quantizer)
        if input_scale is not None and target_layer not in input_scales:
            input_scales[target_layer] = input_scale

    return quant_targets, input_scales, skipped_layers


def _quantize_weight_from_float(
    weight: torch.Tensor, fmt: str, block_size: int | None
):
    if fmt == "nvfp4":
        if block_size is None:
            block_size = 16
        q_tensor, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(
            weight,
            block_size,
            weights_scaling_factor_2=None,
            keep_high_precision=False,
            try_tensorrt=False,
        )
        return q_tensor._quantized_data, weight_scale, weight_scale_2

    if fmt == "float8_e4m3fn":
        q_tensor, weight_scale = FP8QTensor.quantize(
            weight,
            scales=None,
            axis=None,
            block_sizes=None,
        )
        return q_tensor._quantized_data, weight_scale, None

    raise ValueError(f"Unsupported quant format: {fmt}")


def _quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    quant_targets: dict[str, dict[str, object]],
    input_scales: dict[str, torch.Tensor],
    logger: logging.Logger,
):
    out_sd: dict[str, torch.Tensor] = {}
    quant_layers: dict[str, dict[str, str]] = {}

    for key, tensor in state_dict.items():
        if key.endswith(".weight"):
            layer_name = key[: -len(".weight")]
            target = quant_targets.get(layer_name)
            if target and tensor.ndim == 2:
                fmt = target.get("format")
                block_size = target.get("block_size")
                try:
                    q_weight, weight_scale, weight_scale_2 = _quantize_weight_from_float(
                        tensor, fmt, block_size
                    )
                except Exception as exc:
                    logger.warning(f"Skipping layer {layer_name}: {exc}")
                    out_sd[key] = tensor
                    continue

                out_sd[key] = q_weight
                if fmt == "nvfp4":
                    out_sd[f"{layer_name}.weight_scale"] = weight_scale.to(
                        dtype=torch.float8_e4m3fn
                    )
                    out_sd[f"{layer_name}.weight_scale_2"] = weight_scale_2.to(
                        dtype=torch.float32
                    )
                else:
                    out_sd[f"{layer_name}.weight_scale"] = weight_scale.to(
                        dtype=torch.float32
                    )

                input_scale = input_scales.get(layer_name)
                if input_scale is not None:
                    out_sd[f"{layer_name}.input_scale"] = input_scale.to(
                        dtype=torch.float32
                    )

                quant_layers[layer_name] = {"format": fmt}
                continue

        out_sd[key] = tensor

    for layer_name in quant_targets:
        if layer_name not in quant_layers:
            logger.warning(f"Quantized layer missing in export: {layer_name}")

    return out_sd, quant_layers


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
    logger.info("Version: 2026-01-14-v4 (Flux ComfyUI mapping + quant)")
    logger.info("=" * 80)

    logger.info("Extracting quantized weights and scales from backbone...")
    raw_state_dict = backbone.state_dict()
    filtered_state_dict = {
        k: v for k, v in raw_state_dict.items() if not _is_quantizer_key(k)
    }

    if _is_flux_diffusers_state_dict(filtered_state_dict):
        logger.info("Detected FLUX diffusers checkpoint, converting to ComfyUI naming...")
        key_map = _build_flux_key_map(filtered_state_dict)
        converted_state_dict = _convert_flux_state_dict(filtered_state_dict, key_map)
        quant_targets, input_scales, skipped_layers = _collect_quant_targets(
            backbone, key_map
        )
        state_dict, quant_layers = _quantize_state_dict(
            converted_state_dict, quant_targets, input_scales, logger
        )
        raw_key_count = len(raw_state_dict)
        logger.info(
            f"   Raw keys: {raw_key_count} | Converted keys: {len(converted_state_dict)}"
        )
    else:
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
