# Forge Neo Krea2 LoRA Tools + SVD Resizer

Standalone Forge Neo extension for analyzing, structurally stripping, and SVD-resizing Krea2 LoRAs.

## SVD Resize

The SVD tool works on Krea2 LoRA `lora_A.weight` / `lora_B.weight` pairs using an efficient QR + small-SVD decomposition. It avoids constructing the large dense delta matrix `B @ A`.

- Select a target rank from 1 to the source rank.
- Default `Match new rank` sets `ss_network_alpha` to the new rank, keeping alpha/r = 1.0. This is the recommended setting for faithful A/B comparison.
- `Preserve original alpha` is available for experiments, but changes the effective alpha/r when the rank changes.
- Output is saved as `_SVD_r{rank}.safetensors` and the source file is never modified.
- The tool reports estimated size reduction and SVD energy retained.
- Compute device can be Auto (CUDA when available), CUDA, or CPU; only small temporary float32 matrices are placed on the GPU.
- Current implementation requires a uniform LoRA rank and targets Krea2 LoRAs.

## Structural stripper

The original profiles remain available:

- Max (txtfusion only)
- Balanced (keep 50% diffusion blocks)
- Compact (keep 25% diffusion blocks)
- Light (aux only)

## Installation

Copy this folder to:

`webui/extensions/forge-neo-krea2-lora-tools/`

or extract the release ZIP there, then restart Forge.

The extension uses Forge's existing Python environment, PyTorch, Gradio, and safetensors.

## Notes

This is a personal/recovery build based on our Forge Neo Krea2 LoRA work. Review upstream licensing/attribution before public redistribution.
