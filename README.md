# Forge Neo Krea2 LoRA Tools + SVD Resizer

A standalone [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic) extension for inspecting, structurally stripping, and SVD-rank-resizing **Krea2** LoRAs directly from the WebUI.

It adds a new **Krea2 LoRA Tools** tab to the WebUI, with two modes:

- **SVD Resize** — analyze a LoRA's per-layer SVD energy curve, get a recommended rank for a given quality target, and re-save the LoRA at a smaller rank.
- **Structural Stripper** — remove specific groups of Krea2 diffusion-block tensors (by profile) to shrink a LoRA, with a dry-run mode and a risk-threshold safety check.

Structural analysis reads only tensor *shape/dtype metadata* from the safetensors header wherever possible, while SVD analysis loads the A/B tensor data needed for the computation.

## Features

### SVD Resize

The SVD rank-resizing implementation in this extension is a separate approach
from the structural stripping method above. It performs per-layer SVD
compression of LoRA A/B pairs and does not remove diffusion blocks by profile.

- Detects `lora_A` / `lora_B` pairs and validates that the file uses a supported Krea2 key layout (`diffusion_model.blocks.*`, `diffusion_model.txtfusion.*`, `transformer.transformer_blocks.*`, `transformer.text_fusion.*`).
- **Analyze & recommend rank**: computes the retained-energy curve for each A/B pair using a QR + small-SVD approach, then suggests a target rank for Aggressive (95%), Balanced (98%), and Conservative (99%) quality targets — without writing any file.
- **SVD Resize LoRA**: truncates every pair to the chosen target rank and saves a new `..._SVD_r<rank>.safetensors` file.
- Configurable alpha handling and compute device (`auto` / `cpu` / `cuda`).
- Dry-run mode shows the estimated size reduction before you commit to writing a file.

### Structural Stripper

The original Krea2 LoRA stripping approach was developed by **Winnougan**
in [Krea2_LoRA_Stripper](https://github.com/Winnougan/Krea2_LoRA_Stripper).

That project credits [**Puppet_Master**](https://civitai.red/user/Puppet_Master) on Civitai Red as the original source
of the technique/code for reducing Krea 2 LoRA file sizes by removing
diffusion/DIT weights while retaining the text-fusion layers.

Our Structural Stripper is an expanded Forge Neo implementation inspired by
that approach, with additional profiles, dry-run analysis, risk thresholds,
and Forge Neo UI integration.

Original source: [Puppet_Master on Civitai Red](https://civitai.red/models/2742336/nsfw-krea2-low-vram?modelVersionId=3089248)

- Four built-in profiles:
  - **Max (txtfusion only)** — strips all diffusion-block LoRA weights, keeps only text-fusion tensors.
  - **Balanced (keep 50% diffusion blocks)**
  - **Compact (keep 25% diffusion blocks)**
  - **Light (aux only)** — strips only auxiliary tensors (`first`, `last_linear`, `tmlp_`, `tproj_1`, etc.).
- **Analyze LoRA**: reports total size, per-group tensor/byte breakdown, and dtype counts.
- **Strip LoRA**: writes a new, smaller `..._<Profile>.safetensors` file. Dry-run mode is available, and a configurable risk threshold flags profiles that would strip an unusually large share of the file.

### General
- Live-streaming log output and a progress bar for long-running operations, based on parsed `N/M` progress information from the processing logs.
- Won't run processing operations while a generation job is active in Forge Neo, to avoid interfering with the model/LoRA state or contending for GPU/VRAM mid-generation.
- Scans all of Forge's known LoRA directories (`models/Lora`, `models/lora`, `models/LoRA`, `models/loras`, plus any `--lora-dir`/`--lora-dirs` set on the command line).

## Requirements

```
safetensors
torch
gradio
```

These dependencies are normally already provided by a standard Forge Neo / Stable Diffusion WebUI Forge installation, so no extra installation is usually required.

## Installation

1. Download or clone this repository into your Forge Neo `extensions/` folder:
   ```
   cd <your-forge-neo>/extensions
   git clone https://github.com/dbodik-git/forge-neo-krea2-lora-tools.git
   ```
   Replace `<your-forge-neo>` with your Forge Neo installation directory.

   (Or extract the release ZIP into `extensions/` directly.)
2. Restart Forge Neo (or use **Reload UI**).
3. A new **Krea2 LoRA Tools** tab will appear in the WebUI.

## Usage

1. Open the **Krea2 LoRA Tools** tab.
2. Pick a LoRA from the dropdown (click **🔄 Refresh** if you just added a file).
3. Choose a mode:
   - **SVD Resize** — click **Analyze & recommend rank** first to see the energy curve and a suggested rank, then set **SVD target rank** and click **SVD Resize LoRA** (or enable **SVD dry run** to preview without writing a file).
   - **Structural Stripper** — pick a **Profile**, click **Analyze LoRA** to review what would be removed, then **Strip LoRA** to write the reduced file (or enable **Dry run** first).
4. Progress and results are streamed into the log box below the buttons; output files are written alongside the source LoRA with a suffix describing what was done (e.g. `_SVD_r16`, `_Max`, `_Balanced`).

## Notes & Limitations

- Only Krea2-layout LoRAs are supported for the SVD path — a file must expose matching `lora_A`/`lora_B` pairs under one of the recognized key prefixes, or analysis/resize will report that no compatible pairs were found.
- SVD Resize builds the output tensors in memory before saving, so peak RAM/VRAM usage scales with the size of the generated output (plus whatever Forge already has resident). On memory-constrained systems, very large source LoRAs combined with a large loaded checkpoint may still hit a memory ceiling.
- This extension only processes local LoRA files already present on disk; it does not download or fetch anything from the network.

## License

License MIT
