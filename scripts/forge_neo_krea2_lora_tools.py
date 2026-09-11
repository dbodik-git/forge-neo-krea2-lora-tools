import os
import re
import gradio as gr

from modules import paths, shared, script_callbacks
from forge_neo_lora_core import (
    analyze_krea2_lora,
    strip_krea2_lora,
    svd_resize_krea2_lora,
    inspect_svd_krea2_lora,
    KREA2_LORA_PROFILES,
)

_LORA_CHOICES = {}
_PROGRESS_RE = re.compile(r"(?:SVD|Strip) progress:\s*(\d+)\s*/\s*(\d+)", re.I)

_PROFILE_SUFFIXES = {
    "Max (txtfusion only)": "_Max",
    "Balanced (keep 50% diffusion blocks)": "_Balanced",
    "Compact (keep 25% diffusion blocks)": "_Compact",
    "Light (aux only)": "_Light",
}


def _as_dir_list(value):
    if not value:
        return []
    if isinstance(value, (str, bytes, os.PathLike)):
        return [os.fspath(value)]
    try:
        return [os.fspath(x) for x in value if x]
    except TypeError:
        return [os.fspath(value)]


def _existing_unique_dirs(dirs):
    out, seen = [], set()
    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(os.fspath(d))
        key = d.lower()
        if key in seen or not os.path.isdir(d):
            continue
        seen.add(key)
        out.append(d)
    return out


def _candidate_lora_dirs():
    dirs = [
        os.path.join(paths.models_path, 'Lora'),
        os.path.join(paths.models_path, 'lora'),
        os.path.join(paths.models_path, 'LoRA'),
        os.path.join(paths.models_path, 'loras'),
        *_as_dir_list(getattr(shared.cmd_opts, 'lora_dir', None)),
        *_as_dir_list(getattr(shared.cmd_opts, 'lora_dirs', None)),
    ]
    return _existing_unique_dirs(dirs)


def _list_loras(roots):
    choices = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith('.safetensors'):
                    full = os.path.abspath(os.path.join(dirpath, name))
                    rel = os.path.relpath(full, root).replace('\\', '/')
                    label = rel
                    n = 2
                    while label in choices:
                        label = f'{rel} [{n}]'
                        n += 1
                    choices[label] = full
    return choices


def _refresh():
    global _LORA_CHOICES
    _LORA_CHOICES = _list_loras(_candidate_lora_dirs())
    vals = list(_LORA_CHOICES)
    return gr.update(choices=vals, value=vals[0] if vals else None)


def _resolve(name):
    return _LORA_CHOICES.get(name, name)


def _analyze(name):
    if not name:
        return 'No LoRA selected. Click Refresh.'
    try:
        return analyze_krea2_lora(_resolve(name))
    except Exception as e:
        return f'ERROR: {e}'


def _inspect_svd(name):
    if not name:
        return 'No LoRA selected. Click Refresh.'
    try:
        info = inspect_svd_krea2_lora(_resolve(name))
        return (f"Krea2 SVD READY\n"
                f"Pairs: {len(info['pairs'])}\n"
                f"Source rank: {info['rank']}\n"
                f"Dtypes: {info['dtypes']}\n"
                f"Layout: {info['layout']}\n"
                f"Architecture: {info['metadata'].get('modelspec.architecture', info['layout'])}\n"
                f"Alpha: {info['alpha']}")
    except Exception as e:
        return f'ERROR: {e}'


def _progress_from_log(progress, msg):
    text = str(msg)
    match = _PROGRESS_RE.search(text)
    if match and progress is not None:
        current = int(match.group(1))
        total = max(1, int(match.group(2)))
        progress(min(1.0, current / total), desc=f"{current}/{total}")


def _strip(name, threshold, dry, profile, progress=gr.Progress(track_tqdm=False)):
    if not name:
        return 'No LoRA selected. Click Refresh.'
    logs = []
    def log(msg):
        logs.append(str(msg))
        _progress_from_log(progress, msg)
    suffix = _PROFILE_SUFFIXES.get(profile, '_Processed')
    try:
        if progress is not None:
            progress(0.0, desc='Preparing stripper...')
        status, _ = strip_krea2_lora(
            _resolve(name), suffix, float(threshold), bool(dry), profile, log=log
        )
        if progress is not None:
            progress(1.0, desc='Strip complete')
        return '\n'.join(logs + [status])
    except Exception as e:
        return '\n'.join(logs + [f'ERROR: {e}'])


def _svd(name, rank, alpha_mode, dry, device, progress=gr.Progress(track_tqdm=False)):
    if not name:
        return 'No LoRA selected. Click Refresh.'
    logs = []
    def log(msg):
        logs.append(str(msg))
        _progress_from_log(progress, msg)
    try:
        if progress is not None:
            progress(0.0, desc='Preparing SVD...')
        status, _ = svd_resize_krea2_lora(
            _resolve(name), int(rank), alpha_mode, bool(dry), log=log, device=str(device or 'auto').lower()
        )
        if progress is not None:
            progress(1.0, desc='SVD complete')
        return '\n'.join(logs + [status])
    except Exception as e:
        return '\n'.join(logs + [f'ERROR: {e}'])


def _lora_tab_ui():
    _refresh()
    choices = list(_LORA_CHOICES)

    with gr.Blocks() as ui:
        gr.Markdown('## 🐈‍⬛ Krea2 LoRA Tools')
        gr.Markdown(
            'Analyze, structurally strip, or SVD-resize Krea2 LoRAs. '
            'SVD keeps the strongest low-rank components and writes a new file; the source is never modified. '
            'Supports both `diffusion_model.*` and `transformer.*` Krea2 key layouts.'
        )

        with gr.Row():
            lora_name = gr.Dropdown(
                label='Krea2 LoRA', choices=choices,
                value=choices[0] if choices else None, scale=8,
            )
            refresh_loras = gr.Button('🔄 Refresh', scale=1)

        gr.Markdown('### 🔬 SVD Resizer')
        with gr.Row():
            svd_device = gr.Dropdown(
                label='Compute device',
                choices=[('Auto (CUDA if available)', 'auto'), ('CUDA', 'cuda'), ('CPU', 'cpu')],
                value='auto', scale=2,
            )
            svd_rank = gr.Slider(
                minimum=1, maximum=32, value=17, step=1,
                label='SVD target rank', scale=2,
            )
            svd_alpha = gr.Dropdown(
                label='Alpha handling',
                choices=[
                    ('Match new rank (recommended)', 'match_rank'),
                    ('Preserve original alpha', 'preserve_alpha'),
                ],
                value='match_rank', scale=3,
            )
            svd_dry = gr.Checkbox(label='SVD dry run', value=False, scale=1)

        with gr.Row():
            svd_inspect_button = gr.Button('Check SVD compatibility')
            svd_button = gr.Button('🔬 SVD Resize LoRA')

        svd_output = gr.Textbox(label='SVD log / result', value='Ready.', lines=16, interactive=False)

        gr.Markdown('### 🧰 Legacy structural stripper')
        gr.Markdown('Structural profiles are kept as a separate legacy method. For fidelity-preserving size reduction, prefer SVD.')
        with gr.Row():
            lora_profile = gr.Dropdown(
                label='Profile', choices=list(KREA2_LORA_PROFILES.keys()),
                value='Max (txtfusion only)', scale=2,
            )
            lora_risk_threshold = gr.Slider(
                minimum=0, maximum=100, value=10, step=0.5,
                label='Risk threshold (%)', scale=1,
            )
            lora_dry_run = gr.Checkbox(label='Dry run', value=False, scale=1)
        with gr.Row():
            lora_analyze_button = gr.Button('Analyze Krea2 LoRA')
            lora_strip_button = gr.Button('Strip Krea2 LoRA')
        lora_log_output = gr.Textbox(label='Stripper log / result', value='Ready.', lines=16, interactive=False)

        refresh_loras.click(fn=_refresh, outputs=[lora_name])
        svd_inspect_button.click(fn=_inspect_svd, inputs=[lora_name], outputs=[svd_output])
        svd_button.click(fn=_svd, inputs=[lora_name, svd_rank, svd_alpha, svd_dry, svd_device], outputs=[svd_output])
        lora_analyze_button.click(fn=_analyze, inputs=[lora_name], outputs=[lora_log_output])
        lora_strip_button.click(
            fn=_strip,
            inputs=[lora_name, lora_risk_threshold, lora_dry_run, lora_profile],
            outputs=[lora_log_output],
        )

    return [(ui, 'Krea2 LoRA Tools', 'krea2_lora_tools')]


script_callbacks.on_ui_tabs(_lora_tab_ui)
