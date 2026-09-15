import os
import gc
from collections import Counter

import torch
import safetensors
import safetensors.torch

LORA_OUTPUT_SUFFIX = "_stripped"
SVD_OUTPUT_SUFFIX = "_SVD_r{rank}"
LORA_RISK_THRESHOLD_PCT = 10.0
KREA2_LORA_SIGNATURE = "diffusion_model.txtfusion."
# Krea2 checkpoints/LoRAs exist in multiple key-layout variants.  The
# SVD path accepts only these known Krea2 layouts rather than treating any
# arbitrary LoRA containing A/B pairs as Krea2.
KREA2_LAYOUT_PREFIXES = (
    "diffusion_model.blocks.",
    "diffusion_model.txtfusion.",
    "transformer.transformer_blocks.",
    "transformer.text_fusion.",
)
KREA2_LORA_STRIP_PREFIXES_RAW = ("blocks", "first", "last_linear", "last.linear", "tmlp_", "tproj_1")
KREA2_LORA_STRIP_PREFIXES = tuple("diffusion_model." + name for name in KREA2_LORA_STRIP_PREFIXES_RAW)
KREA2_AUX_PREFIXES = tuple("diffusion_model." + name for name in ("first", "last_linear", "last.linear", "tmlp_", "tproj_1"))
KREA2_LORA_PROFILES = {
    "Max (txtfusion only)": KREA2_LORA_STRIP_PREFIXES,
    "Balanced (keep 50% diffusion blocks)": "__sample_blocks_50__",
    "Compact (keep 25% diffusion blocks)": "__sample_blocks_25__",
    "Light (aux only)": KREA2_AUX_PREFIXES,
}


def _noop_logger(_message):
    pass


def _mb(v):
    return v / (1024 * 1024)


# Byte size per element for the dtype spellings we might see back from
# safetensors' get_slice().get_dtype() -- covers both the safetensors-native
# short names (F32, BF16, ...) and torch-style names (float32, bfloat16, ...)
# in case that differs across safetensors/torch versions.
_DTYPE_BYTE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "FLOAT64": 8, "FLOAT32": 4, "FLOAT16": 2, "BFLOAT16": 2,
    "INT64": 8, "INT32": 4, "INT16": 2, "INT8": 1, "UINT8": 1,
}


def _slice_dtype_and_itemsize(tensor_slice):
    """Return (dtype_name, bytes_per_element) for a safe_open().get_slice()
    object using only header metadata -- no tensor payload is read from disk.

    This is what makes byte-size / dtype reporting on a multi-GB LoRA fast:
    get_tensor(key) has to read and materialize that tensor's full data,
    while get_slice(key) only reads the (shape, dtype) recorded in the
    safetensors header. Falls back to loading the tensor just this once if a
    safetensors version ever reports a dtype spelling outside the table
    above, so this can never silently mis-size a tensor.
    """
    dtype_name = str(tensor_slice.get_dtype()).replace("torch.", "")
    itemsize = _DTYPE_BYTE_SIZES.get(dtype_name.upper())
    if itemsize is not None:
        return dtype_name, itemsize
    tensor = tensor_slice[:]
    return str(tensor.dtype).replace("torch.", ""), tensor.element_size()


def _slice_nbytes_and_dtype(tensor_slice):
    """Total byte size and dtype name of a get_slice() object, from shape
    metadata alone (see _slice_dtype_and_itemsize)."""
    dtype_name, itemsize = _slice_dtype_and_itemsize(tensor_slice)
    numel = 1
    for dim in tensor_slice.get_shape():
        numel *= int(dim)
    return numel * itemsize, dtype_name


def _sample_block_keep_set(total_blocks, fraction):
    if total_blocks <= 0:
        return set()
    keep_count = max(1, int(round(total_blocks * fraction)))
    if keep_count >= total_blocks:
        return set(range(total_blocks))
    picks = []
    for i in range(keep_count):
        idx = int(round(i * (total_blocks - 1) / max(1, keep_count - 1))) if keep_count > 1 else 0
        if idx not in picks:
            picks.append(idx)
    for idx in range(total_blocks):
        if len(picks) >= keep_count:
            break
        if idx not in picks:
            picks.append(idx)
    return set(sorted(picks[:keep_count]))


def _resolve_krea2_profile(profile, keys):
    name = str(profile or "Max (txtfusion only)")
    aliases = {
        "max": "Max (txtfusion only)",
        "balanced": "Balanced (keep 50% diffusion blocks)",
        "compact": "Compact (keep 25% diffusion blocks)",
        "light": "Light (aux only)",
    }
    name = aliases.get(name.strip().lower(), name)
    block_prefix = "diffusion_model.blocks."
    block_ids = set()
    for key in keys:
        if key.startswith(block_prefix):
            token = key[len(block_prefix):].split('.', 1)[0]
            if token.isdigit():
                block_ids.add(int(token))
    total_blocks = (max(block_ids) + 1) if block_ids else 0

    if name == "Balanced (keep 50% diffusion blocks)":
        keep = _sample_block_keep_set(total_blocks, 0.50)
        def should_strip(key):
            if key.startswith(block_prefix):
                token = key[len(block_prefix):].split('.', 1)[0]
                return token.isdigit() and int(token) not in keep
            return any(key.startswith(p) for p in KREA2_AUX_PREFIXES)
        return name, should_strip, total_blocks, keep

    if name == "Compact (keep 25% diffusion blocks)":
        keep = _sample_block_keep_set(total_blocks, 0.25)
        def should_strip(key):
            if key.startswith(block_prefix):
                token = key[len(block_prefix):].split('.', 1)[0]
                return token.isdigit() and int(token) not in keep
            return any(key.startswith(p) for p in KREA2_AUX_PREFIXES)
        return name, should_strip, total_blocks, keep

    if name == "Light (aux only)":
        def should_strip(key):
            return any(key.startswith(p) for p in KREA2_AUX_PREFIXES)
        return name, should_strip, total_blocks, set()

    def should_strip(key):
        return any(key.startswith(p) for p in KREA2_LORA_STRIP_PREFIXES)
    return "Max (txtfusion only)", should_strip, total_blocks, set()


def inspect_krea2_lora(path, profile="Max (txtfusion only)", log=_noop_logger):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LoRA not found: {path}")
    try:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            metadata = f.metadata() or {}
        profile_name, should_strip, block_count, keep_blocks = _resolve_krea2_profile(profile, keys)
        is_krea2 = any(k.startswith(KREA2_LORA_SIGNATURE) for k in keys)
        total_bytes = strip_bytes = keep_count = strip_count = 0
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Only byte-size accounting is needed here, so read shape/
                # dtype from the header (see _slice_nbytes_and_dtype) instead
                # of loading each tensor's actual payload -- on a large LoRA
                # this turns a full-file read into an instant metadata scan.
                nbytes, _ = _slice_nbytes_and_dtype(f.get_slice(key))
                total_bytes += nbytes
                if should_strip(key):
                    strip_count += 1
                    strip_bytes += nbytes
                else:
                    keep_count += 1
        gc.collect()
        return {
            "metadata": metadata, "is_krea2": is_krea2, "profile": profile_name,
            "block_count": block_count, "keep_blocks": keep_blocks,
            "total_count": len(keys), "keep_count": keep_count, "strip_count": strip_count,
            "total_bytes": total_bytes, "strip_bytes": strip_bytes, "keep_bytes": total_bytes - strip_bytes,
        }
    except Exception as exc:
        raise RuntimeError(f"Could not read LoRA '{os.path.basename(path)}': {exc}") from exc


def analyze_krea2_lora(model_path, log=_noop_logger):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LoRA not found: {model_path}")
    before_bytes = os.path.getsize(model_path)
    group_stats = {}
    key_rows = []
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        is_krea2 = any(k.startswith(KREA2_LORA_SIGNATURE) for k in keys)
        total_tensor_bytes = 0
        dtype_counts = Counter()
        for key in keys:
            # This loop used to call f.get_tensor(key) for every tensor in
            # the file just to compute byte sizes -- i.e. a full read of the
            # entire LoRA's data on every "Analyze" click. Shape/dtype from
            # the header is all this needs.
            tensor_bytes, dtype_name = _slice_nbytes_and_dtype(f.get_slice(key))
            total_tensor_bytes += tensor_bytes
            dtype_counts[dtype_name] += 1
            parts = key.split('.')
            group = '.'.join(parts[:3]) if len(parts) >= 3 else ('.'.join(parts[:2]) if len(parts) >= 2 else parts[0])
            stat = group_stats.setdefault(group, {"count": 0, "bytes": 0})
            stat["count"] += 1
            stat["bytes"] += tensor_bytes
            key_rows.append((key, tensor_bytes, dtype_name))
    gc.collect()

    lines = [
        "Krea2 LoRA ANALYSIS", "=" * 72,
        f"File: {os.path.basename(model_path)}",
        f"Size on disk: {_mb(before_bytes):.2f} MB",
        f"Tensors: {len(keys)}",
        f"Tensor bytes: {_mb(total_tensor_bytes):.2f} MB",
        f"Krea2 signature: {'YES' if is_krea2 else 'NO'}",
        "", "Metadata:",
        f"  rank: {metadata.get('ss_network_dim', '?')}",
        f"  alpha: {metadata.get('ss_network_alpha', '?')}",
        "", "Dtype counts:",
    ]
    for dtype, count in sorted(dtype_counts.items()):
        lines.append(f"  {dtype}: {count}")
    lines.extend(["", "Top groups by tensor bytes:"])
    for group, stat in sorted(group_stats.items(), key=lambda x: x[1]["bytes"], reverse=True)[:30]:
        pct = 100.0 * stat["bytes"] / total_tensor_bytes if total_tensor_bytes else 0.0
        lines.append(f"  {group}: {stat['count']} tensors, {_mb(stat['bytes']):.2f} MB ({pct:.2f}%)")
    lines.extend(["", "Largest individual tensors:"])
    for key, tensor_bytes, dtype in sorted(key_rows, key=lambda x: x[1], reverse=True)[:25]:
        lines.append(f"  {_mb(tensor_bytes):.2f} MB  {dtype:<10} {key}")
    if is_krea2:
        lines.extend(["", "Use SVD Resize to reduce LoRA rank while preserving the strongest components."])
    else:
        lines.extend(["", f"This file does not contain the expected Krea2 txtfusion signature."])
    status = "\n".join(lines)
    log(status)
    return status


def strip_krea2_lora(model_path, output_suffix=LORA_OUTPUT_SUFFIX, risk_threshold_pct=LORA_RISK_THRESHOLD_PCT,
                     dry_run=False, profile="Max (txtfusion only)", log=_noop_logger):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LoRA not found: {model_path}")
    suffix = str(output_suffix or LORA_OUTPUT_SUFFIX).strip() or LORA_OUTPUT_SUFFIX
    if not suffix.startswith('_'):
        suffix = '_' + suffix
    before_bytes = os.path.getsize(model_path)
    before_mb = _mb(before_bytes)
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    profile_name, should_strip, block_count, keep_blocks = _resolve_krea2_profile(profile, keys)
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        total_bytes = strip_bytes = 0
        strip_count = keep_count = 0
        for key in f.keys():
            # Stats-only pre-pass (used for the risk-ratio check and the
            # dry-run report) -- shape/dtype metadata is enough, no need to
            # load every tensor's actual data a second time before the real
            # write loop further down even attempts anything.
            nbytes, _ = _slice_nbytes_and_dtype(f.get_slice(key))
            total_bytes += nbytes
            if should_strip(key):
                strip_count += 1
                strip_bytes += nbytes
            else:
                keep_count += 1
    info = {
        "is_krea2": any(k.startswith(KREA2_LORA_SIGNATURE) for k in keys),
        "total_count": len(keys), "keep_count": keep_count, "strip_count": strip_count,
        "total_bytes": total_bytes, "strip_bytes": strip_bytes, "keep_bytes": total_bytes - strip_bytes,
    }
    if not info["is_krea2"]:
        return (f"Skipped (not Krea2 / no '{KREA2_LORA_SIGNATURE}' tensors found)\n"
                f"Input: {os.path.basename(model_path)}\nOriginal size: {before_mb:.2f} MB\nNo output was created."), None
    kept_pct = 100.0 * info["keep_bytes"] / info["total_bytes"] if info["total_bytes"] else 0.0
    risky = kept_pct < float(risk_threshold_pct)
    if profile_name.startswith(('Balanced', 'Compact')):
        log(f"Profile block plan: {block_count} total, {len(keep_blocks)} kept diffusion blocks: {sorted(keep_blocks)}")
    log(f"Krea2 signature found. Tensors: {info['total_count']} total, {info['strip_count']} to strip, {info['keep_count']} to keep")
    log(f"Kept-byte ratio: {kept_pct:.2f}%" + (" [higher fidelity-loss risk]" if risky else ""))
    if dry_run:
        return (f"Dry run OK (no file written)\nInput: {os.path.basename(model_path)}\nOriginal size: {before_mb:.2f} MB\n"
                f"Would remove: {info['strip_count']} tensors\nWould keep: {info['keep_count']} tensors ({kept_pct:.2f}% of tensor bytes)\n"
                f"Risk threshold: {float(risk_threshold_pct):.1f}%" + (" [!] HIGHER FIDELITY-LOSS RISK" if risky else "")), None
    output_path = os.path.splitext(model_path)[0] + suffix + '.safetensors'
    kept_tensors = {}
    try:
        log("Stripping DIT/UNET tensors...")
        total_keys = info["total_count"]
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for idx, key in enumerate(f.keys(), start=1):
                tensor = f.get_tensor(key)
                if should_strip(key):
                    del tensor
                    continue
                kept_tensors[key] = tensor
                # Periodic progress line (same "N/M tensors" shape the UI's
                # progress bar already parses for the SVD tool) so large
                # strips don't look stalled for the whole run.
                if idx == 1 or idx == total_keys or idx % 64 == 0:
                    log(f"Strip progress: {idx}/{total_keys} tensors")
        if not kept_tensors:
            raise RuntimeError("Refusing to save an empty LoRA: no tensors remain after stripping.")
        log(f"Saving: {output_path}")
        safetensors.torch.save_file(kept_tensors, output_path, metadata=metadata)
    finally:
        del kept_tensors
        gc.collect()
    after_bytes = os.path.getsize(output_path)
    after_mb = _mb(after_bytes)
    reduction = 100.0 * (1 - after_bytes / before_bytes) if before_bytes else 0.0
    status = (f"Success (Krea2 LoRA stripped)\nProfile: {profile_name}\nInput: {os.path.basename(model_path)}\n"
              f"Original size: {before_mb:.2f} MB\nNew size: {after_mb:.2f} MB ({reduction:.1f}% smaller)\n"
              f"Tensors: removed {info['strip_count']}, kept {info['keep_count']}\nKept-byte ratio: {kept_pct:.2f}%" +
              ("  [!] LOW -- higher risk of fidelity loss; A/B test recommended" if risky else "") +
              f"\nSaved to: {output_path}")
    log(status)
    return status, output_path


def _find_lora_pairs(keys):
    key_set = set(keys)
    pairs = []
    for key in keys:
        if not key.endswith('.lora_A.weight'):
            continue
        b_key = key[:-len('.lora_A.weight')] + '.lora_B.weight'
        if b_key in key_set:
            pairs.append((key, b_key))
    return pairs


def _krea2_layout_for_pairs(pairs):
    """Return a known Krea2 layout name, or None.

    We deliberately require the A/B pairs to live under known Krea2
    transformer/text-fusion namespaces.  This prevents a generic LoRA with
    ordinary lora_A/lora_B tensors from being accepted merely because it has
    matrix pairs.
    """
    if not pairs:
        return None
    pair_keys = [a for a, _ in pairs]
    has_diffusion = any(
        k.startswith(("diffusion_model.blocks.", "diffusion_model.txtfusion."))
        for k in pair_keys
    )
    has_transformer = any(
        k.startswith(("transformer.transformer_blocks.", "transformer.text_fusion."))
        for k in pair_keys
    )

    # A single known namespace is enough; mixed layouts are also accepted
    # because some converted files can contain both families.
    if has_transformer and has_diffusion:
        return "krea2/mixed"
    if has_transformer:
        return "krea2/transformer"
    if has_diffusion:
        return "krea2/diffusion_model"
    return None


def _metadata_rank_alpha(metadata, source_rank):
    rank = metadata.get('ss_network_dim')
    alpha = metadata.get('ss_network_alpha')
    try:
        rank = int(rank) if rank is not None else source_rank
    except (TypeError, ValueError):
        rank = source_rank
    if rank != source_rank:
        # Tensor shapes are authoritative for the actual factors.
        rank = source_rank
    return rank, (str(alpha) if alpha is not None else str(source_rank))


def inspect_svd_krea2_lora(model_path):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LoRA not found: {model_path}")
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}

    pairs = _find_lora_pairs(keys)
    if not pairs:
        raise RuntimeError("No lora_A/lora_B pairs were found.")

    layout = _krea2_layout_for_pairs(pairs)
    if layout is None:
        raise RuntimeError(
            "A/B LoRA pairs were found, but their key layout is not a recognized Krea2 layout. "
            "Expected diffusion_model.blocks/txtfusion or transformer.transformer_blocks/text_fusion namespaces."
        )

    ranks = []
    dtypes = Counter()
    # This function runs before every SVD-related action (Analyze &
    # recommend rank, SVD dry-run, SVD resize), so it's the single hottest
    # path in the plugin. It only ever needs each pair's SHAPE (for the rank/
    # consistency check) and dtype NAME (informational only, unused
    # elsewhere) -- get_slice() reads both from the safetensors header
    # without loading the tensor payload, unlike get_tensor(). Previously
    # this loaded every A/B pair's full data just to inspect .shape, which
    # is a full read of the entire LoRA's tensor bytes before any real work
    # (or even the rank-recommendation report) could start.
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for a_key, b_key in pairs:
            a_slice = f.get_slice(a_key)
            b_slice = f.get_slice(b_key)
            a_shape = tuple(a_slice.get_shape())
            b_shape = tuple(b_slice.get_shape())
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise RuntimeError(
                    f"Unsupported LoRA pair shapes: {a_key}: {a_shape}, "
                    f"{b_key}: {b_shape}"
                )
            ranks.append(int(a_shape[0]))
            dtypes[str(a_slice.get_dtype()).replace('torch.', '')] += 1

    if len(set(ranks)) != 1:
        raise RuntimeError(f"Mixed LoRA ranks are not supported by this SVD resizer: {sorted(set(ranks))}")

    source_rank = ranks[0]
    meta_rank, meta_alpha = _metadata_rank_alpha(metadata, source_rank)
    # A file can have no metadata (seen in the wild), so report the shape-
    # derived rank and a safe alpha default rather than '?' for processing.
    return {
        "is_krea2": True,
        "keys": keys,
        "metadata": metadata,
        "pairs": pairs,
        "rank": source_rank,
        "dtypes": dict(dtypes),
        "layout": layout,
        "metadata_rank": meta_rank,
        "alpha": meta_alpha,
    }


def svd_resize_krea2_lora(model_path, target_rank, alpha_mode="match_rank", dry_run=False,
                           output_suffix=None, log=_noop_logger, device="auto"):
    """Truncate each Krea2 LoRA A/B pair with an efficient QR+small-SVD decomposition.

    The calculation is performed in float32 on CPU and the resulting factors are cast
    back to the original tensor dtype. With alpha_mode='match_rank', metadata alpha is
    set to the new rank so alpha/r remains 1.0, matching the original rank-32/alpha-32
    effective scale of the source LoRA.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LoRA not found: {model_path}")
    info = inspect_svd_krea2_lora(model_path)

    source_rank = info["rank"]
    try:
        target_rank = int(target_rank)
    except Exception as exc:
        raise ValueError("Target rank must be an integer.") from exc
    if target_rank < 1 or target_rank > source_rank:
        raise ValueError(f"Target rank must be between 1 and {source_rank}.")
    if target_rank == source_rank:
        log(f"Target rank equals source rank ({source_rank}); no dimensional reduction will occur.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available in this Forge Python environment.")

    before_bytes = os.path.getsize(model_path)
    # Single pass computing both the source payload and the target-rank
    # estimate, from header shape/dtype metadata only (get_slice(), not
    # get_tensor()) -- this runs before dry-run and before the real SVD work
    # even starts, so it shouldn't cost a full read of the source LoRA.
    estimated_tensor_bytes = 0
    source_tensor_bytes = 0
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for a_key, b_key in info["pairs"]:
            a_slice = f.get_slice(a_key)
            b_slice = f.get_slice(b_key)
            a_shape = tuple(a_slice.get_shape())
            b_shape = tuple(b_slice.get_shape())
            _, dtype_bytes = _slice_dtype_and_itemsize(a_slice)
            a_numel = a_shape[0] * a_shape[1]
            b_numel = b_shape[0] * b_shape[1]
            estimated_tensor_bytes += target_rank * (a_shape[1] + b_shape[0]) * dtype_bytes
            source_tensor_bytes += a_numel * dtype_bytes + b_numel * dtype_bytes

    original_alpha = info["alpha"]
    if alpha_mode == "match_rank":
        new_alpha = str(target_rank)
        alpha_note = f"alpha={new_alpha} (effective alpha/r = 1.0; recommended)"
    elif alpha_mode == "preserve_alpha":
        new_alpha = str(original_alpha)
        alpha_note = f"alpha={new_alpha} (metadata preserved; effective strength may change)"
    else:
        raise ValueError("Unknown alpha mode.")

    suffix = output_suffix or SVD_OUTPUT_SUFFIX.format(rank=target_rank)
    if not suffix.startswith('_'):
        suffix = '_' + suffix
    output_path = os.path.splitext(model_path)[0] + suffix + '.safetensors'

    est_reduction = 100.0 * (1.0 - estimated_tensor_bytes / source_tensor_bytes) if source_tensor_bytes else 0.0
    log(f"Compute device: {device}")
    log(f"Krea2 LoRA: {len(info['pairs'])} A/B pairs, source rank {source_rank} -> target rank {target_rank}")
    log(f"Estimated tensor payload: {_mb(source_tensor_bytes):.2f} MiB -> {_mb(estimated_tensor_bytes):.2f} MiB ({est_reduction:.1f}% smaller; headers add a small amount)")
    log(alpha_note)

    if dry_run:
        return (f"Dry run OK (no file written)\nInput: {os.path.basename(model_path)}\n"
                f"Rank: {source_rank} -> {target_rank}\nPairs: {len(info['pairs'])}\n"
                f"Estimated tensor payload: {_mb(source_tensor_bytes):.2f} MiB -> {_mb(estimated_tensor_bytes):.2f} MiB\n"
                f"Estimated tensor reduction: {est_reduction:.1f}%\n{alpha_note}"), None

    out_tensors = {}
    total_energy = 0.0
    kept_energy = 0.0
    processed = 0
    try:
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            for key in f.keys():
                # Non-A/B tensors are copied untouched.
                if key not in {p[0] for p in info["pairs"]} and key not in {p[1] for p in info["pairs"]}:
                    out_tensors[key] = f.get_tensor(key)

            for a_key, b_key in info["pairs"]:
                a_orig = f.get_tensor(a_key)
                b_orig = f.get_tensor(b_key)
                original_dtype = a_orig.dtype
                if a_orig.ndim != 2 or b_orig.ndim != 2 or a_orig.shape[0] != b_orig.shape[1]:
                    raise RuntimeError(f"Unsupported pair shapes: {a_key}: {tuple(a_orig.shape)}, {b_key}: {tuple(b_orig.shape)}")

                # QR+SVD avoids constructing the huge dense delta matrix B@A.
                a = a_orig.to(device=device, dtype=torch.float32)
                b = b_orig.to(device=device, dtype=torch.float32)
                with torch.inference_mode():
                    qb, rb = torch.linalg.qr(b, mode='reduced')
                    qa, ra = torch.linalg.qr(a.transpose(0, 1), mode='reduced')
                    core = rb @ ra.transpose(0, 1)
                    u, s, vh = torch.linalg.svd(core, full_matrices=False)
                    r = min(target_rank, s.numel())
                    u_r = qb @ u[:, :r]
                    b_new = u_r * s[:r]
                    a_new = vh[:r, :] @ qa.transpose(0, 1)

                # Numerical energy of the original delta matrix, represented by s.
                s2 = s.square()
                total_energy += float(s2.sum().item())
                kept_energy += float(s2[:r].sum().item())

                out_tensors[b_key] = b_new.to(device='cpu', dtype=original_dtype).contiguous()
                out_tensors[a_key] = a_new.to(device='cpu', dtype=original_dtype).contiguous()
                del a_orig, b_orig, a, b, qb, rb, qa, ra, core, u, s, vh, u_r, b_new, a_new
                processed += 1
                if processed == 1 or processed == len(info["pairs"]) or processed % 16 == 0:
                    log(f"SVD progress: {processed}/{len(info['pairs'])} pairs")
                # gc.collect() is deliberately NOT called per-pair here: a full
                # generational collection scales with total heap size (which in
                # a Forge process holding a loaded checkpoint can be large), so
                # calling it once per A/B pair turned an O(pairs) loop into an
                # O(pairs * heap_size) one. One collection every 32 pairs bounds
                # memory growth without that per-iteration cost.
                if processed % 32 == 0:
                    gc.collect()

            metadata["ss_network_dim"] = str(target_rank)
            metadata["ss_network_alpha"] = new_alpha
            metadata["modelspec.architecture"] = metadata.get("modelspec.architecture", "krea2/lora")
            metadata["modelspec.title"] = metadata.get("modelspec.title", "")

        if total_energy > 0:
            energy_pct = 100.0 * kept_energy / total_energy
        else:
            energy_pct = 100.0
        log(f"SVD energy retained: {energy_pct:.4f}%")
        log(f"Saving: {output_path}")
        safetensors.torch.save_file(out_tensors, output_path, metadata=metadata)
    finally:
        del out_tensors
        gc.collect()

    after_bytes = os.path.getsize(output_path)
    reduction = 100.0 * (1.0 - after_bytes / before_bytes) if before_bytes else 0.0
    status = (f"Success (Krea2 LoRA SVD resized)\n"
              f"Input: {os.path.basename(model_path)}\n"
              f"Rank: {source_rank} -> {target_rank}\n"
              f"Pairs: {len(info['pairs'])}\n"
              f"Original size: {_mb(before_bytes):.2f} MiB\n"
              f"New size: {_mb(after_bytes):.2f} MiB ({reduction:.1f}% smaller)\n"
              f"SVD energy retained: {energy_pct:.4f}%\n"
              f"{alpha_note}\n"
              f"Saved to: {output_path}")
    log(status)
    return status, output_path
