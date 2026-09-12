import os

import torch
import safetensors

from forge_neo_lora_core import inspect_svd_krea2_lora

DEFAULT_RANKS = (4, 8, 12, 16, 20, 24, 28, 32)
ENERGY_TARGETS = (
    ("Aggressive", 95.0),
    ("Balanced", 98.0),
    ("Conservative", 99.0),
)


def _resolve_device(device):
    device = str(device or "auto").lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available in this Forge Python environment.")
    return device


def _recommended_rank(energy_by_rank, threshold):
    for rank in range(1, len(energy_by_rank)):
        if energy_by_rank[rank] >= threshold:
            return rank
    return len(energy_by_rank) - 1


def analyze_svd_rank_profile(model_path, device="auto", log=lambda _message: None):
    """Analyze the SVD energy curve without writing a LoRA.

    One QR + small-SVD pass per A/B pair yields the retained-energy curve for
    every possible target rank. No truncated LoRA is created during analysis.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"LoRA not found: {model_path}")

    info = inspect_svd_krea2_lora(model_path)
    source_rank = info["rank"]
    pairs = info["pairs"]
    device = _resolve_device(device)

    energy_by_rank = torch.zeros(source_rank + 1, dtype=torch.float64)
    total_energy = 0.0
    source_ab_bytes = 0
    target_bytes_per_rank = 0
    total_pairs = len(pairs)

    log(f"SVD rank analysis: {total_pairs} A/B pairs, source rank {source_rank}")
    log(f"Compute device: {device}")

    with torch.inference_mode():
        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
            for processed, (a_key, b_key) in enumerate(pairs, start=1):
                a_orig = f.get_tensor(a_key)
                b_orig = f.get_tensor(b_key)
                source_ab_bytes += a_orig.nelement() * a_orig.element_size()
                source_ab_bytes += b_orig.nelement() * b_orig.element_size()
                target_bytes_per_rank += (a_orig.shape[1] + b_orig.shape[0]) * a_orig.element_size()

                a = a_orig.to(device=device, dtype=torch.float32)
                b = b_orig.to(device=device, dtype=torch.float32)
                qb, rb = torch.linalg.qr(b, mode="reduced")
                qa, ra = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
                core = rb @ ra.transpose(0, 1)
                singular_values = torch.linalg.svdvals(core)

                s2 = singular_values.square().to(dtype=torch.float64)
                cumulative = torch.cumsum(s2, dim=0)
                total = float(cumulative[-1].item()) if cumulative.numel() else 0.0
                total_energy += total

                max_rank = min(source_rank, cumulative.numel())
                energy_by_rank[1:max_rank + 1] += cumulative[:max_rank].cpu()

                del a_orig, b_orig, a, b, qb, rb, qa, ra, core, singular_values, s2, cumulative

                if processed == 1 or processed == total_pairs or processed % 16 == 0:
                    log(f"SVD analysis progress: {processed}/{total_pairs} pairs")

    if total_energy > 0:
        energy_pct_by_rank = [0.0] + [
            100.0 * float(energy_by_rank[r].item()) / total_energy
            for r in range(1, source_rank + 1)
        ]
    else:
        energy_pct_by_rank = [100.0] * (source_rank + 1)

    recommendations = {
        label: _recommended_rank(energy_pct_by_rank, threshold)
        for label, threshold in ENERGY_TARGETS
    }

    output_ranks = sorted(set(
        r for r in DEFAULT_RANKS if r <= source_rank
    ) | set(recommendations.values()) | {source_rank})

    before_bytes = os.path.getsize(model_path)
    estimated_sizes = {
        rank: max(0, before_bytes - source_ab_bytes + rank * target_bytes_per_rank)
        for rank in output_ranks
    }

    return {
        "source_rank": source_rank,
        "pairs": total_pairs,
        "device": device,
        "layout": info["layout"],
        "energy_pct_by_rank": energy_pct_by_rank,
        "recommendations": recommendations,
        "ranks": output_ranks,
        "estimated_sizes": estimated_sizes,
        "original_size": before_bytes,
    }
