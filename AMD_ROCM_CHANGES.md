# AMD ROCm Anpassungen

## Übersicht der Änderungen

### 1. `pyproject.toml`
- PyTorch-Index auf ROCm 6.2.4 geändert (`https://download.pytorch.org/whl/rocm6.2.4`)
- Optionale Abhängigkeit `flash-attn-rocm` hinzugefügt
- Projektname zu `autoresearch-amd` geändert

### 2. `train.py`
- **Flash Attention**: Dynamische Erkennung von AMD vs NVIDIA GPUs
  - AMD: Versucht `flash-attn-rocm` zu laden
  - NVIDIA: Verwendet `kernels` Paket (Flash Attention 3)
  - Fallback: Eager Attention Implementation
  
- **GPU-Erkennung**: `get_device()` und `get_gpu_peak_flops()` Funktionen
  - Automatische Erkennung von AMD GPUs (MI300X, MI250X, MI210, MI100)
  - Automatische Erkennung von NVIDIA GPUs (H100, A100, L40S, RTX 4090/3090)
  - MFU-Berechnung mit GPU-spezifischen Peak FLOPS

- **Device-Handling**: ROCm verwendet dieselben `torch.cuda` APIs
  - `torch.cuda.synchronize()` - mit hasattr-Check
  - `torch.cuda.max_memory_allocated()` - mit Fallback

### 3. `prepare.py`
- `pin_memory=True` nur wenn CUDA verfügbar
- Keine anderen Änderungen nötig (ROCm verwendet gleiche APIs)

### 4. `README.md` & `program.md`
- Dokumentation für AMD ROCm Unterstützung hinzugefügt
- Liste der unterstützten AMD GPUs

## Installation

```bash
# Repository klonen
git clone https://github.com/nikay99/autoresearch-amd.git
cd autoresearch-amd

# Abhängigkeiten installieren (PyTorch mit ROCm)
uv sync

# Optional: Flash Attention für AMD installieren
pip install flash-attn-rocm

# Daten vorbereiten
uv run prepare.py

# Training starten
uv run train.py
```

## Unterstützte GPUs

### AMD (ROCm 6.2.4+)
- MI300X (~1.3 PFLOPS BF16)
- MI250X (~383 TFLOPS BF16)
- MI210 (~181 TFLOPS BF16)
- MI100 (~184 TFLOPS BF16)

### NVIDIA (CUDA, weiterhin unterstützt)
- H100/H200 (~989 TFLOPS BF16)
- A100 (~312 TFLOPS BF16)
- L40S (~183 TFLOPS BF16)
- RTX 4090 (~82.6 TFLOPS BF16)
- RTX 3090 (~71 TFLOPS BF16)

## Hinweise

- Ohne `flash-attn-rocm` wird automatisch auf Eager Attention zurückgegriffen
- ROCm verwendet dieselben PyTorch APIs wie CUDA (`torch.cuda.*`)
- Die Performance mit Eager Attention ist langsamer als mit Flash Attention
