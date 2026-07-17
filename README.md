# UVRC — Ultimate Vocal Remover Colab

State-of-the-art audio source separation using deep learning, packaged for Google Colab and CLI use.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/UVRC/blob/main/UVRC.ipynb)

## Features

- **80+ pre-trained models** — vocals, instrumental, drums, de-reverb, denoise, karaoke, crowd removal, guitar, male/female separation, and more
- **10 model architectures** — MelBand-Roformer, BS-Roformer, SCNet, MDX23C, HTDemucs, BandIt, Segmentation Models, TorchSeg, Swin-UperNet
- **CLI & Colab** — use from the command line or the interactive Colab notebook
- **FLAC & WAV output** — with configurable PCM bit depth
- **Test-Time Augmentation** — optional 3-pass TTA for improved quality
- **GPU & CPU** — automatic CUDA detection with CPU fallback

## Installation

```bash
pip install git+https://github.com/BF667/UVRC.git
```

## Quick Start (CLI)

```bash
# Process a single file
uvr-cli \
    --model_type mel_band_roformer \
    --config_path config.yaml \
    --start_check_point model.ckpt \
    --input_file song.mp3 \
    --store_dir output/ \
    --extract_instrumental \
    --flac_file

# List available models
uvr-cli --list_models --sort_by vocals --limit 10
```

## Programmatic Usage (Python)

```python
from UVRC.multi import resolve_model, MODEL_LIST
from UVRC.inference import proc_file

# See all available models
print(MODEL_LIST)

# Resolve a model (downloads config + checkpoint automatically)
model_type, config_path, ckpt_path = resolve_model(
    "VOCALS-MelBand-Roformer (by KimberleyJSN)",
    chunk_size=485100,
    overlap=4,
)
```

## Credits

- [BF667](https://github.com/BF667) — UVRC package & Colab integration
- [Anjok07](https://github.com/anjok07) — UVR5 Creator
- [ZFTurbo](https://github.com/ZFTurbo) — Music-Source-Separation-Training base code

## License

MIT
