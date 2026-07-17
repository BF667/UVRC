# coding: utf-8
"""
UVRC CLI — audio source separation tool with model leaderboard.

Entry point: ``uvr-cli`` (registered in pyproject.toml).
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# --- Package-relative imports ------------------------------------------------
# We prefer proper package imports; fall back to sys.path for editable installs
try:
    from UVRC.utils import demix, get_model_from_config
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    if _current_dir not in sys.path:
        sys.path.append(_current_dir)
    from utils import demix, get_model_from_config

logger = logging.getLogger(__name__)

# Only suppress specific noisy warnings, not all
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# ---------------------------------------------------------------------------
# Args dataclass
# ---------------------------------------------------------------------------

class Args:
    """Simple container for inference parameters."""

    def __init__(
        self,
        input_file: str,
        store_dir: str,
        model_type: str,
        extract_instrumental: bool,
        disable_detailed_pbar: bool,
        flac_file: bool,
        pcm_type: Optional[str],
        use_tta: bool,
    ):
        self.input_file = input_file
        self.model_type = model_type
        self.store_dir = store_dir
        self.extract_instrumental = extract_instrumental
        self.disable_detailed_pbar = disable_detailed_pbar
        self.flac_file = flac_file
        self.pcm_type = pcm_type
        self.use_tta = use_tta


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def run_file(model, args, config, device, verbose=False):
    """Process a single audio file through the separation model."""
    start_time = time.time()
    model.eval()

    if not os.path.isfile(args.input_file):
        logger.error("File not found: %s", args.input_file)
        return

    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    os.makedirs(args.store_dir, exist_ok=True)

    logger.info("Starting processing track: %s", args.input_file)
    try:
        # Use the sample rate from config if available, otherwise default to 44100
        sr = getattr(config.audio, "samplerate", 44100)
        mix, sr = librosa.load(args.input_file, sr=sr, mono=False)
    except Exception as e:
        logger.error("Cannot read track: %s — %s", args.input_file, e)
        return

    # Convert mono to stereo if needed
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if "normalize" in config.inference:
        if config.inference["normalize"] is True:
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

    if args.use_tta:
        # Original, channel inverse, polarity inverse
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1.0 * mix.copy()]
    else:
        track_proc_list = [mix.copy()]

    full_result = []
    for mix in track_proc_list:
        waveforms = demix(
            config, model, mix, device, pbar=verbose, model_type=args.model_type
        )
        full_result.append(waveforms)

    # Average all TTA passes
    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] = waveforms[el] / len(full_result)

    # Create 'instrumental' stem if requested
    if args.extract_instrumental:
        instr = "vocals" if "vocals" in instruments else instruments[0]
        instruments.append("instrumental")
        waveforms["instrumental"] = mix_orig - waveforms[instr]

    for instr in instruments:
        estimates = waveforms[instr].T
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                estimates = estimates * std + mean
        file_name, _ = os.path.splitext(os.path.basename(args.input_file))
        if args.flac_file:
            output_file = os.path.join(args.store_dir, f"{file_name}_{instr}.flac")
            subtype = "PCM_16" if args.pcm_type == "PCM_16" else "PCM_24"
            sf.write(output_file, estimates, sr, subtype=subtype)
        else:
            output_file = os.path.join(args.store_dir, f"{file_name}_{instr}.wav")
            sf.write(output_file, estimates, sr, subtype="FLOAT")

    time.sleep(1)
    logger.info("Elapsed time: %.2f sec", time.time() - start_time)


# ---------------------------------------------------------------------------
# Model manager (leaderboard / remote metadata)
# ---------------------------------------------------------------------------

class ModelManager:
    """Manager for handling model listing, configurations, and links."""

    def __init__(self):
        self.cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "audio-separator"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.urls = {
            "download_checks": "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json",
            "model_scores": "https://raw.githubusercontent.com/TRvlvr/application_data/main/model_data/model_scores.json",
            "vr_model_data": "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json",
            "mdx_model_data": "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json",
        }

        self.repo_urls = {
            "public": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models",
            "vip": "https://github.com/Anjok0109/ai_magic/releases/download/v5",
            "audio_separator": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs",
        }

        self.model_data_cache = {}

    def fetch_json_data(self, url_key: str, cache_filename: str) -> Dict:
        """Fetch JSON data from URL with local caching."""
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("Loaded cached data from %s", cache_filename)
                    return data
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupted cache file %s — re-fetching", cache_filename)

        try:
            logger.info("Fetching %s from %s …", url_key, self.urls[url_key])
            response = requests.get(self.urls[url_key], timeout=30)
            response.raise_for_status()
            data = response.json()

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return data
        except Exception as e:
            logger.error("Error fetching %s: %s", url_key, e)
            raise

    def get_model_download_links(self, model_info: Dict) -> List[str]:
        """Get download links for a model."""
        links = []
        model_type = model_info.get("type", "").lower()
        is_vip = "VIP" in model_info.get("name", "")
        repo_url = self.repo_urls["vip"] if is_vip else self.repo_urls["public"]

        download_files = model_info.get(
            "download_files", [model_info.get("filename")]
        )

        for file in download_files:
            if file.startswith("http"):
                links.append(file)
            else:
                if file.endswith(".yaml"):
                    if model_type == "mdxc":
                        yaml_url = f"{repo_url}/mdx_model_data/mdx_c_configs/{file}"
                        links.append(yaml_url)
                        backup_url = (
                            f"{self.repo_urls['audio_separator']}/{file}"
                        )
                        links.append(f"Backup: {backup_url}")
                    else:
                        links.append(f"{repo_url}/{file}")
                else:
                    links.append(f"{repo_url}/{file}")

        return links

    def get_model_config_details(
        self, model_info: Dict, model_data: Dict = None
    ) -> Dict:
        """Extract configuration details from model data."""
        config_details = {}
        if model_data:
            common_params = [
                "sr", "n_fft", "hop_length", "n_bins", "chunk_size",
                "seed", "dim_t", "dim_c", "self_attention", "depth",
                "window_size", "batch_size", "segment_size", "overlap",
            ]
            for param in common_params:
                if param in model_data:
                    config_details[param] = model_data[param]

            if "architecture" in model_data:
                config_details["architecture"] = model_data["architecture"]
            if "inference" in model_data:
                config_details["inference"] = model_data["inference"]
            if "training" in model_data:
                config_details["training_stems"] = model_data["training"].get(
                    "instruments", []
                )
                if "target_instrument" in model_data["training"]:
                    config_details["target_stem"] = model_data["training"][
                        "target_instrument"
                    ]
        return config_details

    def get_model_hash_info(self, filename: str) -> Dict:
        """Get model hash and size information."""
        hash_info = {}
        local_path = os.path.join(self.cache_dir, "models", filename)
        if os.path.exists(local_path):
            try:
                size_bytes = os.path.getsize(local_path)
                hash_info["size_mb"] = round(size_bytes / (1024 * 1024), 2)

                md5_hash = hashlib.md5()
                with open(local_path, "rb") as f:
                    chunk = f.read(10 * 1024 * 1024)
                    md5_hash.update(chunk)
                hash_info["md5"] = md5_hash.hexdigest()[:8] + "..."
            except OSError:
                pass
        return hash_info

    def get_all_model_data(self) -> Dict:
        """Get all model data including configurations and links."""
        logger.info("Fetching model data from various sources…")

        try:
            download_checks = self.fetch_json_data(
                "download_checks", "download_checks.json"
            )
            model_scores = self.fetch_json_data(
                "model_scores", "model_scores.json"
            )
            vr_model_data = self.fetch_json_data(
                "vr_model_data", "vr_model_data.json"
            )
            mdx_model_data = self.fetch_json_data(
                "mdx_model_data", "mdx_model_data.json"
            )

            try:
                import importlib.resources
                with importlib.resources.open_text(
                    "audio_separator", "models.json"
                ) as f:
                    audio_separator_models = json.load(f)
            except (ImportError, FileNotFoundError, OSError):
                audio_separator_models = {
                    "vr_download_list": {},
                    "mdx_download_list": {},
                    "mdx23c_download_list": {},
                    "roformer_download_list": {},
                }

            try:
                with importlib.resources.open_text(
                    "audio_separator", "model-data.json"
                ) as f:
                    audio_separator_model_data = json.load(f)
            except (ImportError, FileNotFoundError, OSError):
                audio_separator_model_data = {
                    "vr_model_data": {},
                    "mdx_model_data": {},
                }

            vr_model_data = {
                **vr_model_data,
                **audio_separator_model_data.get("vr_model_data", {}),
            }
            mdx_model_data = {
                **mdx_model_data,
                **audio_separator_model_data.get("mdx_model_data", {}),
            }

        except Exception as e:
            logger.error("Error fetching model data: %s", e)
            return {}

        all_models = {}

        # Process MDX models
        mdx_models = {
            **download_checks["mdx_download_list"],
            **download_checks["mdx_download_vip_list"],
            **audio_separator_models["mdx_download_list"],
        }
        for name, filename in mdx_models.items():
            model_key = filename
            all_models[model_key] = {
                "name": name,
                "filename": filename,
                "type": "MDX",
                "is_vip": "VIP" in name,
                "scores": model_scores.get(filename, {}).get("median_scores", {}),
                "stems": model_scores.get(filename, {}).get("stems", []),
                "target_stem": model_scores.get(filename, {}).get("target_stem"),
                "download_files": [filename],
            }

        # Process Demucs v4 models
        filtered_demucs = {
            k: v
            for k, v in download_checks["demucs_download_list"].items()
            if k.startswith("Demucs v4")
        }
        for name, files in filtered_demucs.items():
            yaml_file = next((f for f in files.keys() if f.endswith(".yaml")), None)
            if yaml_file:
                model_key = yaml_file
                all_models[model_key] = {
                    "name": name,
                    "filename": yaml_file,
                    "type": "Demucs",
                    "is_vip": False,
                    "scores": model_scores.get(yaml_file, {}).get(
                        "median_scores", {}
                    ),
                    "stems": model_scores.get(yaml_file, {}).get("stems", []),
                    "target_stem": model_scores.get(yaml_file, {}).get(
                        "target_stem"
                    ),
                    "download_files": list(files.values()),
                }

        # Process MDXC models
        mdxc_sources = {
            **download_checks["mdx23c_download_list"],
            **download_checks["mdx23c_download_vip_list"],
            **download_checks["roformer_download_list"],
            **audio_separator_models["mdx23c_download_list"],
            **audio_separator_models["roformer_download_list"],
        }
        for name, files in mdxc_sources.items():
            model_file = next(iter(files.keys()))
            model_key = model_file
            all_models[model_key] = {
                "name": name,
                "filename": model_file,
                "type": "MDXC",
                "is_vip": "VIP" in name,
                "scores": model_scores.get(model_file, {}).get(
                    "median_scores", {}
                ),
                "stems": model_scores.get(model_file, {}).get("stems", []),
                "target_stem": model_scores.get(model_file, {}).get(
                    "target_stem"
                ),
                "download_files": list(files.keys()) + list(files.values()),
            }

        self.add_model_data_from_hashes(all_models, vr_model_data, mdx_model_data)
        return all_models

    def add_model_data_from_hashes(
        self, all_models: Dict, vr_model_data: Dict, mdx_model_data: Dict
    ):
        """Add model configuration data using hash lookup."""
        hash_to_data = {**vr_model_data, **mdx_model_data}

        for model_key, model_info in list(all_models.items()):
            filename = model_info["filename"]
            try:
                model_path = self.download_model_file(filename)
                if model_path and os.path.exists(model_path):
                    model_hash = self.calculate_model_hash(model_path)
                    if model_hash and model_hash in hash_to_data:
                        model_info["model_data"] = hash_to_data[model_hash]
                        model_info["hash"] = model_hash
                        hash_info = self.get_model_hash_info(filename)
                        if hash_info:
                            model_info["hash_info"] = hash_info
            except Exception:
                pass

    def download_model_file(self, filename: str) -> Optional[str]:
        """Download a model file if not already cached."""
        models_dir = os.path.join(self.cache_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            return model_path

        try:
            url = f"{self.repo_urls['public']}/{filename}"
            response = requests.get(url, stream=True, timeout=60)

            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return model_path
        except requests.RequestException:
            pass

        return None

    def calculate_model_hash(
        self, model_path: str, bytes_to_hash: int = 10000 * 1024
    ) -> Optional[str]:
        """Calculate hash of model file (same as UVR method)."""
        try:
            file_size = os.path.getsize(model_path)
            with open(model_path, "rb") as f:
                if file_size < bytes_to_hash:
                    return hashlib.md5(f.read()).hexdigest()
                else:
                    f.seek(file_size - bytes_to_hash)
                    return hashlib.md5(f.read()).hexdigest()
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Leaderboard helpers
# ---------------------------------------------------------------------------

def list_supported_models(
    filter_sort_by: Optional[str] = None,
    show_details: bool = False,
    show_links: bool = False,
    show_config: bool = False,
    limit: int = 20,
) -> Dict:
    """List supported models with detailed information."""
    manager = ModelManager()
    all_models = manager.get_all_model_data()

    if not all_models:
        logger.error("Failed to load model data.")
        return {}

    simplified_list = {}

    for model_key, model_info in all_models.items():
        filename = model_info["filename"]
        name = model_info["name"]
        model_type = model_info["type"]
        is_vip = model_info.get("is_vip", False)
        scores = model_info.get("scores") or {}
        stems = model_info.get("stems") or []
        target_stem = model_info.get("target_stem")
        model_data = model_info.get("model_data", {})

        stems_with_scores = []
        stem_sdr_dict = {}

        for stem in stems:
            stem_scores = scores.get(stem, {})
            stem_display = f"{stem}*" if stem == target_stem else stem

            if isinstance(stem_scores, dict) and "SDR" in stem_scores:
                sdr = round(stem_scores["SDR"], 1)
                stems_with_scores.append(f"{stem_display} ({sdr})")
                stem_sdr_dict[stem.lower()] = sdr
            else:
                stems_with_scores.append(stem_display)
                stem_sdr_dict[stem.lower()] = None

        if not stems_with_scores:
            stems_with_scores = ["Unknown"]
            stem_sdr_dict["unknown"] = None

        download_links = (
            manager.get_model_download_links(model_info) if show_links else []
        )
        config_details = (
            manager.get_model_config_details(model_info, model_data)
            if show_config
            else {}
        )
        hash_info = model_info.get("hash_info", {})

        simplified_list[filename] = {
            "Name": name,
            "Type": model_type,
            "VIP": is_vip,
            "Filename": filename,
            "Stems": stems_with_scores,
            "SDR": stem_sdr_dict,
            "ModelData": model_data if show_details else {},
            "Config": config_details if show_config else {},
            "Links": download_links if show_links else [],
            "HashInfo": hash_info,
        }

    if filter_sort_by:
        if filter_sort_by == "name":
            simplified_list = dict(
                sorted(simplified_list.items(), key=lambda x: x[1]["Name"])
            )
        elif filter_sort_by == "filename":
            simplified_list = dict(sorted(simplified_list.items()))
        elif filter_sort_by in [
            "vocals",
            "instrumental",
            "drums",
            "bass",
            "other",
        ]:
            sort_by_lower = filter_sort_by.lower()
            filtered = {
                k: v
                for k, v in simplified_list.items()
                if sort_by_lower in v["SDR"]
                and v["SDR"][sort_by_lower] is not None
            }
            sorted_items = sorted(
                filtered.items(),
                key=lambda x: x[1]["SDR"][sort_by_lower],
                reverse=True,
            )
            simplified_list = dict(sorted_items)

    if limit:
        limited_items = list(simplified_list.items())[:limit]
        simplified_list = dict(limited_items)

    return simplified_list


def display_model_leaderboard(
    models_dict: Dict,
    sort_by: Optional[str] = None,
    show_details: bool = False,
    show_links: bool = False,
    show_config: bool = False,
    limit: int = 20,
):
    """Display models in a formatted leaderboard."""
    if not models_dict:
        print("No models found.")
        return

    terminal_width = 100
    print("\n" + "=" * terminal_width)
    print("AUDIO SEPARATION MODEL LEADERBOARD".center(terminal_width))
    print("=" * terminal_width)

    filter_info = []
    if sort_by:
        filter_info.append(f"Sort: {sort_by}")
    if limit:
        filter_info.append(f"Limit: {limit}")
    if show_details:
        filter_info.append("Details: ON")
    if show_links:
        filter_info.append("Links: ON")
    if show_config:
        filter_info.append("Config: ON")

    if filter_info:
        print(f"Filters: {' | '.join(filter_info)}")
        print("-" * terminal_width)

    for i, (filename, model_info) in enumerate(models_dict.items(), 1):
        name = model_info["Name"]
        model_type = model_info["Type"]
        is_vip = model_info["VIP"]
        stems = ", ".join(model_info["Stems"])

        vip_marker = " [VIP]" if is_vip else ""
        header = f"{i}. {name}{vip_marker}"
        print(f"\n{header}")
        print("-" * min(len(header), terminal_width))

        print(f"   Type: {model_type} | File: {filename}")

        if model_info["Stems"]:
            print(f"   Stems: {stems}")

        if model_info.get("HashInfo"):
            hash_info = model_info["HashInfo"]
            size_info = (
                f"Size: {hash_info.get('size_mb', 'N/A')}MB"
                if hash_info.get("size_mb")
                else ""
            )
            hash_part = (
                f"Hash: {hash_info.get('md5', 'N/A')}"
                if hash_info.get("md5")
                else ""
            )
            if size_info or hash_part:
                print(f"   {size_info} {hash_part}".strip())

        if show_links and model_info["Links"]:
            print("   Download Links:")
            for j, link in enumerate(model_info["Links"][:3], 1):
                link_display = (
                    link[:67] + "..." if len(link) > 70 else link
                )
                print(f"     {j}. {link_display}")
            if len(model_info["Links"]) > 3:
                print(
                    f"     ... and {len(model_info['Links']) - 3} more"
                )

        if show_config and model_info["Config"]:
            print("   Configuration:")
            config = model_info["Config"]
            basic_params = {}
            arch_params = {}
            for key, value in config.items():
                if key in [
                    "sr", "n_fft", "hop_length", "segment_size",
                    "batch_size", "overlap",
                ]:
                    basic_params[key] = value
                elif key in [
                    "architecture", "window_size", "chunk_size",
                    "dim_t", "dim_c", "depth",
                ]:
                    arch_params[key] = value

            if basic_params:
                basic_str = ", ".join(
                    [f"{k}: {v}" for k, v in basic_params.items()]
                )
                print(f"     Basic: {basic_str}")
            if arch_params:
                arch_str = ", ".join(
                    [f"{k}: {v}" for k, v in arch_params.items()]
                )
                print(f"     Architecture: {arch_str}")
            if "training_stems" in config:
                print(
                    f"     Training Stems: {', '.join(config['training_stems'])}"
                )
            if "target_stem" in config:
                print(f"     Target Stem: {config['target_stem']}")

        if show_details and model_info.get("ModelData"):
            print("   Raw Model Data (first 5 keys):")
            model_data = model_info["ModelData"]
            for j, (key, value) in enumerate(list(model_data.items())[:5]):
                value_str = str(value)
                if len(value_str) > 40:
                    value_str = value_str[:37] + "..."
                print(f"     {key}: {value_str}")
            if len(model_data) > 5:
                print(f"     ... and {len(model_data) - 5} more parameters")

    print("\n" + "=" * terminal_width)
    print("LEGEND:")
    print("  * = Primary target stem")
    print("  [VIP] = VIP model (support Anjok07 on Patreon)")
    print("  SDR = Signal-to-Distortion Ratio (higher is better)")
    print(f"Total models in list: {len(models_dict)}")
    print("=" * terminal_width + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def proc_file(args):
    """Main CLI handler — either list models or process audio."""
    parser = argparse.ArgumentParser(
        description="Audio Separation Tool with Model Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model listing
  uvr-cli --list_models

  # List top 10 vocal models
  uvr-cli --list_models --sort_by vocals --limit 10

  # Show models with download links
  uvr-cli --list_models --show_links

  # Process audio file
  uvr-cli --input_file song.mp3 --model_type mdx23c --config_path config.yaml --start_check_point model.ckpt
        """,
    )

    # Audio processing arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="mdx23c",
        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg",
    )
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument(
        "--start_check_point",
        type=str,
        default="",
        help="Initial checkpoint to valid weights",
    )
    parser.add_argument(
        "--input_file", type=str, help="path to audio file to process"
    )
    parser.add_argument(
        "--store_dir",
        default="",
        type=str,
        help="path to store results as wav file",
    )
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=0,
        help="list of gpu ids",
    )
    parser.add_argument(
        "--extract_instrumental",
        action="store_true",
        help="invert vocals to get instrumental if provided",
    )
    parser.add_argument(
        "--disable_detailed_pbar",
        action="store_true",
        help="disable detailed progress bar",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force the use of CPU even if CUDA is available",
    )
    parser.add_argument(
        "--flac_file",
        action="store_true",
        help="Output flac file instead of wav",
    )
    parser.add_argument(
        "--pcm_type",
        type=str,
        choices=["PCM_16", "PCM_24"],
        default="PCM_24",
        help="PCM type for FLAC files (PCM_16 or PCM_24)",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.",
    )

    # Model listing arguments
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all available models with their SDR scores",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=[
            "name",
            "filename",
            "vocals",
            "instrumental",
            "drums",
            "bass",
            "other",
        ],
        help="Sort models by name, filename, or stem SDR score",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit number of models shown in leaderboard",
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Show detailed model data including raw parameters",
    )
    parser.add_argument(
        "--show_links",
        action="store_true",
        help="Show download links for each model",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Show model configuration details",
    )
    parser.add_argument(
        "--export_json",
        type=str,
        help="Export model list to JSON file",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Handle list_models argument
    if args.list_models:
        print("Loading model database...")
        models = list_supported_models(
            filter_sort_by=args.sort_by,
            show_details=args.show_details,
            show_links=args.show_links,
            show_config=args.show_config,
            limit=args.limit,
        )

        if args.export_json:
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(models, f, indent=2, ensure_ascii=False)
            print(f"Model list exported to {args.export_json}")

        display_model_leaderboard(
            models,
            sort_by=args.sort_by,
            show_details=args.show_details,
            show_links=args.show_links,
            show_config=args.show_config,
            limit=args.limit,
        )
        return

    # ── Audio processing ─────────────────────────────────────────────────
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print("CUDA is available, use --force_cpu to disable it.")
        if isinstance(args.device_ids, list):
            device = f"cuda:{args.device_ids[0]}"
        else:
            device = f"cuda:{args.device_ids}"
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != "":
        print("Start from checkpoint: {}".format(args.start_check_point))
        if args.model_type == "htdemucs":
            state_dict = torch.load(
                args.start_check_point, map_location=device, weights_only=False
            )
            if "state" in state_dict:
                state_dict = state_dict["state"]
        else:
            state_dict = torch.load(
                args.start_check_point, map_location=device, weights_only=True
            )
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    # Multi-GPU DataParallel
    if (
        isinstance(args.device_ids, list)
        and len(args.device_ids) > 1
        and not args.force_cpu
    ):
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_file(model, args, config, device, verbose=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    proc_file(None)
