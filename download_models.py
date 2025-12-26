"""
Download all anime diffusion models to /content/models/
Run this ONCE in Colab Cell 2 after installing dependencies.

This downloads ~22GB of models (or select subset to save space).
Models persist in /content/models/ for the entire Colab session.
"""

import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
AVAILABLE_MODELS = {
    "balanced": "xyn-ai/anything-v4.0",           # ~4.2GB - General anime, good quality
    "quality": "stablediffusionapi/counterfeit-v30"  # ~4.8GB - High quality, detailed
}

MODELS_PATH = "/content/models"


def download_all_models():
    """Download all anime diffusion models to /content/models/."""
    logger.info("\n" + "=" * 70)
    logger.info("📥 DOWNLOADING ALL ANIME DIFFUSION MODELS")
    logger.info("=" * 70)
    logger.info(f"Target directory: {MODELS_PATH}")
    logger.info(f"Total models: {len(AVAILABLE_MODELS)}")
    logger.info(f"Estimated total size: ~9GB")
    logger.info("=" * 70 + "\n")
    
    # Create models directory
    os.makedirs(MODELS_PATH, exist_ok=True)
    logger.info(f"✅ Created directory: {MODELS_PATH}\n")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 GPU: {gpu_name}\n")
    else:
        logger.warning("⚠️  No GPU detected - downloads will be slow!\n")
    
    # Download each model
    downloaded = 0
    skipped = 0
    failed = 0
    
    for idx, (model_name, model_id) in enumerate(AVAILABLE_MODELS.items(), 1):
        model_path = f"{MODELS_PATH}/{model_name}"
        
        logger.info("─" * 70)
        logger.info(f"[{idx}/{len(AVAILABLE_MODELS)}] {model_name.upper()}")
        logger.info(f"HuggingFace ID: {model_id}")
        logger.info(f"Save path: {model_path}")
        
        # Check if already downloaded
        if os.path.exists(model_path) and os.path.isdir(model_path):
            config_file = os.path.join(model_path, "model_index.json")
            if os.path.exists(config_file):
                logger.info(f"✅ Already downloaded - skipping")
                skipped += 1
                continue
        
        try:
            logger.info(f"📥 Downloading from HuggingFace...")
            logger.info(f"   (This may take 2-5 minutes per model)")
            
            # Download model
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                resume_download=True  # Resume interrupted downloads
            )
            
            # Save to local storage
            logger.info(f"💾 Saving to {model_path}...")
            pipe.save_pretrained(model_path)
            
            logger.info(f"✅ {model_name} downloaded successfully!")
            downloaded += 1
            
            # Free memory
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            failed += 1
            continue
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Downloaded: {downloaded} models")
    logger.info(f"⏭️  Skipped (already exists): {skipped} models")
    logger.info(f"❌ Failed: {failed} models")
    logger.info(f"📁 Models location: {MODELS_PATH}")
    logger.info("=" * 70)
    
    if downloaded + skipped == len(AVAILABLE_MODELS):
        logger.info("\n✅ All models ready! You can now run diffusion_server.py")
    else:
        logger.warning(f"\n⚠️  Some models failed to download. Check errors above.")
    
    logger.info("=" * 70 + "\n")
    
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "total": len(AVAILABLE_MODELS)
    }


def download_selected_models(model_names):
    """
    Download only selected models (to save space).
    
    Args:
        model_names: List of model names to download (e.g., ["balanced", "quality"])
    
    Example:
        download_selected_models(["balanced", "quality"])  # Only 2 models (~9GB)
    """
    logger.info("\n" + "=" * 70)
    logger.info("📥 DOWNLOADING SELECTED MODELS")
    logger.info("=" * 70)
    logger.info(f"Selected: {', '.join(model_names)}")
    logger.info("=" * 70 + "\n")
    
    # Create models directory
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            logger.warning(f"⚠️  Unknown model: {model_name}, skipping")
            continue
        
        model_id = AVAILABLE_MODELS[model_name]
        model_path = f"{MODELS_PATH}/{model_name}"
        
        logger.info(f"📥 Downloading {model_name} ({model_id})...")
        
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                resume_download=True
            )
            
            pipe.save_pretrained(model_path)
            logger.info(f"✅ {model_name} saved to {model_path}\n")
            
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"❌ Failed: {e}\n")
    
    logger.info("=" * 70)
    logger.info("✅ Selected models downloaded!")
    logger.info("=" * 70 + "\n")


def list_downloaded_models():
    """List all models currently downloaded in /content/models/."""
    if not os.path.exists(MODELS_PATH):
        logger.warning(f"⚠️  Models directory not found: {MODELS_PATH}")
        return []
    
    downloaded = []
    for model_name in AVAILABLE_MODELS.keys():
        model_path = f"{MODELS_PATH}/{model_name}"
        if os.path.exists(model_path) and os.path.isdir(model_path):
            config_file = os.path.join(model_path, "model_index.json")
            if os.path.exists(config_file):
                # Get size
                size_mb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(model_path)
                    for filename in filenames
                ) / (1024 * 1024)
                
                downloaded.append({
                    "name": model_name,
                    "path": model_path,
                    "size_mb": round(size_mb, 1)
                })
    
    logger.info("\n" + "=" * 70)
    logger.info("📁 DOWNLOADED MODELS")
    logger.info("=" * 70)
    
    if downloaded:
        for model in downloaded:
            logger.info(f"✅ {model['name']}: {model['size_mb']} MB")
        logger.info("=" * 70)
        logger.info(f"Total: {len(downloaded)} models, {sum(m['size_mb'] for m in downloaded):.1f} MB")
    else:
        logger.info("❌ No models downloaded yet")
        logger.info("   Run: download_models.download_all_models()")
    
    logger.info("=" * 70 + "\n")
    return downloaded


if __name__ == "__main__":
    # When run directly, download all models
    download_all_models()
