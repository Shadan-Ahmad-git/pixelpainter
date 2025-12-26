"""
Flask server for running Stable Diffusion remotely on Google Colab with GPU.
This server receives GAN outputs from the main app and applies diffusion refinement.

Usage on Google Colab (3-cell setup):

CELL 1 - Install dependencies:
!pip install flask pyngrok diffusers transformers accelerate torch pillow flask-cors

CELL 2 - Download models (run once):
import download_models
download_models.download_all_models()

CELL 3 - Run server:
!python diffusion_server.py

The server will automatically:
- Load models from /content/models/ (downloaded in Cell 2)
- Start ngrok tunnel and save URL to npoint.io API
- Send Telegram notification with server URL

Architecture:
- Main Flask app (app2.py) runs GAN locally
- This server runs Stable Diffusion on Colab GPU via ngrok tunnel
- Main app sends GAN output to this server for refinement
- Models are downloaded once and persist in /content/models/ during session
"""

from flask import Flask, request, jsonify, send_file
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import io
import base64
import logging
import os
import atexit
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
diffusion_pipes = {}  # Store multiple loaded models
current_model = None
ngrok_tunnel = None
ngrok_url = None

# Configuration
PORT = int(os.getenv("PORT", 5000))

# Multiple anime diffusion models
AVAILABLE_MODELS = {
    "balanced": "xyn-ai/anything-v4.0",
    "quality": "stablediffusionapi/counterfeit-v30"
}

# Local models directory (no Google Drive needed)
MODELS_PATH = "/content/models"

# Hardcoded ngrok auth token (replace with your token)
NGROK_AUTH_TOKEN = ""

# Telegram Bot Configuration (for sending ngrok URL)
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Enable CORS
try:
    from flask_cors import CORS
    CORS(app)
    logger.info("CORS enabled (flask-cors installed)")
except ImportError:
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        return response
    logger.info("CORS enabled (manual headers)")


def detect_gpu():
    """Detect GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        logger.warning("⚠️  No GPU detected. Diffusion will be very slow on CPU.")
        return False


def load_model(model_name):
    """Load a specific model from local storage."""
    global diffusion_pipes, current_model
    
    # Return if already loaded in memory
    if model_name in diffusion_pipes:
        logger.info(f"✅ {model_name} already loaded in memory")
        current_model = model_name
        return diffusion_pipes[model_name]
    
    model_path = f"{MODELS_PATH}/{model_name}"
    model_id = AVAILABLE_MODELS.get(model_name)
    
    if not model_id:
        logger.error(f"❌ Unknown model: {model_name}")
        return None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load from local storage (downloaded by download_models.py)
        if os.path.exists(model_path) and os.path.isdir(model_path):
            logger.info(f"📂 Loading {model_name} from {model_path}...")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True
            )
        else:
            logger.error(f"❌ {model_name} not found at {model_path}")
            logger.error(f"   Run download_models.py first to download all models")
            return None
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        diffusion_pipes[model_name] = pipe
        current_model = model_name
        
        logger.info(f"✅ {model_name} loaded successfully on {device}")
        return pipe
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        return None


def send_telegram_message(message: str):
    """Send a message to Telegram."""
    try:
        import requests
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured")
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200 and response.json().get("ok"):
            logger.info("✅ Telegram notification sent")
            return True
        else:
            logger.warning(f"Telegram failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram: {e}")
        return False


def save_ngrok_url_to_api(url: str, api_key: str = "diffusion_server"):
    """Save ngrok URL to npoint.io API. Handles null/empty JSON and updates it."""
    try:
        import requests
        api_url = "https://api.npoint.io/bc5f0114df0586ffd535"
        
        # First, try to get current data to preserve any other fields
        current_data = {}
        try:
            current_response = requests.get(api_url, timeout=5)
            if current_response.status_code == 200:
                response_text = current_response.text.strip()
                # Handle null, empty string, or invalid JSON
                if response_text and response_text != "null" and response_text != "":
                    try:
                        current_data = current_response.json()
                        # If it's null, empty, or not a dict, start fresh
                        if current_data is None or not isinstance(current_data, dict):
                            current_data = {}
                    except ValueError:
                        # JSON decode error - start fresh
                        current_data = {}
                else:
                    # Response is null or empty
                    current_data = {}
            else:
                logger.debug(f"GET request returned {current_response.status_code}, starting fresh")
                current_data = {}
        except Exception as e:
            logger.debug(f"Error fetching current data: {e}, starting fresh")
            current_data = {}
        
        # Create payload with new ngrok URL under specific key
        payload = current_data.copy()
        payload[api_key] = {
            "url": url,
            "updated_at": datetime.now().isoformat()
        }
        
        logger.info(f"💾 Saving ngrok URL to npoint.io API ({api_key}): {url}")
        logger.debug(f"Payload: {payload}")
        
        # Use POST to save/update (npoint.io uses POST for updates)
        response = requests.post(api_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            # Verify the save worked by reading it back
            verify_response = requests.get(api_url, timeout=5)
            if verify_response.status_code == 200:
                verify_data = verify_response.json()
                if verify_data and verify_data.get(api_key, {}).get("url") == url:
                    logger.info("✅ Ngrok URL saved to npoint.io API successfully (verified)")
                    return True
                else:
                    logger.warning(f"⚠️ Verification failed. API returned: {verify_data}")
            else:
                logger.warning(f"⚠️ Could not verify save: {verify_response.status_code}")
            
            logger.info("✅ Ngrok URL saved to npoint.io API")
            return True
        else:
            logger.error(f"❌ Failed to save ngrok URL to API: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return False
    except Exception as e:
        logger.error(f"Error saving ngrok URL to API: {e}", exc_info=True)
        return False


def setup_ngrok():
    """Setup ngrok tunnel."""
    global ngrok_tunnel, ngrok_url
    
    if not NGROK_AUTH_TOKEN or NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTH_TOKEN_HERE":
        logger.warning("⚠️  NGROK_AUTH_TOKEN not configured")
        logger.info("Server will run on localhost only")
        return None
    
    try:
        from pyngrok import ngrok
        
        logger.info("Setting ngrok auth token...")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
        logger.info(f"Creating ngrok tunnel for port {PORT}...")
        tunnel = ngrok.connect(PORT, bind_tls=True)
        
        public_url = tunnel.public_url if hasattr(tunnel, 'public_url') else str(tunnel)
        if public_url.startswith('http://'):
            public_url = public_url.replace('http://', 'https://')
        
        ngrok_tunnel = tunnel
        ngrok_url = public_url
        
        logger.info("=" * 60)
        logger.info("🚀 NGROK TUNNEL CREATED!")
        logger.info("=" * 60)
        logger.info(f"Public URL: {ngrok_url}")
        logger.info(f"Set this in your main app:")
        logger.info(f"   DIFFUSION_REMOTE_URL={ngrok_url}")
        logger.info("=" * 60)
        
        # Save ngrok URL to npoint.io API
        save_ngrok_url_to_api(ngrok_url, "diffusion_server")
        
        # Send to Telegram
        telegram_message = f"🎨 Diffusion Server Online!\n\nURL: {ngrok_url}\n\nGPU: {'Yes' if torch.cuda.is_available() else 'No'}\n\nURL saved to npoint.io API"
        send_telegram_message(telegram_message)
        
        atexit.register(cleanup_ngrok)
        return ngrok_url
        
    except ImportError:
        logger.error("pyngrok not installed. Install: pip install pyngrok")
        return None
    except Exception as e:
        logger.error(f"Failed to create ngrok tunnel: {e}")
        return None


def cleanup_ngrok():
    """Clean up ngrok tunnel."""
    global ngrok_tunnel
    if ngrok_tunnel:
        try:
            from pyngrok import ngrok
            if hasattr(ngrok_tunnel, 'public_url'):
                ngrok.disconnect(ngrok_tunnel.public_url)
            else:
                ngrok.disconnect(str(ngrok_tunnel))
            logger.info("Ngrok tunnel closed")
        except:
            try:
                ngrok.kill()
                logger.info("All ngrok tunnels closed")
            except Exception as e:
                logger.warning(f"Error closing ngrok: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        gpu_info["name"] = torch.cuda.get_device_name(0)
        gpu_info["memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1024**3
    
    return jsonify({
        "status": "healthy",
        "models_loaded": list(diffusion_pipes.keys()),
        "available_models": list(AVAILABLE_MODELS.keys()),
        "current_model": current_model,
        "gpu": gpu_info,
        "ngrok_url": ngrok_url
    }), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "status": "running",
        "message": "Diffusion Server for Sketch2Color",
        "endpoints": {
            "health": "/health",
            "refine": "/refine",
            "models": "/models"
        },
        "ngrok_url": ngrok_url,
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(diffusion_pipes.keys()),
        "available_models": list(AVAILABLE_MODELS.keys())
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "available_models": {
            name: {
                "id": model_id,
                "loaded": name in diffusion_pipes,
                "current": name == current_model
            }
            for name, model_id in AVAILABLE_MODELS.items()
        },
        "current_model": current_model
    }), 200


@app.route('/refine', methods=['POST'])
def refine_image():
    """
    Apply diffusion refinement to GAN output.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_string",
        "model": "balanced" (optional, defaults to current),
        "prompt": "colorful anime art, vibrant colors",
        "strength": 0.3,
        "guidance_scale": 7.5,
        "num_inference_steps": 30
    }
    
    Returns:
    {
        "success": true/false,
        "image": "base64_encoded_refined_image",
        "model_used": "balanced",
        "error": "error message if failed"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Get requested model
        requested_model = data.get("model", current_model or "balanced")
        
        # Load model if not already loaded
        if requested_model not in diffusion_pipes:
            logger.info(f"Loading requested model: {requested_model}")
            pipe = load_model(requested_model)
            if not pipe:
                return jsonify({
                    "success": False,
                    "error": f"Failed to load model: {requested_model}"
                }), 503
        else:
            pipe = diffusion_pipes[requested_model]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get parameters
        prompt = data.get("prompt", "colorful anime art, vibrant colors, detailed")
        strength = float(data.get("strength", 0.3))
        guidance_scale = float(data.get("guidance_scale", 7.5))
        num_inference_steps = int(data.get("num_inference_steps", 30))
        
        logger.info(f"Refining with {requested_model}: {init_image.size}, strength={strength}, steps={num_inference_steps}")
        
        # Apply diffusion
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        
        refined_image = result.images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        refined_image.save(buffer, format="PNG")
        refined_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"✅ Image refined successfully with {requested_model}")
        
        return jsonify({
            "success": True,
            "image": refined_base64,
            "model_used": requested_model
        }), 200
        
    except Exception as e:
        logger.error(f"Error refining image: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🎨 Starting Multi-Model Diffusion Server")
    logger.info("=" * 60)
    logger.info(f"Port: {PORT}")
    logger.info(f"Available Models: {len(AVAILABLE_MODELS)}")
    for name, model_id in AVAILABLE_MODELS.items():
        logger.info(f"  • {name}: {model_id}")
    
    # Detect GPU
    gpu_available = detect_gpu()
    
    if not gpu_available:
        logger.warning("\n⚠️  WARNING: No GPU detected!")
        logger.warning("Diffusion will be extremely slow on CPU.")
        logger.warning("This server is designed for Google Colab with GPU.\n")
    
    # Check if models directory exists
    if not os.path.exists(MODELS_PATH):
        logger.error("\n" + "=" * 60)
        logger.error("❌ MODELS NOT FOUND!")
        logger.error("=" * 60)
        logger.error(f"Models directory not found: {MODELS_PATH}")
        logger.error("\nPlease run download_models.py first:")
        logger.error("  import download_models")
        logger.error("  download_models.download_all_models()")
        logger.error("=" * 60 + "\n")
        exit(1)
    
    # Load first model (balanced) for immediate use
    logger.info("\n" + "=" * 60)
    logger.info("Loading Default Model (balanced)...")
    logger.info("=" * 60)
    
    if load_model("balanced"):
        logger.info("✅ Default model loaded successfully!")
    else:
        logger.error("❌ Failed to load default model.")
        logger.error("Make sure you ran download_models.py first!")
        exit(1)
    
    # Setup ngrok
    logger.info("\n" + "=" * 60)
    logger.info("Setting up ngrok tunnel...")
    logger.info("=" * 60)
    
    public_url = setup_ngrok()
    
    if public_url:
        logger.info(f"\n✅ Server accessible at: {public_url}")
        logger.info(f"   Health check: {public_url}/health")
        logger.info(f"   Models list: {public_url}/models")
        logger.info(f"   Refine endpoint: {public_url}/refine")
        logger.info(f"\n📝 Server URL saved to npoint.io API automatically")
    else:
        logger.info(f"\n📡 Server running on: http://localhost:{PORT}")
        logger.info(f"   Health check: http://localhost:{PORT}/health")
        logger.info(f"   Models list: http://localhost:{PORT}/models")
        logger.info(f"   Refine endpoint: http://localhost:{PORT}/refine")
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Multi-Model Server Ready!")
    logger.info(f"   {len(diffusion_pipes)} model(s) loaded")
    logger.info(f"   {len(AVAILABLE_MODELS)} model(s) available")
    logger.info("=" * 60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Shutting down server...")
        cleanup_ngrok()
