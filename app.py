"""
Flask Deployment for Anime Sketch Colorization - Local Version with Remote Diffusion
Two-Stage Model: pix2pix GAN (local) + Stable Diffusion (remote via ngrok)

This version runs locally (no ngrok) and connects to remote diffusion server:
- Runs on localhost (no ngrok needed)
- Auto-fetches remote diffusion server URL from npoint.io API
- Diffusion server (diffusion_server.py) runs on Google Colab with GPU via ngrok
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers
import requests
import warnings
import logging

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
generator = None

# Configuration
PORT = int(os.getenv("PORT", 5000))

# Remote diffusion server URL (auto-fetched from npoint.io API)
DIFFUSION_REMOTE_URL = None  # Will be fetched automatically from API

# Enable CORS
try:
    from flask_cors import CORS
    CORS(app)
    logger.info("CORS enabled")
except ImportError:
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        return response

# ============================================================================
# Model Architecture
# ============================================================================

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    inputs = layers.Input(shape=[512, 512, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    return Model(inputs=inputs, outputs=x)

# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    global generator
    
    logger.info("Loading GAN Generator...")
    
    # Legacy initializer wrappers for TF 1.x to 2.x compatibility
    class LegacyTruncatedNormal(initializers.TruncatedNormal):
        def __init__(self, mean=0.0, stddev=0.05, seed=None, **kwargs):
            kwargs.pop('dtype', None)
            super().__init__(mean=mean, stddev=stddev, seed=seed)
    
    class LegacyZeros(initializers.Zeros):
        def __init__(self, **kwargs):
            kwargs.pop('dtype', None)
            super().__init__()
    
    class LegacyOnes(initializers.Ones):
        def __init__(self, **kwargs):
            kwargs.pop('dtype', None)
            super().__init__()
    
    class LegacyRandomNormal(initializers.RandomNormal):
        def __init__(self, mean=0.0, stddev=0.05, seed=None, **kwargs):
            kwargs.pop('dtype', None)
            super().__init__(mean=mean, stddev=stddev, seed=seed)
    
    # Handle TensorFlowOpLayer from TF 1.x models
    class TensorFlowOpLayer(tf.keras.layers.Layer):
        def __init__(self, node_def, name=None, **kwargs):
            super().__init__(name=name)
            self.node_def = node_def
        
        def call(self, inputs):
            return inputs
    
    custom_objects = {
        'TruncatedNormal': LegacyTruncatedNormal,
        'Zeros': LegacyZeros,
        'Ones': LegacyOnes,
        'RandomNormal': LegacyRandomNormal,
        'TensorFlowOpLayer': TensorFlowOpLayer,
        'tf': tf
    }
    
    checkpoint_path = "checkpoints/generator_weights.h5"
    pretrained_model_path = "Pretrained models/generator_model_epoch_43.h5"
    
    model_loaded = False
    
    # Try loading pretrained model
    if os.path.exists(checkpoint_path):
        try:
            generator = load_model(checkpoint_path, custom_objects=custom_objects, compile=False)
            logger.info(f"✓ Loaded model from {checkpoint_path}")
            model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load from {checkpoint_path}: {e}")
    
    if not model_loaded and os.path.exists(pretrained_model_path):
        try:
            generator = load_model(pretrained_model_path, custom_objects=custom_objects, compile=False)
            logger.info(f"✓ Loaded pretrained model from {pretrained_model_path}")
            model_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load from {pretrained_model_path}: {e}")
    
    if not model_loaded:
        logger.warning("WARNING: No pretrained model found. Creating new generator (will produce random outputs).")
        logger.warning(f"Place trained model at: {checkpoint_path} or {pretrained_model_path}")
        generator = Generator()
    
    logger.info("Models ready!")

# ============================================================================
# Image Processing
# ============================================================================

def preprocess_sketch(image_path):
    """Load and preprocess sketch for GAN input"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    img_array = np.array(img, dtype='float32')
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(img_array, axis=0)

def postprocess_gan_output(tensor):
    """Convert GAN output tensor to PIL Image"""
    img_array = (tensor[0].numpy() * 0.5 + 0.5)  # Denormalize to [0, 1]
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_diffusion_remote(gan_output, strength=0.3, model="balanced"):
    """Send image to remote diffusion server for refinement"""
    if not DIFFUSION_REMOTE_URL:
        logger.warning("No remote diffusion server configured")
        return gan_output
    
    try:
        # Convert PIL image to base64
        buffer = io.BytesIO()
        gan_output.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send to remote server
        logger.info(f"Sending to remote diffusion server: {DIFFUSION_REMOTE_URL} (model: {model})")
        response = requests.post(
            f"{DIFFUSION_REMOTE_URL}/refine",
            json={
                "image": image_base64,
                "model": model,
                "prompt": "colorful anime art, vibrant colors, detailed artwork",
                "strength": strength,
                "guidance_scale": 7.5,
                "num_inference_steps": 30
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                refined_base64 = result.get("image")
                refined_bytes = base64.b64decode(refined_base64)
                model_used = result.get("model_used", model)
                logger.info(f"✓ Diffusion refinement applied successfully (model: {model_used})")
                return Image.open(io.BytesIO(refined_bytes))
        
        logger.warning(f"Diffusion server returned error: {response.status_code}")
        return gan_output
        
    except Exception as e:
        logger.error(f"Error applying remote diffusion: {e}")
        return gan_output

# ============================================================================
# Ngrok Setup
# ============================================================================

def get_diffusion_url_from_api():
    """Get diffusion server URL from npoint.io API."""
    try:
        api_url = "https://api.npoint.io/"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, dict) and "diffusion_server" in data:
                url = data["diffusion_server"].get("url")
                if url:
                    logger.info(f"✅ Retrieved diffusion URL from API: {url}")
                    return url
        logger.warning("No diffusion server URL found in API")
        return None
    except Exception as e:
        logger.error(f"Error getting diffusion URL from API: {e}")
        return None

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": generator is not None,
        "diffusion_remote_url": DIFFUSION_REMOTE_URL
    }), 200

@app.route('/colorize', methods=['POST'])
def colorize():
    try:
        if 'sketch' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['sketch']
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        use_diffusion = request.form.get('use_diffusion', 'false').lower() == 'true'
        strength = float(request.form.get('strength', 0.3))
        diffusion_model = request.form.get('diffusion_model', 'balanced')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing {filename}...")
        
        # GAN processing
        sketch_tensor = preprocess_sketch(filepath)
        gan_output = generator(sketch_tensor, training=False)
        gan_image = postprocess_gan_output(gan_output)
        
        # Get the preprocessed sketch image (512x512) for display consistency
        preprocessed_sketch = Image.fromarray(((sketch_tensor[0] + 1) * 127.5).astype(np.uint8))
        
        # Diffusion refinement (if enabled and remote server available)
        final_output = gan_image
        diffusion_used = False
        model_used = None
        
        if use_diffusion and DIFFUSION_REMOTE_URL:
            logger.info(f"Applying remote diffusion refinement (model: {diffusion_model})...")
            final_output = apply_diffusion_remote(gan_image, strength, diffusion_model)
            diffusion_used = True
            model_used = diffusion_model
        
        # Save results (use preprocessed sketch so all images are same size)
        sketch_path = os.path.join(app.config['RESULTS_FOLDER'], f'sketch_{filename}')
        gan_path = os.path.join(app.config['RESULTS_FOLDER'], f'gan_{filename}')
        final_path = os.path.join(app.config['RESULTS_FOLDER'], f'final_{filename}')
        
        preprocessed_sketch.save(sketch_path)
        gan_image.save(gan_path)
        final_output.save(final_path)
        
        # Convert to base64 for response
        def image_to_base64(img_path):
            with open(img_path, 'rb') as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        
        return jsonify({
            "success": True,
            "sketch": image_to_base64(sketch_path),
            "gan_output": image_to_base64(gan_path),
            "final_output": image_to_base64(final_path),
            "diffusion_used": diffusion_used,
            "diffusion_model": model_used,
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download/<output_type>/<filename>', methods=['GET'])
def download(output_type, filename):
    try:
        prefix_map = {'sketch': 'sketch_', 'gan': 'gan_', 'final': 'final_'}
        prefix = prefix_map.get(output_type, 'final_')
        filepath = os.path.join(app.config['RESULTS_FOLDER'], f'{prefix}{filename}')
        
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=f'{output_type}_{filename}')
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🎨 Anime Sketch Colorization Server (Colab Version)")
    logger.info("=" * 60)
    logger.info(f"Port: {PORT}")
    
    # Load models
    load_models()
    
    # Try to auto-fetch diffusion URL from API if not set
    if not DIFFUSION_REMOTE_URL:
        logger.info("\n🔍 Checking npoint.io API for diffusion server URL...")
        DIFFUSION_REMOTE_URL = get_diffusion_url_from_api()
    
    if DIFFUSION_REMOTE_URL:
        logger.info(f"\n🎨 Remote Diffusion Server: {DIFFUSION_REMOTE_URL}")
        logger.info("   Diffusion refinement available!")
    else:
        logger.info("\n⚠️  No remote diffusion server configured (GAN only)")
        logger.info("   Start diffusion_server.py on Colab with GPU to enable diffusion")
    
    logger.info(f"\n📡 Server running on: http://localhost:{PORT}")
    logger.info(f"   Open in browser: http://localhost:{PORT}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Server is ready!")
    logger.info("=" * 60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False)
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Shutting down server...")
