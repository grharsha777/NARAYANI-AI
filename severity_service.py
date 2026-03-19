"""
NARAYANI Severity Prediction Microservice v2
Downloads ML models from Hugging Face on startup,
caches in memory, and exposes a Flask API for severity scoring.
"""

import os
import sys
import time
import json
import tempfile
import requests
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from flask import Flask, request, jsonify
from threading import Thread

app = Flask(__name__)

# ==================== MODEL CONFIGURATION ====================
MODEL_URLS = {
    'ensemble3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_ensemble_v3.pkl',
    'encoder3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_encoder_v3.pkl',
    'scaler3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_scaler_v3.pkl',
    'tfidf3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_tfidf_v3.pkl',
    'rf3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_rf_v3.pkl',
    'xgb3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_xgb_v3.pkl',
    'lgbm3': 'https://huggingface.co/grharsha777/narayani-severity-models/resolve/main/narayani_lgbm_v3.pkl',
}

# In-memory model cache
models = {}
models_loaded = False
models_downloading = False
models_loaded_list = []

# ==================== KEYWORDS & DANGER COMBOS ====================
KEYWORDS = [
    'not breathing', 'no breath', 'not responding', 'collapsed', 'chest pain',
    'heart attack', 'cardiac', 'no pulse', 'unconscious', 'fainted',
    'fell down', 'eyes closed', 'bleeding', 'blood', 'cut',
    'wound', 'heavy bleeding', 'cant breathe', 'choking', 'stuck throat',
    'gasping', 'burn', 'fire', 'flame', 'hot water',
    'acid', 'explosion', 'stroke', 'face drooping', 'arm weak',
    'speech slurred', 'accident', 'crash', 'fell', 'collision',
    'hit', 'child', 'baby', 'infant', 'elderly',
    'pregnant', 'alone', 'nobody', 'no one', 'dying',
    'dead', 'not moving', 'please help', 'emergency', 'hurry',
    'serious', 'critical', 'swallowed', 'poison', 'overdose',
    'pesticide', 'snake', 'bitten', 'bite', 'drowning',
    'water', 'submerged', 'seizure', 'fitting', 'shaking',
    'convulsion', 'unresponsive', 'pale', 'sweating',
    'vomiting blood', 'fracture', 'broken bone', 'severe', 'intense',
    'extreme', 'massive', 'head injury', 'trauma', 'multiple injuries',
    'anaphylaxis', 'allergy', 'swelling throat', 'high fever', 'temperature',
    'hypertension', 'heart disease', 'angina', 'exertion',
]

DANGER_COMBOS = [
    ('not', 'breathing'), ('no', 'pulse'), ('heavy', 'bleeding'),
    ('chest', 'pain'), ('not', 'responding'), ('child', 'breathing'),
    ('baby', 'breathing'), ('cant', 'breathe'), ('wont', 'wake'),
    ('eyes', 'closed'), ('took', 'pills'), ('heart', 'attack'),
    ('not', 'moving'), ('severe', 'pain'), ('high', 'blood pressure'),
]

DOWNGRADE_KEYWORDS = [
    'minor', 'little', 'small', 'tiny', 'mild', 'scratch', 'scrape', 
    'stubbed', 'paper cut', 'surface', 'bit', 'slightly', 'nothing serious',
    'just', 'not deep'
]

SEVERITY_MAP = {
    'low': 2,
    'medium': 5,
    'high': 8,
    'critical': 10,
}


# ==================== FEATURE EXTRACTION ====================
def extract_features(transcript):
    """Extract exactly 117 features from a transcript string."""
    import string
    text = transcript.lower()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    features = []

    # Keyword features
    for kw in KEYWORDS:
        features.append(1 if kw in clean_text else 0)

    # Danger combo features
    for combo in DANGER_COMBOS:
        features.append(1 if all(w in clean_text for w in combo) else 0)

    # Downgrade context features
    for kw in DOWNGRADE_KEYWORDS:
        features.append(1 if kw in clean_text else 0)

    # Additional features
    words = clean_text.split()
    features.append(len(words))
    features.append(text.count('!'))
    features.append(text.count('?'))
    features.append(len(text))
    features.append(clean_text.count('please'))
    features.append(clean_text.count('help'))
    features.append(clean_text.count('hurry'))
    features.append(1 if any(w in clean_text for w in ['child', 'baby', 'infant']) else 0)
    features.append(1 if any(w in clean_text for w in ['alone', 'nobody', 'no one']) else 0)
    features.append(1 if 'not' in clean_text and 'breath' in clean_text else 0)
    features.append(1 if 'heart' in clean_text and 'attack' in clean_text else 0)
    features.append(1 if 'heavy' in clean_text and 'bleed' in clean_text else 0)
    features.append(1 if 'not' in clean_text and 'respond' in clean_text else 0)
    features.append(1 if 'chest' in clean_text and 'pain' in clean_text else 0)

    return features


# ==================== MODEL DOWNLOAD ====================
def download_model(name, url, tmp_dir):
    """Download a single model file with progress logging."""
    filepath = os.path.join(tmp_dir, f'{name}.pkl')

    # Skip if already downloaded in this session
    if os.path.exists(filepath):
        print(f'  ✅ {name} — already cached at {filepath}')
        return filepath

    print(f'  ⬇️  Downloading {name}...')
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    bar_len = 30
                    filled = int(bar_len * downloaded // total_size)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    size_mb = total_size / (1024 * 1024)
                    done_mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f'\r     {name}: [{bar}] {pct:.1f}% ({done_mb:.1f}/{size_mb:.1f} MB)')
                    sys.stdout.flush()

    print(f'\n  ✅ {name} — downloaded ({total_size / (1024*1024):.1f} MB)')
    return filepath


def load_all_models():
    """Download and load all 7 ML models from Hugging Face (or use local copies)."""
    global models, models_loaded, models_downloading, models_loaded_list

    models_downloading = True
    tmp_dir = os.path.join(tempfile.gettempdir(), 'narayani_models_v3')
    os.makedirs(tmp_dir, exist_ok=True)

    # Local Datasets folder (may contain pre-trained models)
    local_datasets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')

    # Map model names to local filenames
    LOCAL_FILES = {
        'ensemble3': 'narayani_ensemble_v3.pkl',
        'encoder3': 'narayani_encoder_v3.pkl',
        'scaler3': 'narayani_scaler_v3.pkl',
        'tfidf3': 'narayani_tfidf_v3.pkl',
        'rf3': 'narayani_rf_v3.pkl',
        'xgb3': 'narayani_xgb_v3.pkl',
        'lgbm3': 'narayani_lgbm_v3.pkl',
    }

    print('\n🧠 NARAYANI ML Engine v3 — Loading Models')
    print(f'   Cache directory: {tmp_dir}')
    print(f'   Local datasets:  {local_datasets}')
    start_time = time.time()

    for name, url in MODEL_URLS.items():
        try:
            # Priority 1: Check local Datasets folder
            local_path = os.path.join(local_datasets, LOCAL_FILES.get(name, ''))
            if os.path.exists(local_path):
                print(f'  📂 {name} — loading from local Datasets/')
                models[name] = joblib.load(local_path)
                models_loaded_list.append(name)
                print(f'  ✅ {name} loaded from local file')
                continue

            # Priority 2: Check tmp cache
            cached_path = os.path.join(tmp_dir, f'{name}.pkl')
            if os.path.exists(cached_path):
                print(f'  💾 {name} — loading from cache')
                models[name] = joblib.load(cached_path)
                models_loaded_list.append(name)
                print(f'  ✅ {name} loaded from cache')
                continue

            # Priority 3: Download from Hugging Face
            filepath = download_model(name, url, tmp_dir)
            models[name] = joblib.load(filepath)
            models_loaded_list.append(name)
            print(f'  🔧 {name} loaded into memory')
        except Exception as e:
            print(f'  ❌ Failed to load {name}: {e}')

    elapsed = time.time() - start_time
    models_loaded = len(models) == len(MODEL_URLS)
    models_downloading = False

    # Patch sklearn compatibility: older models may lack 'clip' attribute
    if 'scaler3' in models:
        if not hasattr(models['scaler3'], 'clip'):
            models['scaler3'].clip = False
            print('  🔧 Patched scaler3 with missing clip attribute')

    if models_loaded:
        print(f'\n✅ All 7 ML models loaded successfully in {elapsed:.1f}s')
    else:
        print(f'\n⚠️  Only {len(models)}/{len(MODEL_URLS)} models loaded in {elapsed:.1f}s')


# ==================== PREDICTION ====================
def predict_severity(transcript):
    """Run the full severity prediction pipeline."""
    if not models_loaded:
        return None

    # Step 1: TF-IDF vectorization
    tfidf_matrix = models['tfidf3'].transform([transcript])

    # Step 2: Extract keyword features -> sparse matrix
    keyword_features = extract_features(transcript)
    keyword_sparse = csr_matrix([keyword_features])

    # Step 3: Combine TF-IDF + keyword features
    combined = hstack([tfidf_matrix, keyword_sparse])

    # Step 4: Scale features
    combined_scaled = models['scaler3'].transform(combined)

    # Step 5: Individual model predictions
    rf_pred = models['rf3'].predict(combined_scaled)
    xgb_pred = models['xgb3'].predict(combined_scaled)
    lgbm_pred = models['lgbm3'].predict(combined_scaled)

    # Step 6: Ensemble prediction
    ensemble_pred = models['ensemble3'].predict(combined_scaled)

    # Step 7: Confidence score
    ensemble_proba = models['ensemble3'].predict_proba(combined_scaled)
    confidence = float(np.max(ensemble_proba)) * 100

    # Step 8: Inverse transform labels
    rf_label = models['encoder3'].inverse_transform(rf_pred)[0]
    xgb_label = models['encoder3'].inverse_transform(xgb_pred)[0]
    lgbm_label = models['encoder3'].inverse_transform(lgbm_pred)[0]
    ensemble_label = models['encoder3'].inverse_transform(ensemble_pred)[0]

    # Map to severity score
    severity_score = SEVERITY_MAP.get(ensemble_label.lower(), 5)
    all_agreed = (rf_label == xgb_label == lgbm_label)

    return {
        'severity': severity_score,
        'severity_label': ensemble_label,
        'confidence': f'{confidence:.1f}%',
        'random_forest': rf_label,
        'xgboost': xgb_label,
        'lightgbm': lgbm_label,
        'ensemble_decision': ensemble_label,
        'all_agreed': all_agreed,
        'model_version': 'v3',
    }


# ==================== API ENDPOINTS ====================
@app.route('/health', methods=['GET'])
def health():
    ml_status = {
        'models_loaded': models_loaded,
        'loaded_models': models_loaded_list,
    }
    if models_downloading:
        ml_status['downloading'] = True
        ml_status['message'] = 'ML models loading please wait'

    return jsonify({
        'status': 'ok',
        'service': 'narayani-severity-service',
        'ml_models': ml_status,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        if models_downloading:
            return jsonify({'error': 'ML models still loading, please wait'}), 503
        return jsonify({'error': 'ML models not loaded'}), 500

    data = request.get_json()
    transcript = data.get('transcript', '')

    if not transcript:
        return jsonify({'error': 'transcript is required'}), 400

    try:
        result = predict_severity(transcript)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== STARTUP ====================
if __name__ == '__main__':
    # Load models in background thread so Flask starts immediately
    model_thread = Thread(target=load_all_models, daemon=True)
    model_thread.start()

    print('\n🔬 Severity Prediction Service starting on http://localhost:5050')
    app.run(host='0.0.0.0', port=5050, debug=False)
