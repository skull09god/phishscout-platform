import sys
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlparse

# 1. Mock heavy training-only dependencies so Python doesn't try to load them
sys.modules['datasets'] = MagicMock()

# 2. Add the vendored v2 source to the path so imports resolve.
_V2_SRC = Path(__file__).parent / "vendor" / "PhishScout" / "v2" / "src"
sys.path.insert(0, str(_V2_SRC))

from phishscout_v2.onnx_model import PhishScoutV2  # noqa: E402

_model = None

# 3. Define a list of highly trusted domains that should NEVER be blocked by a URL model.
# (Feel free to add more as you find them)
TRUSTED_DOMAINS = {
    "google.com", "github.com", "microsoft.com", "apple.com", 
    "amazon.com", "wikipedia.org", "reddit.com", "stackoverflow.com", 
    "mozilla.org", "python.org", "w3.org", "ibm.com", "oracle.com",
    "netflix.com", "canva.com", "flipkart.com", "bbc.com", 
    "linkedin.com", "youtube.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "yahoo.com", "bing.com"
}

def _get_model():
    global _model
    if _model is None:
        _model = PhishScoutV2()
    return _model

def is_trusted_domain(url: str) -> bool:
    """Helper function to check if a URL belongs to a trusted root domain."""
    try:
        domain = urlparse(url).netloc.lower()
        # Remove port number if present (e.g., 'example.com:8080')
        domain = domain.split(':')[0]
        
        # Check exact match or subdomain match (e.g., 'en.wikipedia.org')
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                return True
        return False
    except Exception:
        return False

def detect_url(url: str) -> dict:
    """Classify a URL as phishing or legitimate using the ONNX model."""
    model = _get_model()
    result = model.predict_url(url)
    return {
        "is_phishing": bool(result["is_phishing"]),
        "probability": float(result["probability"]),
        "label": int(result["label"]),
    }

def categorize_url(url: str) -> dict:
    """
    Classifies a URL into a risk tier: 'safe', 'suspicious', or 'malicious'.
    Includes a whitelist bypass for trusted domains.
    """
    # Step 1: Check Whitelist First (Saves CPU and prevents false positives)
    if is_trusted_domain(url):
        return {
            "verdict": "safe",
            "probability": 0.0,
            "raw_result": {"note": "Bypassed by trusted domain whitelist"}
        }

    # Step 2: Run the Model
    result = detect_url(url)
    prob = result["probability"]

    # Step 3: Categorize based on thresholds
    MALICIOUS_THRESHOLD = 0.80  # High confidence
    SUSPICIOUS_THRESHOLD = 0.40 # Medium confidence

    if prob >= MALICIOUS_THRESHOLD:
        verdict = "malicious"
    elif prob >= SUSPICIOUS_THRESHOLD:
        verdict = "suspicious"
    else:
        verdict = "safe"

    return {
        "verdict": verdict,
        "probability": prob,
        "raw_result": result
    }