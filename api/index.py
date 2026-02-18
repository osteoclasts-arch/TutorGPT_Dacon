from flask import Flask, jsonify, request
import sys
import os

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

@app.route('/')
def home():
    return "TutorGPT Dacon AI Engine is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lazy import for inference service to avoid loading heavy ML libs on startup
        # This prevents Vercel cold start timeouts or memory issues if libs are present
        # but large. Note: On Vercel, heavy ML libs (xgboost, catboost) might not be installed.
        try:
            from src.inference import InferenceService
        except ImportError as e:
             return jsonify({
                "status": "error", 
                "message": f"ML libraries not available in this environment: {str(e)}"
            }), 503

        # data = request.json
        # result = InferenceService.predict(data)
        return jsonify({"status": "success", "message": "Prediction endpoint ready for integration"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Vercel expects the app instance to be exposed
# If running locally
if __name__ == '__main__':
    app.run(debug=True)
