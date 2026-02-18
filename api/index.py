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
        # Placeholder for future inference service integration
        # from src.inference import InferenceService
        # data = request.json
        # result = InferenceService.predict(data)
        return jsonify({"status": "success", "message": "Prediction endpoint ready for integration"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Vercel expects the app instance to be exposed
# If running locally
if __name__ == '__main__':
    app.run(debug=True)
