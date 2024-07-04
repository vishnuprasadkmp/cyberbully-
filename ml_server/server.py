from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import os


# Define paths to the model and vectorizer
model_path = 'model_files/linear_svc_hate_speech_model.joblib'
vectorizer_path = 'model_files/tfidf_vectorizer.joblib'
# Check if the model and vectorizer files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Load the saved model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/analyze-comment', methods=['POST'])
def analyze_comment():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Extract the comment and userId
        comment = data['comment']
        user_id = data['userId']
        
        # Preprocess the comment using the loaded vectorizer
        transformed_comment = vectorizer.transform([comment])
        
        # Get the prediction from the model
        result = model.predict(transformed_comment)[0]
        prediction = 'Not harmful' if result == 0 else 'Harmful'
        
        result = {
            "comment": comment,
            "userId": user_id,
            "prediction": prediction  # Assuming prediction is a list with one element
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)