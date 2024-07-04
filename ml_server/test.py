import joblib
import os
import sklearn
def test_model():
    try:
        # Define paths to the model and vectorizer
        model_path = 'model_files/linear_svc_hate_speech_model.joblib'
        vectorizer_path = 'model_files/tfidf_vectorizer.joblib'

        # Check if the model and vectorizer files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        # Load the model and vectorizer using joblib
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)

        # Sample comment to test the model
        sample_comment = "I will kill you Motherfucker!"

        # Transform the sample comment using the loaded vectorizer
        transformed_comment = loaded_vectorizer.transform([sample_comment])

        # Make a prediction using the loaded model
        result = loaded_model.predict(transformed_comment)[0]
        result_str = 'Not harmful' if result == 0 else 'Harmful'

        print(f"Test comment: {sample_comment}")
        print(f"Prediction: {result_str}")

    except Exception as e:
        print(f"Error during model testing: {e}")

if __name__ == "__main__":
    test_model()
