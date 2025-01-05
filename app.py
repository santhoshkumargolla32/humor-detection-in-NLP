from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Add this import
import pickle
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = joblib.load('vect.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data
        data = request.get_json()
        text = data['text']

        # Ensure text is a string, if it's a list, take the first element
        if isinstance(text, list):
            text = text[0]  # Assuming the list contains a single string

        # Print the received text for debugging
        print(f"Received text: {text}")

        # Ensure the text is a string
        if not isinstance(text, str):
            return jsonify({'error': 'Invalid input, text must be a string'}), 400

        # Transform the text using the vectorizer
        text_vec = vectorizer.transform([text])
        print(f"Transformed text into vector: {text_vec}")

        # Predict using the model
        prediction = model.predict(text_vec)
        print(f"Prediction result: {prediction}")

        # Ensure the prediction is a boolean value (True or False)
        classification = "humor" if prediction else "not humor"  # Convert to a boolean value (True/False)
        print(classification)

        print(jsonify({'classification': classification}))

        return jsonify({'prediction': classification})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Print the error message for debugging
        return jsonify({'error': 'Prediction failed, please check the server logs.'}), 500




if __name__ == '__main__':
    app.run(debug=True)
