from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the models
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Financial Fraud Detection System"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Get the input data in JSON format
    input_features = np.array(data['features']).reshape(1, -1)  # Convert the input into a format the model expects

    # Predict using both models
    logistic_pred = logistic_model.predict(input_features)[0]
    rf_pred = random_forest_model.predict(input_features)[0]

    # Return the predictions
    return jsonify({
        'logistic_prediction': int(logistic_pred),
        'random_forest_prediction': int(rf_pred)
    })

if __name__ == "__main__":
    app.run(debug=True)
