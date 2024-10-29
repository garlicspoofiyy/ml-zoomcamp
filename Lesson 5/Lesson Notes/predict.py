import pickle

from flask import Flask
from flask import request
from flask import jsonify

# Path to the pre-trained model file
model_file = 'model_C=1.0.bin'

# Load the dictionary vectorizer and the model from the pickle file
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Initialize Flask application
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    # Get customer data from POST request JSON body
    customer = request.get_json()

    # Transform customer data using dictionary vectorizer
    X = dv.transform([customer])
    # Get prediction probability for customer churning
    y_pred = model.predict_proba(X)[0, 1]
    # Classify as churn if probability >= 0.5
    churn = y_pred >= 0.5

    # Prepare response with prediction results
    result = {
        'churn_probability': float(y_pred),  # Convert numpy float to Python float
        'churn': bool(churn)                 # Convert numpy bool to Python bool
    }

    # Return JSON response
    return jsonify(result)


if __name__ == "__main__":
    # Run Flask app in debug mode, accessible from any IP on port 9696
    app.run(debug=True, host='0.0.0.0', port=9696)