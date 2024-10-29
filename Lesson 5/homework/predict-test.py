import pickle

# Load the model
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

# Create test data
test_data = {
    "job": "management",
    "duration": 400,
    "poutcome": "success"
}

X = dv.transform([test_data])

# Make prediction
prediction = model.predict(X)
probability = model.predict_proba(X)

print(f"Probability: {probability[0][1]}")