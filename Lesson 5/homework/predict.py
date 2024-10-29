import requests

def predict_subscription(client_data):
    url = "http://localhost:1111/predict"
    
    response = requests.post(url, json=client_data).json()
    
    if response['subscription rate'] >= 0.5:
        print(f"client likely to subscribe (probability: {response['subscription rate']}) - contact them")
    else:
        print(f"client unlikely to subscribe (probability: {response['subscription rate']}) - skip them")

if __name__ == "__main__":
    # Test client data
    client = {
        "job": "student",
        "duration": 280,
        "poutcome": "failure"
    }
    
    # Make prediction for test client
    predict_subscription(client)