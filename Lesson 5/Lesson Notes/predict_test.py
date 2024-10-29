import requests

def predict_customer_churn(customer_data):
    url = 'http://localhost:9696/predict'
    
    response = requests.post(url, json=customer_data).json()
    
    if response['churn'] == True:
        print('sending promo email to customer')
    else:
        print('not sending promo email to customer')

if __name__ == "__main__":
    # Test customer data
    customer = {
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "phoneservice": "no",
        "multiplelines": "no_phone_service",
        "internetservice": "dsl",
        "onlinesecurity": "no",
        "onlinebackup": "yes",
        "deviceprotection": "no",
        "techsupport": "no",
        "streamingtv": "no",
        "streamingmovies": "no",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "tenure": 1,
        "monthlycharges": 29.85,
        "totalcharges": 29.85
    }
    
    # Make prediction for test customer
    predict_customer_churn(customer)