import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
    "TransactionId": "TransactionId_1",
    "BatchId": "BatchId_1",
    "AccountId": "AccountId_1",
    "SubscriptionId": "SubscriptionId_1",
    "CustomerId": "CustomerId_1",
    "CurrencyCode": "UGX",
    "CountryCode": 256,
    "ProviderId": "ProviderId_1",
    "ProductId": "ProductId_1",
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_1",
    "Amount": 1000.0,
    "Value": 1000,
    "TransactionStartTime": "2018-11-15T02:18:49Z",
    "PricingStrategy": 2
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
