import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"input": "i love you"},
)
print(f"Status: {response.status_code}\nResponse:\n {response.json()}")
