import requests

texts = ["i love you", "i hate you", "dumbass"]

response = requests.post("http://127.0.0.1:8000/predict", json={"input": texts})

print(f"Status: {response.status_code}\nResponse:\n {response.text}")
