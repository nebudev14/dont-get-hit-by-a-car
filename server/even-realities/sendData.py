import requests

url = 'https://latticelike-kyong-befittingly.ngrok-free.dev/receive'
data = {
    "message": "test msg",
    "value": 42
}
try: 
    response = requests.get(url, json=data)
    print("Response:", response.json())
except Exception as e:
    print(f"Error making request: {e}")

# payload = {'key1': 'value1', 'key2': 'value2'}
# response = requests.post('https://httpbin.org/post', data=payload)