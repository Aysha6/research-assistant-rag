import requests

def ask_lm(prompt):
    url = "http://localhost:1234/v1/chat/completions"

    payload = {
        "model": "lmstudio-community/Phi-4-mini-reasoning",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    res = requests.post(url, json=payload)
    return res.json()["choices"][0]["message"]["content"]
