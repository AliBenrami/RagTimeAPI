import os
from dotenv import load_dotenv
import requests


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

def call_gemini_flash_2_0(prompt, history):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key,
    }
    
    data = {
        "contents": history + [{"role": "user", "parts": [{"text": prompt}]}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    

def call_llm(prompt, history, model):
    if model == "gemini-2.0-flash":
        return call_gemini_flash_2_0(prompt, history)
    else:
        raise ValueError(f"Model {model} not supported")