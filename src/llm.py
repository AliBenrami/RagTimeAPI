import os
from dotenv import load_dotenv
import requests
import json


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

def call_gemini_flash_2_0(prompt:str, history:list, system_instruction:str):
    
    
    #   "system_instruction": {
    #   "parts": [
    #     {
    #       "text": "You are a cat. Your name is Neko."
    #     }
    #   ]
    # }
    
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key,
    }
    
    # Format the request data according to Gemini API specification
    data = {
        "contents": history + [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        }
    }
    
    # Add system instruction if provided
    if system_instruction:
        data["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code != 200:
        error_msg = f"API request failed with status {response.status_code}"
        try:
            error_detail = response.json()
            if "error" in error_detail:
                error_msg += f": {error_detail['error'].get('message', 'Unknown error')}"
        except:
            error_msg += f": {response.text}"
        raise Exception(error_msg)
    
    response_json = response.json()
    
    # Check if response has error
    if "error" in response_json:
        error_msg = response_json['error'].get('message', 'Unknown API error')
        raise Exception(f"API returned error: {error_msg}")
    
    # Check if candidates exist
    if "candidates" not in response_json:
        available_keys = list(response_json.keys())
        raise Exception(f"Unexpected response structure. Available keys: {available_keys}")
    
    if not response_json["candidates"]:
        raise Exception("No candidates returned from API")
    
    # Check if the candidate has the expected structure
    candidate = response_json["candidates"][0]
    if "content" not in candidate or "parts" not in candidate["content"]:
        raise Exception("Unexpected candidate structure")
    
    if not candidate["content"]["parts"]:
        raise Exception("No content parts in candidate")
    
    return candidate["content"]["parts"][0]["text"]
    

def call_llm(prompt:str, history:list, model:str, system_instruction:str):
    if model == "gemini-2.0-flash":
        return call_gemini_flash_2_0(prompt, history, system_instruction)
    else:
        raise ValueError(f"Model {model} not supported")