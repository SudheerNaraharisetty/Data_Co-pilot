import os
import re
import json
import google.generativeai as genai
from ollama import Client
from typing import Optional, List, Union

# --- LLM Configuration ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except KeyError:
    print("CRITICAL: GEMINI_API_KEY environment variable not found.")
    gemini_model = None
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_model = None

ollama_client = Client(host='http://localhost:11434')
OLLAMA_MODEL_NAME = "deepseek-r1:8b"

# --- Gemini Functions ---
def query_gemini_for_data(prompt: str, is_numeric: bool = False, expected_count: Optional[int] = None) -> Optional[List]:
    if not gemini_model:
        print("Gemini model is not available.")
        return None
    try:
        full_prompt = f'Please provide the following data as a JSON object with a single key "data" containing a list of values.\nRequest: {prompt}'
        response = gemini_model.generate_content(full_prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        data = json.loads(cleaned_response)
        return data.get("data", [])
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

def web_search_and_verify(company: str, data_point: str) -> str:
    if not gemini_model:
        return "N/A (Gemini not configured)"
    try:
        search_query = f"What is the {data_point} of {company}?"
        response = gemini_model.generate_content(search_query)
        return response.text.strip()
    except Exception as e:
        print(f"Error during web search: {e}")
        return "N/A (Web search failed)"

# --- Ollama Functions (Legacy) ---
def extract_numbered_list(response: str, expected_count: Optional[int] = None) -> List[str]:
    response = response.replace("<think>", "").replace("</think>", "").strip()
    numbered_list = re.findall(r'(?m)^\s*(\d+)[\.\):]\s*(.+?)\s*$', response)
    if numbered_list:
        items = [item for _, item in numbered_list]
        if expected_count and len(items) != expected_count:
            items.extend(["N/A"] * (expected_count - len(items)))
        return items
    lines = response.split('\n')
    items = [re.sub(r'^\s*\d+[\.\):]\s*', '', line).strip() for line in lines if line.strip()]
    if expected_count and len(items) != expected_count:
        items.extend(["N/A"] * (expected_count - len(items)))
    return items

def query_ollama(prompt: str, context: Optional[str] = None, is_numeric: bool = False, expected_count: Optional[int] = None) -> Optional[List]:
    try:
        full_prompt = prompt
        if context:
            full_prompt = f"For the following items:\n{context}\n\n{prompt}"
        response = ollama_client.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": full_prompt}])
        content = response["message"]["content"].strip()
        return extract_numbered_list(content, expected_count)
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

# --- Helper Functions ---
def suggest_column_name(user_request: str) -> str:
    prompt = f'Convert this request: "{user_request}" into a concise, snake_case column name.'
    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip().lower().replace(' ', '_')
        except Exception as e:
            print(f"Gemini error for column name: {e}")
    # Fallback to Ollama or rule-based
    try:
        response = ollama_client.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip().lower().replace(' ', '_')
    except Exception as e:
        print(f"Ollama error for column name: {e}")
        return get_column_name(user_request)

def get_column_name(request: str) -> str:
    request = request.strip().lower()
    request = re.sub(r'[^a-z0-9_\s]', '', request)
    return request.replace(' ', '_')[:30]

def confirm_column_name(suggested: str) -> str:
    confirmation = input(f"Suggested column name is '{suggested}'. Confirm? (y/n): ").strip().lower()
    if confirmation != 'y':
        return input("Enter alternative column name: ").strip()
    return suggested

def get_suggestions(current_columns: list) -> Optional[list]:
    prompt = f"Based on these columns: {', '.join(current_columns)}, suggest 3 more relevant columns in snake_case."
    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            return extract_numbered_list(response.text)
        except Exception as e:
            print(f"Gemini error for suggestions: {e}")
    try:
        response = ollama_client.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return extract_numbered_list(response["message"]["content"])
    except Exception as e:
        print(f"Ollama error for suggestions: {e}")
    return ["No suggestions available."]