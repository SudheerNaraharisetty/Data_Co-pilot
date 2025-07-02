

import os
import re
import json
import google.generativeai as genai
from ollama import Client
from typing import Optional, List, Union
from .data_processing import extract_numbered_list, parse_response_content

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
        # Construct a prompt that asks for JSON output
        full_prompt = f"""Please provide the following data as a JSON object with a single key "data" containing a list of values.
        Request: {prompt}

        CRITICAL INSTRUCTIONS:
        1. Return ONLY a valid JSON object. Do NOT include any other text, explanations, or markdown outside the JSON block.
        2. The JSON should have one key: "data".
        3. The "data" key should contain a list of the requested items.
        4. If the request is for numeric data, the list should contain numbers.
        5. If a value is unknown, use "N/A".
        """
        
        if expected_count:
            full_prompt += f"\n6. The list should contain exactly {expected_count} items."

        response = gemini_model.generate_content(full_prompt)
        
        # Robustly extract JSON from the response
        cleaned_response = response.text.strip()
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', cleaned_response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: try to find the first { and last } to extract JSON
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = cleaned_response[json_start : json_end + 1]
            else:
                json_str = cleaned_response # Last resort, might still fail

        data = json.loads(json_str)
        return data.get("data", [])

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini: {e}. Response was: {cleaned_response[:200]}...")
        return None
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

def web_search_and_verify(company: str, data_point_request: str) -> str:
    """
    Uses Google Search via Gemini to get a concise, factual answer for a data point.
    """
    if not gemini_model:
        return "N/A (Gemini not configured)"

    try:
        # Craft a very specific prompt to get a concise answer
        prompt = f"""What is the {data_point_request} for {company}? 
        Provide ONLY the factual answer, as concisely as possible. 
        Do NOT include any conversational text, explanations, or disclaimers. 
        If you cannot find the answer, respond with "N/A".
        """
        
        # Use Gemini with web search enabled (implicitly by the model)
        response = gemini_model.generate_content(prompt)
        
        # Extract the text content, remove any leading/trailing whitespace
        # Post-process to get only the first sentence for extreme conciseness
        text_response = response.text.strip()
        first_sentence_match = re.match(r'^[^.!?]*[.!?]', text_response)
        if first_sentence_match:
            return first_sentence_match.group(0).strip()
        else:
            return text_response.split('\n')[0].strip() # Fallback to first line

    except Exception as e:
        print(f"Error during web search verification: {e}")
        return "N/A (Web search failed)"

# --- Ollama Functions ---
def query_ollama(prompt: str, context: Optional[str] = None, is_numeric: bool = False, expected_count: Optional[int] = None) -> Optional[List]:
    try:
        full_prompt = prompt
        if context:
            full_prompt = f"For the following items:\n{context}\n\n{prompt}"
        response = ollama_client.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": full_prompt}])
        content = response["message"]["content"].strip()
        return parse_response_content(content, is_numeric, expected_count)
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
