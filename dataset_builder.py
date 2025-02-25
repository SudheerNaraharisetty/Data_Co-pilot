import argparse
import os
import re
import json
import pandas as pd
from ollama import Client
from typing import Optional, List, Dict, Union, Tuple, Any

# Ollama setup
ollama_client = Client(host='http://localhost:11434')
MODEL_NAME = "deepseek-r1:8b"  # Use deepseek-r1:8b for up-to-date data
OUTPUT_CSV: Optional[str] = None  # User-defined filename
initial_items: Optional[List[str]] = None  # Store initial list for context

def extract_numbered_list(response: str, expected_count: Optional[int] = None) -> List[str]:
    """
    Extracts list items formatted as a numbered list from the response.
    Improved to handle various list formats and cleaner extraction.
    """
    # Clean the response by removing thinking tags and unnecessary formatting
    response = response.replace("<think>", "").replace("</think>", "").strip()
    
    # First, try to find a numbered list with the pattern "1. Item"
    numbered_list = re.findall(r'(?m)^\s*(\d+)[.):]\s*(.+?)\s*$', response)
    
    if numbered_list:
        # Return just the items without the numbers or locations
        items = []
        for _, item in numbered_list:
            # Remove locations or anything after a dash, comma, or hyphen
            clean_item = re.sub(r'\s+[-–—]\s+.*', '', item)
            clean_item = re.sub(r',\s+.*', '', clean_item)
            
            # Remove any parentheses and their contents
            clean_item = re.sub(r'\s*\([^)]*\)', '', clean_item)
            
            # Remove any remaining special characters and extra whitespace
            clean_item = clean_item.strip()
            items.append(clean_item)
            
        # Handle expected count
        if expected_count is not None:
            if len(items) > expected_count:
                items = items[:expected_count]
            elif len(items) < expected_count:
                items.extend(["N/A"] * (expected_count - len(items)))
        
        return items
    
    # If no numbered list found, try to split by lines and clean
    lines = response.split('\n')
    items = []
    for line in lines:
        if line.strip():
            # Remove numbers, locations, and other formatting
            clean_line = re.sub(r'^\s*\d+[.):]\s*', '', line)
            clean_line = re.sub(r'\s+[-–—]\s+.*', '', clean_line)
            clean_line = re.sub(r',\s+.*', '', clean_line)
            clean_line = re.sub(r'\s*\([^)]*\)', '', clean_line)
            clean_line = clean_line.strip()
            if clean_line:
                items.append(clean_line)
    
    # Handle expected count for line-based extraction
    if expected_count is not None:
        if len(items) > expected_count:
            items = items[:expected_count]
        elif len(items) < expected_count:
            items.extend(["N/A"] * (expected_count - len(items)))
    
    return items

def extract_numeric_value(text: str) -> Union[int, float, str]:
    """
    Extracts only the numeric value from a string.
    Handles different number formats and units.
    """
    # Remove any text before a colon
    text = re.sub(r'^.*?:\s*', '', text)
    
    # Try to find a numeric value with or without units
    # Look for numbers with units (trillion, billion, million)
    trillion_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:trillion|T)', text, re.IGNORECASE)
    if trillion_match:
        return float(trillion_match.group(1)) * 1_000_000_000_000
    
    billion_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:billion|B)', text, re.IGNORECASE)
    if billion_match:
        return float(billion_match.group(1)) * 1_000_000_000
    
    million_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:million|M)', text, re.IGNORECASE)
    if million_match:
        return float(million_match.group(1)) * 1_000_000
    
    # Handle 'billions' in column name but data doesn't specify units
    # If "billions" is in the column name but the value looks like a raw large number
    # Convert large numbers to billions for better readability
    if re.search(r'billion', text, re.IGNORECASE):
        number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if number_match:
            clean_num = number_match.group(1).replace(',', '')
            if float(clean_num) > 100:  # Likely to be a raw value, not already in billions
                return float(clean_num)
            
    # Look for plain numbers, including with commas
    number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if number_match:
        # Remove commas before converting
        clean_num = number_match.group(1).replace(',', '')
        if '.' in clean_num:
            return float(clean_num)
        else:
            return int(clean_num)
    
    # No numeric value found
    return "N/A"

def parse_response_content(response: str, is_numeric: bool = False, expected_count: Optional[int] = None) -> List:
    """
    Parses the Ollama response to extract clean data.
    For numeric data, extracts only the numbers.
    For text data, extracts clean item names.
    """
    if not response:
        print("Warning: Empty response from Ollama.")
        return ["N/A"] * (expected_count or 1)
    
    # Extract the items as text first
    items = extract_numbered_list(response, expected_count)
    
    # For numeric data, extract just the numbers
    if is_numeric:
        processed = []
        for item in items:
            value = extract_numeric_value(item)
            processed.append(value)
        return processed
    
    return items

def query_ollama(prompt: str, context: Optional[str] = None, is_numeric: bool = False, expected_count: Optional[int] = None) -> Optional[List]:
    """
    Queries Ollama with a prompt, using specific instructions based on the query type.
    """
    try:
        # Check if this is a request about industry segments or categories
        is_categorical = any(keyword in prompt.lower() for keyword in 
                          ["industry", "segment", "category", "type", "sector"])
        
        # Check if this is a request about people (like CEOs, founders)
        is_person = any(keyword in prompt.lower() for keyword in 
                      ["ceo", "founder", "president", "director", "head", "chief", "leader", "name"])
        
        # Check if this is a date-related query
        is_date = any(keyword in prompt.lower() for keyword in 
                    ["date", "year", "founded", "established", "started", "when", "time"])
        
        if context is not None:
            # This is a follow-up column for existing items
            context_items = context.splitlines()
            num_items = len(context_items)
            expected_count = num_items
            
            # Create a structured, clear prompt
            full_prompt = f"For the following list of {num_items} items:\n\n"
            
            # Add the context items as a clean list
            for i, item in enumerate(context_items):
                # Extract just the item name from the context
                clean_item = re.sub(r'^\s*\d+[.):]\s*', '', item).strip()
                full_prompt += f"{i+1}. {clean_item}\n"
            
            # Different prompt based on data type
            if is_numeric:
                full_prompt += f"\nProvide ONLY the {prompt} as numbers for each item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS: 
1. Return ONLY numbers, no text or item names.
2. Format as a numbered list (e.g., "1. 250").
3. For large numbers, use appropriate representation (e.g., "394" for $394 billion).
4. If "billions" is in the request, return the number in billions (e.g., "2.7" for 2.7 billion).
5. If unknown, use "N/A".

Example response format:
1. 394
2. 184
3. N/A"""
            elif is_categorical:
                full_prompt += f"\nProvide the {prompt} for each item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the category/segment, no extra details.
2. Format as a numbered list (e.g., "1. Consumer Electronics").
3. Keep each response to 1-3 words.
4. No explanations or commentary.
5. If unknown, use "N/A".

Example response format:
1. Consumer Electronics
2. Cloud Computing
3. E-Commerce"""
            elif is_person:
                full_prompt += f"\nProvide the {prompt} for each item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the person's name, no titles or extra details.
2. Format as a numbered list (e.g., "1. Tim Cook").
3. Include first and last name.
4. No explanations or commentary.
5. If unknown, use "N/A".

Example response format:
1. Tim Cook
2. Satya Nadella
3. Andy Jassy"""
            elif is_date:
                full_prompt += f"\nProvide the {prompt} for each item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the year or date, no extra details.
2. Format as a numbered list (e.g., "1. 1976").
3. For years, provide just the 4-digit year.
4. No explanations or commentary.
5. If unknown, use "N/A".

Example response format:
1. 1976
2. 1975
3. N/A"""
            else:
                full_prompt += f"\nProvide the {prompt} for each item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the exact values, no extra details.
2. Format as a numbered list (e.g., "1. Value").
3. Keep each response short and specific.
4. No explanations or commentary.
5. Do NOT repeat the names from the list above.
6. If unknown, use "N/A".

Example response format:
1. Value for item 1
2. Value for item 2
3. N/A"""
        else:
            # This is the initial request
            count_str = f"{expected_count} " if expected_count else ""
            
            # Clear instructions for the initial list
            full_prompt = f"Provide a list of {count_str}{prompt}."
            
            if is_numeric:
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY numbers, no text.
2. Format as a numbered list (e.g., "1. 250").
3. For large numbers, provide in an appropriate format.
4. No names, no explanations, just numbers.
5. If unknown, use "N/A".

Example response format:
1. 394
2. 184
3. N/A"""
            else:
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the exact names/values, no locations or extra details.
2. Format as a numbered list (e.g., "1. Apple").
3. Keep each response to 5 words or less.
4. No explanations or commentary.
5. If unknown, use "N/A".

Example response format:
1. Apple
2. Microsoft
3. N/A"""
        
        # Send the request to Ollama
        response = ollama_client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": full_prompt}])
        content = response["message"]["content"].strip()
        
        # Parse the response
        return parse_response_content(content, is_numeric, expected_count)
        
    except Exception as e:
        print(f"Error querying Ollama: {e}. Ensure Ollama is running and '{MODEL_NAME}' is available.")
        return None

def verify_column_data(column_name: str, column_data: List, is_numeric: bool) -> Tuple[bool, str]:
    """
    Internally verifies that the generated column data matches the expected format.
    Returns a tuple of (bool success, str reason) for detailed feedback.
    """
    if not column_data:
        return False, "No data received."
    
    # Check if this is a categorical or person-related column
    is_categorical = any(keyword in column_name.lower() for keyword in 
                       ["industry", "segment", "category", "sector", "type"])
    
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_date = any(keyword in column_name.lower() for keyword in 
                ["date", "year", "founded", "established", "started"])
    
    # For numeric data, check that all items are numbers or N/A
    if is_numeric:
        for item in column_data:
            if item != "N/A" and not isinstance(item, (int, float)):
                return False, f"Non-numeric value found: {item}"
        return True, "Numeric data verified."
    
    # For people's names, check name format
    if is_person:
        for item in column_data:
            if item != "N/A":
                # Check if item matches original company/item name
                if any(item == first_col_item for first_col_item in initial_items):
                    return False, f"Person column contains item names instead of people names: {item}"
                # Check for titles (Mr., Ms., etc.)
                if re.search(r'Mr\.|Ms\.|Mrs\.|Dr\.', str(item)):
                    return False, f"Person name contains titles: {item}"
        return True, "Person data verified."
    
    # For industry/categorical data
    if is_categorical:
        valid = True
        error_msg = ""
        
        for item in column_data:
            if item != "N/A":
                # Check if item matches original company/item name
                if any(item == first_col_item for first_col_item in initial_items):
                    valid = False
                    error_msg = f"Category column contains item names instead of categories: {item}"
                    break
        
        if valid:
            return True, "Category data verified."
        else:
            return False, error_msg
    
    # For date data, check format
    if is_date:
        for item in column_data:
            if item != "N/A":
                # Check if it's a 4-digit year
                if not (isinstance(item, (int, str)) and re.match(r'^\d{4}$', str(item))):
                    if not isinstance(item, (int, float)):
                        return False, f"Date column contains non-date values: {item}"
        return True, "Date data verified."
    
    # For general text data
    for item in column_data:
        if item != "N/A":
            # Check for location patterns (City, State)
            if re.search(r',\s+[A-Z][a-z]+', str(item)):
                return False, f"Item contains location format: {item}"
            
            # Check for dash followed by text (common in explanations)
            if re.search(r'\s+[-–—]\s+', str(item)):
                return False, f"Item contains dash with explanation: {item}"
            
            # Check if item is too long (likely contains extra info)
            if len(str(item).split()) > 5:
                return False, f"Item is too verbose: {item}"
    
    return True, "Text data verified."

def suggest_column_name(user_request: str) -> str:
    """
    Uses the LLM to suggest a clean, appropriate column name based on the request.
    """
    prompt = f"""Convert this request: "{user_request}" into a concise database column name following these rules:
1. Use snake_case (lowercase with underscores)
2. Maximum 30 characters
3. Include units in parentheses if present
4. Remove any phrases like "for each company in the dataset"
5. Focus on the core concept being requested

Example conversions:
- "List of 10 tech companies" → "top_10_tech_companies"
- "Revenue in billions USD for each company" → "revenue_billions_usd"
- "Founded year" → "founded_year"
- "CEOs of each company" → "ceo"

Respond with ONLY the column name, no explanation.
"""
    
    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        suggested = response["message"]["content"].strip()
        
        # Clean up the response
        suggested = re.sub(r'[`"\'*]', '', suggested).strip()
        suggested = suggested.split('\n')[0].strip()
        
        # Check if the result looks like a prompt failure
        if len(suggested) > 40 or ' ' in suggested or suggested.startswith(('1.', '-', '•', '*')):
            # Fall back to algorithmic name generation
            return get_column_name(user_request)
        
        return suggested
    except Exception as e:
        print(f"Error getting column name suggestion: {e}")
        return get_column_name(user_request)

def get_column_name(request: str) -> str:
    """
    Generates a clean, specific column name based on the request.
    """
    # Check if the request already contains a user clarification
    # If so, don't include the clarification part in the column name
    if " based on " in request:
        request = request.split(" based on ")[0]
    
    # Clean up the request
    request = request.strip().lower()
    
    # Handle "list of top X" pattern
    top_match = re.search(r'(?:list of |)(top|best)\s+(\d+)\s+(.*?)(?:\s+in\s+|\s*$)', request)
    if top_match:
        count = top_match.group(2)
        items = top_match.group(3).strip()
        # Remove filler words
        for filler in ["the", "of", "for", "in", "on", "by", "with"]:
            items = items.replace(f" {filler} ", "_")
        items = items.replace(" ", "_")
        return f"top_{count}_{items}"
    
    # Handle specific column types
    if re.search(r'ceo|chief\s+executive', request, re.IGNORECASE):
        return "ceo"
    
    if re.search(r'founder', request, re.IGNORECASE):
        return "founder"
    
    if re.search(r'found(?:ed|ing)\s+year', request, re.IGNORECASE):
        return "founded_year"
        
    if re.search(r'revenue', request, re.IGNORECASE):
        # Check if units are specified
        if re.search(r'billion', request, re.IGNORECASE):
            return "revenue_billions_usd"
        if re.search(r'million', request, re.IGNORECASE):
            return "revenue_millions_usd"
        return "revenue_usd"
    
    if re.search(r'industry|segment|sector', request, re.IGNORECASE):
        return "industry_segment"
    
    if re.search(r'employees|workforce|staff', request, re.IGNORECASE):
        return "employee_count"
    
    if re.search(r'market\s*cap', request, re.IGNORECASE):
        return "market_cap_usd"
    
    # Handle other patterns
    # Remove common prefixes
    for prefix in ["list of ", "provide ", "get ", "what is the ", "what are the "]:
        if request.startswith(prefix):
            request = request[len(prefix):]
    
    # Extract units if present
    units_match = re.search(r'\(([^)]+)\)', request)
    units = f"_{units_match.group(1).lower().replace(' ', '_')}" if units_match else ""
    
    # Remove units from main name
    if units:
        request = re.sub(r'\([^)]+\)', '', request)
    
    # Remove phrases that indicate relationship to dataset
    request = re.sub(r'(?:each|every|all)\s+(?:company|item)s?\s+in\s+(?:the\s+)?dataset', '', request)
    request = re.sub(r'for\s+(?:each|every|all)\s+(?:company|item)s?', '', request)
    request = re.sub(r'of\s+(?:each|every|all)\s+(?:company|item)s?', '', request)
    
    # Remove filler words
    for filler in [" the ", " of ", " for ", " in ", " on ", " by ", " with "]:
        request = request.replace(filler, "_")
    
    # Convert spaces to underscores and clean up
    column_name = request.replace(" ", "_")
    
    # Remove any special characters
    column_name = re.sub(r'[^a-z0-9_]', '', column_name)
    
    # Add units if present
    column_name = column_name + units
    
    # Remove consecutive underscores
    column_name = re.sub(r'_{2,}', '_', column_name)
    
    # Remove trailing underscores
    column_name = column_name.strip('_')
    
    # Limit length
    if len(column_name) > 30:
        column_name = column_name[:30]
        column_name = column_name.rstrip('_')
    
    return column_name

def confirm_column_name(suggested: str) -> str:
    """
    Prompts the user to confirm the suggested column name.
    If the user disagrees, allows them to enter an alternative.
    """
    confirmation = input(f"Suggested column name is '{suggested}'. Do you confirm? (y/n): ").strip().lower()
    if confirmation != 'y':
        new_name = input("Please provide an alternative column name: ").strip()
        
        # If the input starts with 'y' or 'n', it might be a mistyped confirmation
        if new_name.lower() in ['y', 'yes', 'n', 'no'] and len(new_name) <= 3:
            new_name = input("Please provide the actual column name (not y/n): ").strip()
        
        # Convert to snake_case if needed
        if ' ' in new_name:
            new_name = new_name.lower().replace(' ', '_')
        
        # Clean up any special characters
        new_name = re.sub(r'[^a-zA-Z0-9_()]', '_', new_name)
        
        return new_name
    return suggested

def clarify_and_confirm(user_prompt: str, is_numeric: bool) -> Tuple[str, bool]:
    """
    Uses the LLM to analyze the user's request and ask for clarification if needed.
    Returns the clarified prompt and possibly updated is_numeric flag.
    """
    # Define the clarification prompt
    clar_query = f"""Analyze this request: "{user_prompt}"

1. Is this asking for numeric data? (yes/no)
2. Is the request ambiguous or missing important details? (yes/no)
3. If ambiguous, what specific clarification is needed? (Be specific and clear)
4. What's the most likely interpretation of this request? (1 sentence)

Format response as JSON:
{{
  "is_numeric": true/false,
  "needs_clarification": true/false,
  "clarification_question": "Question to ask the user or none",
  "interpretation": "Most likely interpretation"
}}"""

    try:
        # Get the clarification response
        clar_response = ollama_client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": clar_query}])
        clar_text = clar_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', clar_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_numeric = analysis.get("is_numeric", is_numeric)
                needs_clarification = analysis.get("needs_clarification", False)
                clarification_question = analysis.get("clarification_question", None)
                interpretation = analysis.get("interpretation", user_prompt)
                
                if needs_clarification and clarification_question and "none" not in clarification_question.lower():
                    print(f"\nClarification needed: {clarification_question}")
                    user_clarification = input("Your clarification: ").strip()
                    
                    # Combine the original request with the clarification but don't add it to the column name
                    clarified_prompt = f"{user_prompt} based on {user_clarification}"
                    return clarified_prompt, is_numeric
                
                # No clarification needed, show interpretation
                print(f"Interpretation: {interpretation}")
                confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
                if confirmation != 'y':
                    new_prompt = input("Please enter your clarified request: ").strip()
                    return new_prompt, is_numeric
                
                return user_prompt, is_numeric
            except json.JSONDecodeError:
                pass
        
        # Fallback to simpler extraction if JSON parsing fails
        is_numeric_match = re.search(r'1\.\s*(yes|no)', clar_text, re.IGNORECASE)
        if is_numeric_match:
            is_numeric = is_numeric_match.group(1).lower() == "yes"
        
        needs_clarification_match = re.search(r'2\.\s*(yes|no)', clar_text, re.IGNORECASE)
        needs_clarification = needs_clarification_match and needs_clarification_match.group(1).lower() == "yes"
        
        clarification_question_match = re.search(r'3\.\s*\[([^\]]+)\]', clar_text)
        clarification_question = clarification_question_match.group(1) if clarification_question_match else None
        
        if needs_clarification and clarification_question and "none" not in clarification_question.lower():
            print(f"\nClarification needed: {clarification_question}")
            user_clarification = input("Your clarification: ").strip()
            
            # Combine the original request with the clarification
            clarified_prompt = f"{user_prompt} based on {user_clarification}"
            return clarified_prompt, is_numeric
        
        # No clarification needed or couldn't parse correctly
        print(f"Interpretation: {user_prompt}")
        confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
        if confirmation != 'y':
            new_prompt = input("Please enter your clarified request: ").strip()
            return new_prompt, is_numeric
        
        return user_prompt, is_numeric
        
    except Exception as e:
        print(f"Error clarifying request: {e}.")
        # Fall back to the original prompt
        return user_prompt, is_numeric

def get_suggestions(current_columns: list) -> Optional[list]:
    """
    Uses Ollama to suggest 3 additional dataset columns based on the current columns.
    Returns a list of clean column name suggestions.
    """
    # Determine the likely dataset topic
    topic = "this dataset"
    column_str = ", ".join(current_columns)
    
    prompt = f"""Based on these existing columns: {column_str}

Suggest 3 additional columns that would enhance this dataset.
Each suggestion should be a short, descriptive column name in snake_case (lowercase with underscores).
Include appropriate unit indicators in parentheses when relevant.

CRITICAL INSTRUCTIONS:
1. Return ONLY column names, no explanations.
2. Format as a numbered list.
3. Use snake_case with underscores.
4. Keep names short (1-3 words).
5. Names must be relevant to the dataset topic.

Example response format:
1. revenue_usd
2. employee_count 
3. market_share_percent"""
    
    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        content = response["message"]["content"].strip()
        
        # Extract the column suggestions
        suggestions = extract_numbered_list(content)
        
        # Clean up and normalize the column names
        cleaned_suggestions = []
        for suggestion in suggestions:
            # Convert to snake_case
            clean_name = suggestion.lower().replace(' ', '_')
            # Remove any remaining special characters
            clean_name = re.sub(r'[^a-z0-9_()]', '', clean_name)
            cleaned_suggestions.append(clean_name)
        
        return cleaned_suggestions
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return ["No suggestions available."]

def initialize_dataset(column_name: str, data: List[str]) -> None:
    """
    Initializes the CSV file with the first column using the user-defined filename.
    """
    global OUTPUT_CSV, initial_items
    if not OUTPUT_CSV:
        print("Error: Dataset filename not set. Please set a filename before proceeding.")
        return
    try:
        if os.path.exists(OUTPUT_CSV) and not os.access(OUTPUT_CSV, os.W_OK):
            print(f"Permission denied to write to '{OUTPUT_CSV}'.")
            return
            
        # Convert data to strings for consistent CSV handling
        string_data = [str(item) if item != "N/A" else "N/A" for item in data]
        
        df = pd.DataFrame({column_name: string_data})
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Initialized dataset with column '{column_name}' in '{OUTPUT_CSV}'.")
        initial_items = [str(item) if pd.notna(item) and str(item).strip() else "N/A" for item in data]
    except PermissionError as e:
        print(f"Permission error: {e}.")
    except Exception as e:
        print(f"Error initializing dataset: {e}")

def add_column(column_name: str, data: List[Any]) -> None:
    """
    Adds a new column to the existing CSV using the user-defined filename.
    """
    global OUTPUT_CSV, initial_items
    if not os.path.exists(OUTPUT_CSV):
        print("Error: Dataset not initialized. Start with a base column.")
        return
    try:
        if not os.access(OUTPUT_CSV, os.W_OK):
            print(f"Permission denied to write to '{OUTPUT_CSV}'.")
            return
        df = pd.read_csv(OUTPUT_CSV)
        if len(data) != len(df):
            print(f"Error: Data length ({len(data)}) does not match existing rows ({len(df)}).")
            return
        df[column_name] = data
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Added column '{column_name}' to the dataset in '{OUTPUT_CSV}'.")
    except PermissionError as e:
        print(f"Permission error: {e}.")
    except Exception as e:
        print(f"Error adding column: {e}")

def display_dataset() -> None:
    """
    Displays the current state of the dataset.
    """
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print("\nCurrent Dataset:")
        print(df.to_string(index=False))
    else:
        print("No dataset initialized yet.")

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    args = parser.parse_args()

    # Prompt for dataset filename.
    global OUTPUT_CSV
    if not OUTPUT_CSV:
        default_name = "dataset.csv"
        filename = input(f"\nEnter the filename for the dataset (press Enter for default '{default_name}'): ").strip()
        OUTPUT_CSV = filename if filename else default_name
        if not OUTPUT_CSV.endswith(".csv"):
            OUTPUT_CSV += ".csv"
        print(f"Dataset will be saved as '{OUTPUT_CSV}'.")

    while True:
        if args.request:
            user_request = args.request
        else:
            user_request = input("\nEnter your request or 'finalize' to complete: ").strip()

        if user_request.lower() == "finalize" or args.finalize:
            if os.path.exists(OUTPUT_CSV):
                print(f"\nDataset finalized and saved as '{OUTPUT_CSV}'.")
                display_dataset()
                new_name = input("Would you like to rename the dataset? Enter a new filename (or press Enter to keep current name): ").strip()
                if new_name:
                    if not new_name.endswith(".csv"):
                        new_name += ".csv"
                    os.rename(OUTPUT_CSV, os.path.join(os.path.dirname(OUTPUT_CSV), new_name))
                    OUTPUT_CSV = new_name
                    print(f"Dataset renamed to '{OUTPUT_CSV}'.")
            else:
                print("No dataset to finalize.")
            break

        # Detect if this is likely a numeric request
        is_numeric = any(keyword in user_request.lower() for keyword in 
                        ["revenue", "sales", "profit", "price", "cost", "amount", "number", 
                         "count", "total", "size", "year", "age", "employees", "market cap"])
        
        # Clarify the request and confirm with the user
        clarified_request, is_numeric = clarify_and_confirm(user_request, is_numeric)
        
        # Determine expected count (e.g., for "top 10" requests)
        expected_count = None
        count_match = re.search(r'(?:top|best)\s+(\d+)', clarified_request.lower())
        if count_match:
            expected_count = int(count_match.group(1))
        
        # Prepare context for subsequent columns
        context = None
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(df[first_col].tolist())])
                # Set initial_items if it's not already set
                global initial_items
                if not initial_items:
                    initial_items = df[first_col].tolist()
        
        # Make sure the first column is not numeric
        if not os.path.exists(OUTPUT_CSV):
            is_numeric = False
        
        # Get the data from Ollama
        response = query_ollama(clarified_request, context=context, is_numeric=is_numeric, expected_count=expected_count)
        if not response:
            print("Failed to get data. Try again or check the Ollama server.")
            continue
        
        # Generate and confirm the column name
        column_name = suggest_column_name(clarified_request)
        column_name = confirm_column_name(column_name)
        
        # Verify the data matches the expected format
        verified, reason = verify_column_data(column_name, response, is_numeric)
        if not verified:
            print(f"Internal verification: {reason}. Re-querying with improved instructions...")
            
            # Retry with more explicit instructions
            retry_request = clarified_request
            if is_numeric:
                retry_request += " (numbers only, no text)"
            else:
                retry_request += " (concise values only, no descriptions)"
                
            response = query_ollama(retry_request, context=context, is_numeric=is_numeric, expected_count=expected_count)
            
            # Check if the retry was successful
            verified, reason = verify_column_data(column_name, response, is_numeric)
            if not verified:
                print(f"Internal verification: {reason}. Proceeding with unverified data.")
        
        # Add the data to the dataset
        if os.path.exists(OUTPUT_CSV):
            add_column(column_name, response)
        else:
            initialize_dataset(column_name, response)
        
        # Display the current state of the dataset
        display_dataset()
        
        # Suggest additional columns
        current_columns = pd.read_csv(OUTPUT_CSV).columns.tolist()
        suggestions = get_suggestions(current_columns)
        print(f"\nSuggestions for additional columns:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        # Clear the command-line argument after processing
        args.request = None
        print("Note: Data may be incomplete or outdated for post-2023 information. Verify with current sources for accuracy.")

if __name__ == "__main__":
    main()