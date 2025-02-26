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
    # Clean up the response by removing thinking tags and unnecessary formatting
    response = response.replace("<think>", "").replace("</think>", "").strip()
    
    # First, try to find a numbered list with the pattern "1. Item"
    numbered_list = re.findall(r'(?m)^\s*(\d+)[\.\):]\s*(.+?)\s*
    
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
            clean_line = re.sub(r'^\s*\d+[\.\):]\s*', '', line)
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
        
        # Check if this is a competitor-related query
        is_competitor = any(keyword in prompt.lower() for keyword in
                          ["competitor", "rival", "alternative", "competition"])
        
        if context is not None:
            # This is a follow-up column for existing items
            context_items = context.splitlines()
            num_items = len(context_items)
            expected_count = num_items
            
            # Create a structured, clear prompt
            full_prompt = f"For the following list of {num_items} companies/items:\n\n"
            
            # Add the context items as a clean list
            for i, item in enumerate(context_items):
                # Extract just the item name from the context
                clean_item = re.sub(r'^\s*\d+[.):]\s*', '', item).strip()
                full_prompt += f"{i+1}. {clean_item}\n"
            
            # Different prompt based on data type
            if is_numeric:
                full_prompt += f"\nProvide ONLY the {prompt} as numbers for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS: 
1. Return ONLY numbers, no text or company names.
2. Format as a numbered list (e.g., "1. 250").
3. For large numbers, use appropriate representation (e.g., "394" for $394 billion).
4. If "billions" is in the request, return the number in billions (e.g., "2.7" for 2.7 billion).
5. If unknown, use "N/A".
6. DO NOT repeat the company names.

Example response format:
1. 394
2. 184
3. N/A"""
            elif is_person:
                full_prompt += f"\nProvide the {prompt} for each company/item above. DO NOT repeat the company names."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the person's name, no titles or extra details.
2. Format as a numbered list (e.g., "1. Tim Cook").
3. Include first and last name.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example for CEOs:
1. Tim Cook (for Apple)
2. Satya Nadella (for Microsoft)
3. Andy Jassy (for Amazon)"""
            elif is_competitor:
                full_prompt += f"\nProvide ONE direct {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the competitor name, no explanations.
2. Format as a numbered list (e.g., "1. Samsung").
3. Each response should be a different company than the one in the list.
4. DO NOT simply repeat the company names from the list.
5. If unknown, use "N/A".

Example response:
1. Samsung (competitor of Apple)
2. Google (competitor of Microsoft)
3. Alibaba (competitor of Amazon)"""
            elif is_categorical:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the category/segment, no extra details.
2. Format as a numbered list (e.g., "1. Consumer Electronics").
3. Keep each response to 1-3 words.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. Consumer Electronics
2. Cloud Computing
3. E-Commerce"""
            elif is_date:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the year or date, no extra details.
2. Format as a numbered list (e.g., "1. 1976").
3. For years, provide just the 4-digit year.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. 1976
2. 1975
3. N/A"""
            else:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the exact values, no extra details.
2. Format as a numbered list (e.g., "1. Value").
3. Keep each response short and specific.
4. No explanations or commentary.
5. DO NOT repeat the names from the list above.
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
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
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
    
    # For competitor data, make sure it's not just repeating the original items
    if is_competitor:
        # Check if all competitor values match the original items exactly
        exact_match_count = 0
        for i, item in enumerate(column_data):
            if i < len(initial_items) and item == initial_items[i]:
                exact_match_count += 1
                
        # If more than 50% are exact matches, likely an error
        if exact_match_count > len(column_data) / 2:
            return False, "Competitor column seems to be repeating company names instead of listing actual competitors"
    
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
                if not (isinstance(item, (int, str)) and re.match(r'^\d{4}
    
    # For general text data
    for item in column_data:
        if item != "N/A":
            # Check for location patterns (City, State)
            if re.search(r',\s+[A-Z][a-z]+', str(item)):
                return False, f"Item contains location format: {item}"
            
            # Check for dash followed by text (common in explanations)
            if re.search(r'\s+[-\u2013\u2014]\s+', str(item)):
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
        suggested = re.sub(r'[`"\'\*<>]', '', suggested).strip()  # Added <> to remove <think> tags
        suggested = suggested.split('\n')[0].strip()
        
        # Check if the result looks like a prompt failure or contains debug tokens
        if (len(suggested) > 40 or ' ' in suggested or 
            suggested.startswith(('1.', '-', '•', '*')) or
            'think' in suggested.lower()):
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
    print("\nAnalyzing your request...")
    
    # Check if this is an obvious request related to the first column
    obvious_reference = any(pattern in user_prompt.lower() for pattern in 
                           ["each company", "each item", "companies in", "items in", 
                            "listed", "in our", "in the", "from the", "first column", 
                            "these companies", "those companies"])
    
    # Check for specific column types that are often applied to companies
    common_company_attributes = any(word in user_prompt.lower() for word in 
                                 ["ceo", "founder", "revenue", "market", "industry", 
                                  "competitor", "headquarter", "employee", "founded"])
    
    # If we have a clear reference to the first column or it's a common company attribute,
    # we can potentially skip clarification for context-aware questions
    skip_clarification = (obvious_reference or common_company_attributes) and os.path.exists(OUTPUT_CSV)
    
    if skip_clarification:
        # Load the first column as context
        try:
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                print(f"Detected request related to the existing dataset. Using context from first column.")
                # For CEO and similar requests, we can be even more specific
                if "ceo" in user_prompt.lower() or "founder" in user_prompt.lower():
                    return f"{user_prompt} for the companies in the dataset", False
                return user_prompt, is_numeric
        except Exception:
            # If there's any error, fall back to normal clarification
            skip_clarification = False
    
    if skip_clarification:
        return user_prompt, is_numeric
    
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
                
                # Always show the interpretation to the user for transparency
                print(f"LLM's interpretation: {interpretation}")
                
                if needs_clarification and clarification_question and "none" not in clarification_question.lower():
                    print(f"\nClarification needed: {clarification_question}")
                    user_clarification = input("Your clarification: ").strip()
                    
                    # Combine the original request with the clarification but don't add it to the column name
                    clarified_prompt = f"{user_prompt} based on {user_clarification}"
                    return clarified_prompt, is_numeric
                
                # No clarification needed, confirm interpretation
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
        
        interpretation_match = re.search(r'4\.\s*(.+)', clar_text)
        interpretation = interpretation_match.group(1).strip() if interpretation_match else user_prompt
        
        # Always show the interpretation for transparency
        print(f"LLM's interpretation: {interpretation}")
        
        clarification_question_match = re.search(r'3\.\s*(.+)', clar_text)
        if clarification_question_match:
            clarification_text = clarification_question_match.group(1).strip()
            # Clean up the match to get just the question
            clarification_question = re.sub(r'^\[|\]

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    """
    Main function to run the dataset builder tool.
    """
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
        try:
            if args.request:
                user_request = args.request
            else:
                user_request = input("\nEnter your request or 'finalize' to complete: ").strip()

            if user_request.lower() == "finalize" or args.finalize:
                if os.path.exists(OUTPUT_CSV):
                    print(f"\nDataset finalized and saved as '{OUTPUT_CSV}'.")
                    display_dataset()
                    
                    # Ask about visualizations
                    generate_viz = args.visualize
                    if not generate_viz:
                        viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                        generate_viz = viz_response == 'y'
                    
                    if generate_viz:
                        print("\nGenerating visualizations...")
                        generate_visualizations(OUTPUT_CSV)
                    
                    # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), '', clarification_text)
        else:
            clarification_question = None
        
        if needs_clarification and clarification_question and "none" not in clarification_question.lower():
            print(f"\nClarification needed: {clarification_question}")
            user_clarification = input("Your clarification: ").strip()
            
            # Combine the original request with the clarification
            clarified_prompt = f"{user_prompt} based on {user_clarification}"
            return clarified_prompt, is_numeric
        
        # No clarification needed or couldn't parse correctly
        confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
        if confirmation != 'y':
            new_prompt = input("Please enter your clarified request: ").strip()
            return new_prompt, is_numeric
        
        return user_prompt, is_numeric
        
    except Exception as e:
        print(f"Error clarifying request: {e}.")
        # Fall back to the original prompt
        return user_prompt, is_numeric

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), response)
    
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
        
        # Check if this is a competitor-related query
        is_competitor = any(keyword in prompt.lower() for keyword in
                          ["competitor", "rival", "alternative", "competition"])
        
        if context is not None:
            # This is a follow-up column for existing items
            context_items = context.splitlines()
            num_items = len(context_items)
            expected_count = num_items
            
            # Create a structured, clear prompt
            full_prompt = f"For the following list of {num_items} companies/items:\n\n"
            
            # Add the context items as a clean list
            for i, item in enumerate(context_items):
                # Extract just the item name from the context
                clean_item = re.sub(r'^\s*\d+[.):]\s*', '', item).strip()
                full_prompt += f"{i+1}. {clean_item}\n"
            
            # Different prompt based on data type
            if is_numeric:
                full_prompt += f"\nProvide ONLY the {prompt} as numbers for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS: 
1. Return ONLY numbers, no text or company names.
2. Format as a numbered list (e.g., "1. 250").
3. For large numbers, use appropriate representation (e.g., "394" for $394 billion).
4. If "billions" is in the request, return the number in billions (e.g., "2.7" for 2.7 billion).
5. If unknown, use "N/A".
6. DO NOT repeat the company names.

Example response format:
1. 394
2. 184
3. N/A"""
            elif is_person:
                full_prompt += f"\nProvide the {prompt} for each company/item above. DO NOT repeat the company names."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the person's name, no titles or extra details.
2. Format as a numbered list (e.g., "1. Tim Cook").
3. Include first and last name.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example for CEOs:
1. Tim Cook (for Apple)
2. Satya Nadella (for Microsoft)
3. Andy Jassy (for Amazon)"""
            elif is_competitor:
                full_prompt += f"\nProvide ONE direct {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the competitor name, no explanations.
2. Format as a numbered list (e.g., "1. Samsung").
3. Each response should be a different company than the one in the list.
4. DO NOT simply repeat the company names from the list.
5. If unknown, use "N/A".

Example response:
1. Samsung (competitor of Apple)
2. Google (competitor of Microsoft)
3. Alibaba (competitor of Amazon)"""
            elif is_categorical:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the category/segment, no extra details.
2. Format as a numbered list (e.g., "1. Consumer Electronics").
3. Keep each response to 1-3 words.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. Consumer Electronics
2. Cloud Computing
3. E-Commerce"""
            elif is_date:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the year or date, no extra details.
2. Format as a numbered list (e.g., "1. 1976").
3. For years, provide just the 4-digit year.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. 1976
2. 1975
3. N/A"""
            else:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the exact values, no extra details.
2. Format as a numbered list (e.g., "1. Value").
3. Keep each response short and specific.
4. No explanations or commentary.
5. DO NOT repeat the names from the list above.
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
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
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
    
    # For competitor data, make sure it's not just repeating the original items
    if is_competitor:
        # Check if all competitor values match the original items exactly
        exact_match_count = 0
        for i, item in enumerate(column_data):
            if i < len(initial_items) and item == initial_items[i]:
                exact_match_count += 1
                
        # If more than 50% are exact matches, likely an error
        if exact_match_count > len(column_data) / 2:
            return False, "Competitor column seems to be repeating company names instead of listing actual competitors"
    
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
        suggested = re.sub(r'[`"\'*<>]', '', suggested).strip()  # Added <> to remove <think> tags
        suggested = suggested.split('\n')[0].strip()
        
        # Check if the result looks like a prompt failure or contains debug tokens
        if (len(suggested) > 40 or ' ' in suggested or 
            suggested.startswith(('1.', '-', '•', '*')) or
            'think' in suggested.lower()):
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
    print("\nAnalyzing your request...")
    
    # Check if this is an obvious request related to the first column
    obvious_reference = any(pattern in user_prompt.lower() for pattern in 
                           ["each company", "each item", "companies in", "items in", 
                            "listed", "in our", "in the", "from the", "first column", 
                            "these companies", "those companies"])
    
    # Check for specific column types that are often applied to companies
    common_company_attributes = any(word in user_prompt.lower() for word in 
                                 ["ceo", "founder", "revenue", "market", "industry", 
                                  "competitor", "headquarter", "employee", "founded"])
    
    # If we have a clear reference to the first column or it's a common company attribute,
    # we can potentially skip clarification for context-aware questions
    skip_clarification = (obvious_reference or common_company_attributes) and os.path.exists(OUTPUT_CSV)
    
    if skip_clarification:
        # Load the first column as context
        try:
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                print(f"Detected request related to the existing dataset. Using context from first column.")
                # For CEO and similar requests, we can be even more specific
                if "ceo" in user_prompt.lower() or "founder" in user_prompt.lower():
                    return f"{user_prompt} for the companies in the dataset", False
                return user_prompt, is_numeric
        except Exception:
            # If there's any error, fall back to normal clarification
            skip_clarification = False
    
    if skip_clarification:
        return user_prompt, is_numeric
    
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
                
                # Always show the interpretation to the user for transparency
                print(f"LLM's interpretation: {interpretation}")
                
                if needs_clarification and clarification_question and "none" not in clarification_question.lower():
                    print(f"\nClarification needed: {clarification_question}")
                    user_clarification = input("Your clarification: ").strip()
                    
                    # Combine the original request with the clarification but don't add it to the column name
                    clarified_prompt = f"{user_prompt} based on {user_clarification}"
                    return clarified_prompt, is_numeric
                
                # No clarification needed, confirm interpretation
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
        
        interpretation_match = re.search(r'4\.\s*(.+)', clar_text)
        interpretation = interpretation_match.group(1).strip() if interpretation_match else user_prompt
        
        # Always show the interpretation for transparency
        print(f"LLM's interpretation: {interpretation}")
        
        clarification_question_match = re.search(r'3\.\s*(.+)', clar_text)
        if clarification_question_match:
            clarification_text = clarification_question_match.group(1).strip()
            # Clean up the match to get just the question
            clarification_question = re.sub(r'^\[|\]

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), '', clarification_text)
        else:
            clarification_question = None
        
        if needs_clarification and clarification_question and "none" not in clarification_question.lower():
            print(f"\nClarification needed: {clarification_question}")
            user_clarification = input("Your clarification: ").strip()
            
            # Combine the original request with the clarification
            clarified_prompt = f"{user_prompt} based on {user_clarification}"
            return clarified_prompt, is_numeric
        
        # No clarification needed or couldn't parse correctly
        confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
        if confirmation != 'y':
            new_prompt = input("Please enter your clarified request: ").strip()
            return new_prompt, is_numeric
        
        return user_prompt, is_numeric
        
    except Exception as e:
        print(f"Error clarifying request: {e}.")
        # Fall back to the original prompt
        return user_prompt, is_numeric

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), str(item))):
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
        suggested = re.sub(r'[`"\'*<>]', '', suggested).strip()  # Added <> to remove <think> tags
        suggested = suggested.split('\n')[0].strip()
        
        # Check if the result looks like a prompt failure or contains debug tokens
        if (len(suggested) > 40 or ' ' in suggested or 
            suggested.startswith(('1.', '-', '•', '*')) or
            'think' in suggested.lower()):
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
    print("\nAnalyzing your request...")
    
    # Check if this is an obvious request related to the first column
    obvious_reference = any(pattern in user_prompt.lower() for pattern in 
                           ["each company", "each item", "companies in", "items in", 
                            "listed", "in our", "in the", "from the", "first column", 
                            "these companies", "those companies"])
    
    # Check for specific column types that are often applied to companies
    common_company_attributes = any(word in user_prompt.lower() for word in 
                                 ["ceo", "founder", "revenue", "market", "industry", 
                                  "competitor", "headquarter", "employee", "founded"])
    
    # If we have a clear reference to the first column or it's a common company attribute,
    # we can potentially skip clarification for context-aware questions
    skip_clarification = (obvious_reference or common_company_attributes) and os.path.exists(OUTPUT_CSV)
    
    if skip_clarification:
        # Load the first column as context
        try:
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                print(f"Detected request related to the existing dataset. Using context from first column.")
                # For CEO and similar requests, we can be even more specific
                if "ceo" in user_prompt.lower() or "founder" in user_prompt.lower():
                    return f"{user_prompt} for the companies in the dataset", False
                return user_prompt, is_numeric
        except Exception:
            # If there's any error, fall back to normal clarification
            skip_clarification = False
    
    if skip_clarification:
        return user_prompt, is_numeric
    
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
                
                # Always show the interpretation to the user for transparency
                print(f"LLM's interpretation: {interpretation}")
                
                if needs_clarification and clarification_question and "none" not in clarification_question.lower():
                    print(f"\nClarification needed: {clarification_question}")
                    user_clarification = input("Your clarification: ").strip()
                    
                    # Combine the original request with the clarification but don't add it to the column name
                    clarified_prompt = f"{user_prompt} based on {user_clarification}"
                    return clarified_prompt, is_numeric
                
                # No clarification needed, confirm interpretation
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
        
        interpretation_match = re.search(r'4\.\s*(.+)', clar_text)
        interpretation = interpretation_match.group(1).strip() if interpretation_match else user_prompt
        
        # Always show the interpretation for transparency
        print(f"LLM's interpretation: {interpretation}")
        
        clarification_question_match = re.search(r'3\.\s*(.+)', clar_text)
        if clarification_question_match:
            clarification_text = clarification_question_match.group(1).strip()
            # Clean up the match to get just the question
            clarification_question = re.sub(r'^\[|\]

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), '', clarification_text)
        else:
            clarification_question = None
        
        if needs_clarification and clarification_question and "none" not in clarification_question.lower():
            print(f"\nClarification needed: {clarification_question}")
            user_clarification = input("Your clarification: ").strip()
            
            # Combine the original request with the clarification
            clarified_prompt = f"{user_prompt} based on {user_clarification}"
            return clarified_prompt, is_numeric
        
        # No clarification needed or couldn't parse correctly
        confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
        if confirmation != 'y':
            new_prompt = input("Please enter your clarified request: ").strip()
            return new_prompt, is_numeric
        
        return user_prompt, is_numeric
        
    except Exception as e:
        print(f"Error clarifying request: {e}.")
        # Fall back to the original prompt
        return user_prompt, is_numeric

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), response)
    
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
        
        # Check if this is a competitor-related query
        is_competitor = any(keyword in prompt.lower() for keyword in
                          ["competitor", "rival", "alternative", "competition"])
        
        if context is not None:
            # This is a follow-up column for existing items
            context_items = context.splitlines()
            num_items = len(context_items)
            expected_count = num_items
            
            # Create a structured, clear prompt
            full_prompt = f"For the following list of {num_items} companies/items:\n\n"
            
            # Add the context items as a clean list
            for i, item in enumerate(context_items):
                # Extract just the item name from the context
                clean_item = re.sub(r'^\s*\d+[.):]\s*', '', item).strip()
                full_prompt += f"{i+1}. {clean_item}\n"
            
            # Different prompt based on data type
            if is_numeric:
                full_prompt += f"\nProvide ONLY the {prompt} as numbers for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS: 
1. Return ONLY numbers, no text or company names.
2. Format as a numbered list (e.g., "1. 250").
3. For large numbers, use appropriate representation (e.g., "394" for $394 billion).
4. If "billions" is in the request, return the number in billions (e.g., "2.7" for 2.7 billion).
5. If unknown, use "N/A".
6. DO NOT repeat the company names.

Example response format:
1. 394
2. 184
3. N/A"""
            elif is_person:
                full_prompt += f"\nProvide the {prompt} for each company/item above. DO NOT repeat the company names."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the person's name, no titles or extra details.
2. Format as a numbered list (e.g., "1. Tim Cook").
3. Include first and last name.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example for CEOs:
1. Tim Cook (for Apple)
2. Satya Nadella (for Microsoft)
3. Andy Jassy (for Amazon)"""
            elif is_competitor:
                full_prompt += f"\nProvide ONE direct {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the competitor name, no explanations.
2. Format as a numbered list (e.g., "1. Samsung").
3. Each response should be a different company than the one in the list.
4. DO NOT simply repeat the company names from the list.
5. If unknown, use "N/A".

Example response:
1. Samsung (competitor of Apple)
2. Google (competitor of Microsoft)
3. Alibaba (competitor of Amazon)"""
            elif is_categorical:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the category/segment, no extra details.
2. Format as a numbered list (e.g., "1. Consumer Electronics").
3. Keep each response to 1-3 words.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. Consumer Electronics
2. Cloud Computing
3. E-Commerce"""
            elif is_date:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the year or date, no extra details.
2. Format as a numbered list (e.g., "1. 1976").
3. For years, provide just the 4-digit year.
4. No explanations or commentary.
5. If unknown, use "N/A".
6. DO NOT simply repeat the company names.

Example response format:
1. 1976
2. 1975
3. N/A"""
            else:
                full_prompt += f"\nProvide the {prompt} for each company/item above."
                full_prompt += """
                
CRITICAL INSTRUCTIONS:
1. Return ONLY the exact values, no extra details.
2. Format as a numbered list (e.g., "1. Value").
3. Keep each response short and specific.
4. No explanations or commentary.
5. DO NOT repeat the names from the list above.
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
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
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
    
    # For competitor data, make sure it's not just repeating the original items
    if is_competitor:
        # Check if all competitor values match the original items exactly
        exact_match_count = 0
        for i, item in enumerate(column_data):
            if i < len(initial_items) and item == initial_items[i]:
                exact_match_count += 1
                
        # If more than 50% are exact matches, likely an error
        if exact_match_count > len(column_data) / 2:
            return False, "Competitor column seems to be repeating company names instead of listing actual competitors"
    
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
        suggested = re.sub(r'[`"\'*<>]', '', suggested).strip()  # Added <> to remove <think> tags
        suggested = suggested.split('\n')[0].strip()
        
        # Check if the result looks like a prompt failure or contains debug tokens
        if (len(suggested) > 40 or ' ' in suggested or 
            suggested.startswith(('1.', '-', '•', '*')) or
            'think' in suggested.lower()):
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
    print("\nAnalyzing your request...")
    
    # Check if this is an obvious request related to the first column
    obvious_reference = any(pattern in user_prompt.lower() for pattern in 
                           ["each company", "each item", "companies in", "items in", 
                            "listed", "in our", "in the", "from the", "first column", 
                            "these companies", "those companies"])
    
    # Check for specific column types that are often applied to companies
    common_company_attributes = any(word in user_prompt.lower() for word in 
                                 ["ceo", "founder", "revenue", "market", "industry", 
                                  "competitor", "headquarter", "employee", "founded"])
    
    # If we have a clear reference to the first column or it's a common company attribute,
    # we can potentially skip clarification for context-aware questions
    skip_clarification = (obvious_reference or common_company_attributes) and os.path.exists(OUTPUT_CSV)
    
    if skip_clarification:
        # Load the first column as context
        try:
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                print(f"Detected request related to the existing dataset. Using context from first column.")
                # For CEO and similar requests, we can be even more specific
                if "ceo" in user_prompt.lower() or "founder" in user_prompt.lower():
                    return f"{user_prompt} for the companies in the dataset", False
                return user_prompt, is_numeric
        except Exception:
            # If there's any error, fall back to normal clarification
            skip_clarification = False
    
    if skip_clarification:
        return user_prompt, is_numeric
    
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
                
                # Always show the interpretation to the user for transparency
                print(f"LLM's interpretation: {interpretation}")
                
                if needs_clarification and clarification_question and "none" not in clarification_question.lower():
                    print(f"\nClarification needed: {clarification_question}")
                    user_clarification = input("Your clarification: ").strip()
                    
                    # Combine the original request with the clarification but don't add it to the column name
                    clarified_prompt = f"{user_prompt} based on {user_clarification}"
                    return clarified_prompt, is_numeric
                
                # No clarification needed, confirm interpretation
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
        
        interpretation_match = re.search(r'4\.\s*(.+)', clar_text)
        interpretation = interpretation_match.group(1).strip() if interpretation_match else user_prompt
        
        # Always show the interpretation for transparency
        print(f"LLM's interpretation: {interpretation}")
        
        clarification_question_match = re.search(r'3\.\s*(.+)', clar_text)
        if clarification_question_match:
            clarification_text = clarification_question_match.group(1).strip()
            # Clean up the match to get just the question
            clarification_question = re.sub(r'^\[|\]

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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
    main(), '', clarification_text)
        else:
            clarification_question = None
        
        if needs_clarification and clarification_question and "none" not in clarification_question.lower():
            print(f"\nClarification needed: {clarification_question}")
            user_clarification = input("Your clarification: ").strip()
            
            # Combine the original request with the clarification
            clarified_prompt = f"{user_prompt} based on {user_clarification}"
            return clarified_prompt, is_numeric
        
        # No clarification needed or couldn't parse correctly
        confirmation = input("Is this interpretation correct? (y/n): ").strip().lower()
        if confirmation != 'y':
            new_prompt = input("Please enter your clarified request: ").strip()
            return new_prompt, is_numeric
        
        return user_prompt, is_numeric
        
    except Exception as e:
        print(f"Error clarifying request: {e}.")
        # Fall back to the original prompt
        return user_prompt, is_numeric

def verify_data_with_llm(column_name: str, first_column_data: List[str], new_data: List) -> Tuple[bool, str, List]:
    """
    Uses the LLM to verify if the dataset contains correct information.
    Returns a tuple of (is_valid, reason, corrected_data).
    """
    # Prepare the context for LLM verification
    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
    
    # Prepare data for verification
    data_to_verify = "\n".join([f"{i+1}. {item}" for i, item in enumerate(new_data)])
    
    # Enhance the verification for certain column types
    is_person = any(keyword in column_name.lower() for keyword in 
                  ["ceo", "founder", "president", "director", "head", "chief", "leader"])
    
    is_competitor = any(keyword in column_name.lower() for keyword in
                      ["competitor", "rival", "alternative"])
    
    # Special checks
    special_check = ""
    if is_person:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that each CEO name is an actual person, not just the company name repeated
- Correct CEO examples: Tim Cook, Satya Nadella, Andy Jassy
- INCORRECT CEO examples: "Apple", "Microsoft", "Amazon" (these are company names, not CEOs)
"""
    elif is_competitor:
        special_check = """
IMPORTANT ADDITIONAL CHECKS:
- Verify that competitors are not the same as the companies they're competing with
- For example, "Apple" cannot be listed as a competitor of "Apple"
- Each competitor should be a different company in the same industry
"""
    
    # Create verification prompt
    prompt = f"""Verify the accuracy of this data for column '{column_name}':

First column items (companies/items):
{context}

Generated data for '{column_name}':
{data_to_verify}

{special_check}

TASK: Check if the data for '{column_name}' is accurate and appropriate. 
1. Is each item factually correct based on your knowledge?
2. Is each item properly formatted?
3. Is any item clearly wrong, nonsensical, or a clear repetition of the company name?
4. If there are errors, provide corrected data.

Format your response as JSON:
{{
  "is_valid": false,
  "issues_found": "Description of any issues found, ESPECIALLY if company names are being repeated",
  "corrected_data": ["Corrected item 1", "Corrected item 2", ...] (provide full corrected dataset with accurate information)
}}
"""
    
    try:
        # Get verification response
        verification_response = ollama_client.chat(model=MODEL_NAME, 
                                                  messages=[{"role": "user", "content": prompt}])
        verification_text = verification_response["message"]["content"].strip()
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', verification_text)
        if json_match:
            try:
                analysis = json.loads(json_match.group(1))
                is_valid = analysis.get("is_valid", False)
                issues = analysis.get("issues_found", "Unknown issues")
                corrected_data = analysis.get("corrected_data", [])
                
                # Special check for CEO or similar person columns to catch repetition
                if is_person:
                    repetition_count = 0
                    for i, (company, person) in enumerate(zip(first_column_data, new_data)):
                        if company.strip().lower() == person.strip().lower():
                            repetition_count += 1
                    
                    # If more than 30% of the data is just repeating company names, force a correction
                    if repetition_count > len(new_data) * 0.3:
                        is_valid = False
                        issues = f"Company names are being repeated as {column_name} values. Needs correction."
                        
                        # Force a detailed correction attempt if needed
                        if not corrected_data or len(corrected_data) != len(new_data):
                            print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                            return handle_repetition_error(column_name, first_column_data, new_data)
                
                if not is_valid and corrected_data and len(corrected_data) == len(new_data):
                    return False, issues, corrected_data
                
                return is_valid, issues, []
                
            except json.JSONDecodeError:
                pass
        
        # Manual check for common error patterns if JSON parsing fails
        if verification_text:
            # Check for company name repetition
            if is_person:
                repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                           if company.strip().lower() == person.strip().lower())
                
                if repetition_count > len(new_data) * 0.3:
                    print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                    return handle_repetition_error(column_name, first_column_data, new_data)
        
        # If we reach here, the verification was inconclusive
        return True, "Verification inconclusive, proceeding with data", []
        
    except Exception as e:
        print(f"Error verifying data with LLM: {e}")
        # Special handling for person columns to make sure we don't have company name repetition
        if is_person:
            repetition_count = sum(1 for i, (company, person) in enumerate(zip(first_column_data, new_data)) 
                       if company.strip().lower() == person.strip().lower())
            
            if repetition_count > len(new_data) * 0.3:
                print(f"Critical issue detected: Company names being repeated as {column_name}. Requesting specific corrections...")
                return handle_repetition_error(column_name, first_column_data, new_data)
                
        return True, "Verification skipped due to error", []

def handle_repetition_error(column_name: str, companies: List[str], incorrect_data: List) -> Tuple[bool, str, List]:
    """
    Special handler for when company names are being repeated in columns that should have different data.
    Makes a focused request to get the correct information.
    """
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""URGENT CORRECTION NEEDED: The data for column '{column_name}' incorrectly contains company names.

Here are the companies:
{company_list}

For each company above, I need the ACTUAL {column_name.upper()}, not the company name repeated.
For example, if the column is 'CEO', I need the actual CEO name (like 'Tim Cook' for Apple), not just 'Apple' repeated.

Return ONLY a numbered list with the correct information for each company:
1. [Correct info for company 1]
2. [Correct info for company 2]
...

NO EXPLANATIONS, just the numbered list of correct data.
"""
    
    try:
        # Get correction response
        correction_response = ollama_client.chat(model=MODEL_NAME, 
                                               messages=[{"role": "user", "content": prompt}])
        correction_text = correction_response["message"]["content"].strip()
        
        # Extract the corrected data
        corrected_items = extract_numbered_list(correction_text, expected_count=len(companies))
        
        # Verify that the correction isn't still just repeating company names
        repetition_count = 0
        for company, corrected in zip(companies, corrected_items):
            if company.strip().lower() == corrected.strip().lower():
                repetition_count += 1
        
        if repetition_count > len(companies) * 0.3:
            # Still having repetition issues, make one more focused attempt
            return make_final_correction_attempt(column_name, companies)
            
        return False, f"Corrected {column_name} data to use actual values instead of repeating company names", corrected_items
    
    except Exception as e:
        print(f"Error handling repetition correction: {e}")
        # Last resort attempt
        return make_final_correction_attempt(column_name, companies)

def make_final_correction_attempt(column_name: str, companies: List[str]) -> Tuple[bool, str, List]:
    """
    Last-resort attempt for correction using examples and more explicit instructions.
    """
    # Provide clear examples based on column type
    examples = ""
    if "ceo" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Tim Cook
- Microsoft → Satya Nadella
- Amazon → Andy Jassy
- Google/Alphabet → Sundar Pichai
- Facebook/Meta → Mark Zuckerberg
"""
    elif "founder" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Steve Jobs, Steve Wozniak
- Microsoft → Bill Gates, Paul Allen
- Amazon → Jeff Bezos
- Google → Larry Page, Sergey Brin
- Facebook → Mark Zuckerberg
"""
    elif "competitor" in column_name.lower():
        examples = """
EXAMPLES:
- Apple → Samsung
- Microsoft → Google
- Amazon → Walmart
- Google → Microsoft
- Facebook → TikTok
"""
    
    company_list = "\n".join([f"{i+1}. {company}" for i, company in enumerate(companies)])
    
    prompt = f"""FINAL CORRECTION ATTEMPT: The data for '{column_name}' must be fixed.

Companies:
{company_list}

{examples}

For each company number, provide ONLY the correct {column_name} information.
DO NOT repeat the company name as the answer.
If unsure about any company, write "Unknown" rather than guessing or repeating the company name.

FORMAT:
1. [Correct info for company 1]
2. [Correct info for company 2]
...
"""
    
    try:
        # Get final correction
        final_response = ollama_client.chat(model=MODEL_NAME, 
                                           messages=[{"role": "user", "content": prompt}])
        final_text = final_response["message"]["content"].strip()
        
        # Extract the corrected data
        final_items = extract_numbered_list(final_text, expected_count=len(companies))
        
        return False, f"Final correction applied to {column_name} data", final_items
    
    except Exception as e:
        print(f"Error in final correction attempt: {e}")
        # Generate placeholder data as a last resort
        if "ceo" in column_name.lower() or "founder" in column_name.lower() or any(k in column_name.lower() for k in ["president", "director", "chief"]):
            return False, "Generated placeholder CEO names", [f"CEO of {company}" for company in companies]
        elif "competitor" in column_name.lower():
            return False, "Generated placeholder competitor names", [f"Competitor of {company}" for company in companies]
        else:
            return False, "Could not correct data", [f"Data for {company}" for company in companies]

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

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    Returns the path to the HTML file.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
        import numpy as np
        from datetime import datetime
        
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Create HTML for the dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Visualization Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: white; }}
                h1, h2 {{ color: #333; }}
                .stats-container {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dataset Visualization Dashboard</h1>
            <h3>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            
            <div class="stats-container">
                <h2>Dataset Summary</h2>
                <p>Total rows: {len(df)}</p>
                <p>Total columns: {len(df.columns)}</p>
            </div>
        """
        
        # Add descriptive statistics
        html_content += """
            <div class="stats-container">
                <h2>Descriptive Statistics</h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Non-Null</th>
                        <th>Unique Values</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean (if numeric)</th>
                    </tr>
        """
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            
            # Calculate min, max, mean for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min() if not df[col].empty else "N/A"
                max_val = df[col].max() if not df[col].empty else "N/A"
                mean_val = round(df[col].mean(), 2) if not df[col].empty else "N/A"
            else:
                min_val = "N/A"
                max_val = "N/A"
                mean_val = "N/A"
            
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{col_type}</td>
                    <td>{non_null}</td>
                    <td>{unique}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="dashboard">
        """
        
        # Function to convert a matplotlib figure to a base64 encoded string
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 1. Create bar charts for the first column (usually categories)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # If there are numeric columns, create bar charts comparing categories
            if numeric_cols:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    if len(df) <= 15:  # Only create horizontal bar chart for manageable number of categories
                        ax = df.sort_values(num_col, ascending=False).plot(
                            kind='barh', x=first_col, y=num_col, color='skyblue', legend=False
                        )
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        # Convert plot to base64 image
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Create a distribution plot for each numeric column
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[num_col].dropna(), kde=True)
                plt.title(f'Distribution of {num_col}')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Distribution of {num_col}</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
            
            # Create correlation heatmap if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                           mask=mask, cbar_kws={'shrink': .8})
                plt.title('Correlation Between Numeric Columns')
                plt.tight_layout()
                
                img_str = fig_to_base64(plt.gcf())
                plt.close()
                
                html_content += f"""
                    <div class="chart-container">
                        <h2>Correlation Heatmap</h2>
                        <img src="data:image/png;base64,{img_str}" width="100%">
                    </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
        # Try to open the HTML file in the default browser
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
            print("Opened visualizations in your default web browser.")
        except:
            print("Couldn't open the browser automatically. Please open the HTML file manually.")
        
        return output_html
        
    except ImportError as e:
        print(f"\nVisualization generation requires additional packages: {e}")
        print("Please install matplotlib, seaborn, and numpy to enable visualizations.")
        return ""
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using Ollama locally.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
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
                
                # Ask about visualizations
                generate_viz = args.visualize
                if not generate_viz:
                    viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                    generate_viz = viz_response == 'y'
                
                if generate_viz:
                    print("\nGenerating visualizations...")
                    generate_visualizations(OUTPUT_CSV)
                
                # Ask about renaming
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
        
        # Get the first column data for LLM verification
        first_column_data = []
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            if not df.empty:
                first_col = df.columns[0]
                first_column_data = df[first_col].tolist()
        else:
            first_column_data = response
        
        # Verify data with LLM if not the first column
        if os.path.exists(OUTPUT_CSV):
            print("\nVerifying data accuracy with LLM...")
            llm_verified, issues, corrected_data = verify_data_with_llm(column_name, first_column_data, response)
            
            if not llm_verified:
                print(f"LLM Verification: Issues found - {issues}")
                if corrected_data:
                    print("Applying LLM corrections to the data...")
                    response = corrected_data
                else:
                    print("No corrections provided, proceeding with original data.")
            else:
                print(f"LLM Verification: Data looks good - {issues}")
        
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