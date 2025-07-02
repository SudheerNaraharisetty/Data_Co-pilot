import re
from typing import Optional, List, Union, Any

def extract_numbered_list(response: str, expected_count: Optional[int] = None) -> List[str]:
    """
    Extracts list items formatted as a numbered list from the response.
    Improved to handle various list formats and cleaner extraction.
    """
    # Clean up the response by removing thinking tags and unnecessary formatting
    response = response.replace("<think>", "").replace("</think>", "").strip()
    
    # First, try to find a numbered list with the pattern "1. Item"
    numbered_list = re.findall(r'(?m)^\s*(\d+)[\.\):]\s*(.+?)\s*$', response)
    
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
        number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*$', text) # Only match at end of string
        if number_match:
            clean_num = number_match.group(1).replace(',', '')
            if float(clean_num) > 100:  # Likely to be a raw value, not already in billions
                return float(clean_num)
            
    # Look for plain numbers, including with commas
    number_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)(?!\s*(?:trillion|T|billion|B|million|M))', text, re.IGNORECASE)
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
    Parses the LLM response to extract clean data.
    For numeric data, extracts only the numbers.
    For text data, extracts clean item names.
    """
    if not response:
        print("Warning: Empty response from LLM.")
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
