import os
import re
import pandas as pd
from . import data_manager
from . import llm_handler

OUTPUT_CSV: str = ""

def start_session(args):
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
                    data_manager.display_dataset(OUTPUT_CSV)
                    
                    generate_viz = args.visualize
                    if not generate_viz:
                        viz_response = input("Would you like to generate visualizations for this dataset? (y/n): ").strip().lower()
                        generate_viz = viz_response == 'y'
                    
                    if generate_viz:
                        print("\nGenerating visualizations...")
                        data_manager.generate_visualizations(OUTPUT_CSV)
                    
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

            is_numeric = any(keyword in user_request.lower() for keyword in 
                            ["revenue", "sales", "profit", "price", "cost", "amount", "number", 
                             "count", "total", "size", "year", "age", "employees", "market cap"])
            
            expected_count = None
            count_match = re.search(r'(?:top|best)\s+(\d+)', user_request.lower())
            if count_match:
                expected_count = int(count_match.group(1))
            
            context = None
            first_column_data = []
            if os.path.exists(OUTPUT_CSV):
                df = pd.read_csv(OUTPUT_CSV)
                if not df.empty:
                    first_col = df.columns[0]
                    first_column_data = df[first_col].tolist()
                    context = "\n".join([f"{i+1}. {item}" for i, item in enumerate(first_column_data)])
            
            if args.use_gemini:
                response = llm_handler.query_gemini_for_data(user_request, is_numeric=is_numeric, expected_count=expected_count)
            else:
                response = llm_handler.query_ollama(user_request, context=context, is_numeric=is_numeric, expected_count=expected_count)

            if not response:
                print("Failed to get data. Try again or check the LLM server.")
                continue

            if args.verify_web and args.use_gemini and context:
                print("\nVerifying data with web search...")
                verified_response = []
                for i, item in enumerate(response):
                    company = first_column_data[i]
                    verified_item = llm_handler.web_search_and_verify(company, f"{user_request} for {company}")
                    print(f"  - {company}: {item} -> {verified_item}")
                    verified_response.append(verified_item)
                response = verified_response
            
            column_name = llm_handler.suggest_column_name(user_request)
            column_name = llm_handler.confirm_column_name(column_name)

            if os.path.exists(OUTPUT_CSV):
                data_manager.add_column(OUTPUT_CSV, column_name, response)
            else:
                data_manager.initialize_dataset(OUTPUT_CSV, column_name, response)
            
            data_manager.display_dataset(OUTPUT_CSV)
            
            current_columns = pd.read_csv(OUTPUT_CSV).columns.tolist()
            suggestions = llm_handler.get_suggestions(current_columns)
            print(f"\nSuggestions for additional columns:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
            
            args.request = None

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")