import os
import pandas as pd
from typing import List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import base64
from io import BytesIO
import numpy as np
from datetime import datetime

def initialize_dataset(output_csv: str, column_name: str, data: List[str]) -> List[str]:
    """
    Initializes the CSV file with the first column.
    """
    try:
        if os.path.exists(output_csv) and not os.access(output_csv, os.W_OK):
            print(f"Permission denied to write to '{output_csv}'.")
            return []
            
        string_data = [str(item) if item != "N/A" else "N/A" for item in data]
        
        df = pd.DataFrame({column_name: string_data})
        df.to_csv(output_csv, index=False)
        print(f"Initialized dataset with column '{column_name}' in '{output_csv}'.")
        return [str(item) if pd.notna(item) and str(item).strip() else "N/A" for item in data]
    except PermissionError as e:
        print(f"Permission error: {e}.")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
    return []

def add_column(output_csv: str, column_name: str, data: List[Any]) -> None:
    """
    Adds a new column to the existing CSV.
    """
    if not os.path.exists(output_csv):
        print("Error: Dataset not initialized. Start with a base column.")
        return
    try:
        if not os.access(output_csv, os.W_OK):
            print(f"Permission denied to write to '{output_csv}'.")
            return
        df = pd.read_csv(output_csv)
        if len(data) != len(df):
            print(f"Error: Data length ({len(data)}) does not match existing rows ({len(df)}).")
            return
        df[column_name] = data
        df.to_csv(output_csv, index=False)
        print(f"Added column '{column_name}' to the dataset in '{output_csv}'.")
    except PermissionError as e:
        print(f"Permission error: {e}.")
    except Exception as e:
        print(f"Error adding column: {e}")

def display_dataset(output_csv: str) -> None:
    """
    Displays the current state of the dataset.
    """
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print("\nCurrent Dataset:")
        print(df.to_string(index=False))
    else:
        print("No dataset initialized yet.")

def generate_visualizations(dataset_path: str) -> str:
    """
    Generates visualizations based on the dataset and saves them to an HTML file.
    """
    try:
        df = pd.read_csv(dataset_path)
        
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
        
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(df.columns) > 0:
            first_col = df.columns[0]
            
            # Bar charts for first column vs numeric columns
            if numeric_cols and not df.empty:
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    # Filter out non-numeric values for plotting
                    plot_df = df[[first_col, num_col]].dropna(subset=[num_col])
                    if not plot_df.empty and len(plot_df) <= 15:  
                        plot_df = plot_df.sort_values(num_col, ascending=False)
                        sns.barplot(x=num_col, y=first_col, data=plot_df, palette='viridis')
                        plt.title(f'{num_col} by {first_col}')
                        plt.tight_layout()
                        
                        img_str = fig_to_base64(plt.gcf())
                        plt.close()
                        
                        html_content += f"""
                            <div class="chart-container">
                                <h2>{num_col} by {first_col}</h2>
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                        """
            
            # Distribution plots for numeric columns
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
            
            # Correlation heatmap
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
        
        output_html = dataset_path.replace('.csv', '_visualizations.html')
        with open(output_html, 'w') as f:
            f.write(html_content)
        
        print(f"\nVisualizations generated and saved to {output_html}")
        
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
        return ""}