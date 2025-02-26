# Data_Co-pilot

A powerful command-line tool that acts as your AI co-pilot for dataset creation using natural language requests. The tool leverages local large language models (via Ollama) to generate high-quality tabular data and provides automatic data verification, visualization, and insights.

## Features

- 🌍 **Natural Language Dataset Creation**: Request data like "top 10 tech companies" or "market share in percentage" using plain English
- 🔍 **Intelligent Context Awareness**: Automatically detects relationships between columns and understands context
- ✅ **Multi-layer Data Verification**: Combines rule-based and LLM-based verification to ensure high data quality
- 📊 **Automatic Visualization**: Generates interactive HTML dashboards with insights about your dataset
- 💡 **Smart Column Suggestions**: Recommends relevant additional columns based on your current data
- 🛠️ **Robust Error Handling**: Provides clear feedback and recovery options for all operations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Data_Co-pilot.git
cd Data_Co-pilot

# Install required dependencies
pip install pandas ollama matplotlib seaborn numpy
```

## Requirements

- Python 3.7+
- Ollama (with the `deepseek-r1:8b` model installed)
- Pandas
- Additional packages for visualization: matplotlib, seaborn, numpy

## Usage

### Basic Usage

```bash
# Launch the interactive CLI
python Data_Co-pilot.py

# Specify a request directly from the command line
python Data_Co-pilot.py --request "list of top 10 tech companies in USA by revenue"

# Generate a dataset and immediately create visualizations
python Data_Co-pilot.py --finalize --visualize
```

### Example Workflow

1. Start the tool and specify a filename for your dataset
2. Enter your first request (e.g., "list of top 10 tech companies")
3. Confirm or modify the suggested column name
4. Add additional columns by making new requests (e.g., "CEO names", "revenue in billions")
5. Finalize the dataset and generate visualizations
6. The tool will open an HTML dashboard with insights about your data

## How It Works

The tool uses a multi-stage pipeline:

1. **Request Analysis**: Data_Co-pilot analyzes your natural language request to understand what data you need
2. **Data Generation**: It queries the local Ollama LLM with specialized prompts to generate high-quality data
3. **Verification**: Multiple verification layers ensure the data matches expected formats and facts
4. **Correction**: When issues are detected, the tool can automatically correct the data
5. **Visualization**: Creates interactive dashboards with charts relevant to your dataset

## Example Commands

```bash
# Create a dataset of companies
python Data_Co-pilot.py
> Enter filename: tech_companies.csv
> Enter request: List of top 10 tech companies by market cap
> ... (follow the interactive prompts)
> Enter request: CEO names for each company
> ... (follow the interactive prompts)
> Enter request: finalize
```

## Notes

- Data quality depends on the underlying LLM model's knowledge
- For best results, use Ollama with the deepseek-r1:8b model
- The tool may generate visualizations that require significant memory for large datasets

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.