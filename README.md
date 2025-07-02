
# Data Co-pilot

Data Co-pilot is a powerful, interactive CLI tool for building datasets from scratch using the power of Large Language Models (LLMs). It functions as a data research agent, allowing you to generate, refine, and verify data through conversational prompts.

This tool supports both local, private LLMs via **Ollama** and powerful, cloud-based models like **Google's Gemini** for high-quality, fact-checked data generation.

## Key Features

- **Interactive Dataset Builder:** Start with a simple prompt (e.g., "top 10 tech companies") and add new columns iteratively.
- **Dual LLM Support:** 
  - **Ollama (Local):** Use for fast, private, offline-capable data generation.
  - **Google Gemini (Cloud):** Leverage for superior reasoning and fact-checking capabilities.
- **Web-Powered Verification:** Use the `--verify-web` flag with Gemini to validate data against real-time Google Search results, ensuring accuracy for factual data like revenue, CEO names, and more.
- **Smart Suggestions:** The agent suggests relevant new columns to help you expand your dataset.
- **Automatic Visualization:** Automatically generate an HTML dashboard with charts and statistics about your dataset upon completion.
- **Modular Architecture:** The new agent-like structure is easy to understand, maintain, and extend.

## How It Works

The tool mimics the workflow of a research agent like Perplexity:

1.  **Intent Analysis:** It analyzes your request to understand the type of data you need.
2.  **Tool Selection:** It uses the appropriate LLM for the job—Ollama for speed and privacy, or Gemini for accuracy and web verification.
3.  **Structured Output:** It prompts the LLMs for clean, JSON-formatted data, which is then parsed directly into a `pandas` DataFrame.
4.  **Interactive Refinement:** It maintains a conversation loop, allowing you to add columns, finalize the dataset, and get intelligent suggestions for what to do next.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SudheerNaraharisetty/Data_Co-pilot.git
cd Data-Co-pilot
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set Up Your LLMs

You need to configure at least one of the following LLMs.

**Option A: Ollama (for Local/Private Use)**

This is the best option for privacy and offline use.

1.  **Install Ollama:** Follow the official instructions at [ollama.ai](https://ollama.ai/).
2.  **Download the Model:** Run the following command in your terminal to download the `deepseek-r1:8b` model:
    ```bash
    ollama pull deepseek-r1:8b
    ```
3.  **Ensure Ollama is Running:** The Ollama application must be running in the background for the script to connect to it.

**Option B: Google Gemini (for Cloud-Powered Features)**

This is required for the highest quality data and for web-based verification.

1.  **Get an API Key:**
    - Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Create a new API key.

2.  **Set the Environment Variable:**
    - **CRITICAL:** Do NOT paste your key into the code. Set it as an environment variable. Open your terminal and run:
      - **Windows (Command Prompt):**
        ```
        setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
        ```
      - **macOS/Linux:**
        ```
        export GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    - **Important:** You must restart your terminal or IDE for this change to take effect.

## Usage

You can run the tool from the root of the project directory using the `python -m` command, which correctly handles the modular structure.

**Start an Interactive Session (Default - Ollama)**

```bash
python -m data_copilot
```

**Start a Session with Gemini**

```bash
python -m data_copilot --use-gemini
```

**One-Shot Request**

You can provide an initial request directly from the command line.

```bash
# Using Ollama
python -m data_copilot --request "List of 5 largest countries by population"

# Using Gemini
python -m data_copilot --use-gemini --request "List of 5 largest countries by population"
```

**Web-Verified Request (Gemini Required)**

This is the most powerful feature. It generates the data and then uses a second web search-powered query to verify and correct each data point.

```bash
# First, create a dataset of companies
python -m data_copilot --use-gemini --request "Top 5 most valuable car companies"

# Now, add a new column and verify it against the web
python -m data_copilot --use-gemini --verify-web --request "CEO of each company"
```

**Finalizing and Visualization**

During an interactive session, type `finalize` to finish.
The tool will then ask if you want to generate a visualization dashboard, which will be saved as an HTML file and opened in your browser.
