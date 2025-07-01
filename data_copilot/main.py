
import argparse
from . import cli

def main():
    parser = argparse.ArgumentParser(description="CLI tool to build a universal dataset using local and cloud-based LLMs.")
    parser.add_argument("--request", help="Request a new column (e.g., 'list of top 10 tech companies')")
    parser.add_argument("--finalize", action="store_true", help="Finalize and generate the dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after finalizing")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini for data generation (requires GEMINI_API_KEY)")

    args = parser.parse_args()

    cli.start_session(args)

if __name__ == "__main__":
    main()
