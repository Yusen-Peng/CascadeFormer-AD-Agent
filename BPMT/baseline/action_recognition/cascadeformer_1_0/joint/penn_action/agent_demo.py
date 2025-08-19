import os
import json
import argparse

from agent import process_window

def run_on_single_json(json_path: str):
    """
    Load one skeleton window from a JSON file and run the agent once.
    JSON format: nested list shaped like (T, J, C).
    """
    with open(json_path, "r") as f:
        skel_window = json.load(f)

    out = process_window(skel_window)

    print("=== Demo Run ===")
    print("Event:", json.dumps(out["event"], indent=2))
    print("Scores:", json.dumps(out["scores"], indent=2))
    print("Decision:", json.dumps(out["decision"], indent=2))
    print("Result:", out["result"])


def main():
    parser = argparse.ArgumentParser(description="Run CascadeFormer agent demo")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to a JSON file containing a skeleton window (T,J,C)")
    args = parser.parse_args()

    # Ensure OpenAI key is set if using OpenAIEmbeddings/ChatOpenAI
    if ("OPENAI_API_KEY" not in os.environ):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    run_on_single_json(args.json)


if __name__ == "__main__":
    main()